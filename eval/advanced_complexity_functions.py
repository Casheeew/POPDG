import numpy as np
import torch
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/hyhan/project/MotionLCM_ControlNet_v2')
from mld.data.humanml.scripts.motion_process import recover_from_ric
from complexity_functions_revised import (
    JOINT_NAMES, FOOT_JOINTS, LIMB_JOINTS, UPPER_BODY_JOINTS, 
    LOWER_BODY_JOINTS, BILATERAL_PAIRS, simple_savgol_filter
)

def extract_advanced_frequency_features(signal_data, fps=30):
    """Extract advanced frequency domain features"""
    if len(signal_data) < 8:
        return {
            'dominant_freq': 0, 'spectral_centroid': 0, 'spectral_bandwidth': 0,
            'spectral_rolloff': 0, 'spectral_flux': 0, 'harmonic_ratio': 0
        }
    
    # Apply FFT
    fft_data = np.abs(fft(signal_data - np.mean(signal_data)))
    freqs = fftfreq(len(signal_data), 1/fps)
    
    # Only consider positive frequencies
    pos_mask = freqs > 0
    fft_data = fft_data[pos_mask]
    freqs = freqs[pos_mask]
    
    if len(fft_data) == 0:
        return {
            'dominant_freq': 0, 'spectral_centroid': 0, 'spectral_bandwidth': 0,
            'spectral_rolloff': 0, 'spectral_flux': 0, 'harmonic_ratio': 0
        }
    
    # Normalize spectrum
    fft_data = fft_data / (np.sum(fft_data) + 1e-8)
    
    # 1. Dominant frequency
    dominant_freq = freqs[np.argmax(fft_data)]
    
    # 2. Spectral centroid
    spectral_centroid = np.sum(freqs * fft_data)
    
    # 3. Spectral bandwidth
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_data))
    
    # 4. Spectral rolloff (95% of energy)
    cumsum_spectrum = np.cumsum(fft_data)
    rolloff_idx = np.where(cumsum_spectrum >= 0.95)[0]
    spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    # 5. Spectral flux (measure of spectral change over time)
    if len(signal_data) >= 16:
        half_len = len(signal_data) // 2
        fft1 = np.abs(fft(signal_data[:half_len] - np.mean(signal_data[:half_len])))
        fft2 = np.abs(fft(signal_data[half_len:] - np.mean(signal_data[half_len:])))
        min_len = min(len(fft1), len(fft2))
        spectral_flux = np.mean(np.abs(fft1[:min_len] - fft2[:min_len]))
    else:
        spectral_flux = 0
    
    # 6. Harmonic ratio (energy in harmonics vs noise)
    if dominant_freq > 0 and len(freqs) > 10:
        harmonic_freqs = [dominant_freq * i for i in range(2, 5) if dominant_freq * i < freqs[-1]]
        harmonic_energy = 0
        for hf in harmonic_freqs:
            closest_idx = np.argmin(np.abs(freqs - hf))
            harmonic_energy += fft_data[closest_idx]
        harmonic_ratio = harmonic_energy / (np.sum(fft_data) + 1e-8)
    else:
        harmonic_ratio = 0
    
    return {
        'dominant_freq': float(dominant_freq),
        'spectral_centroid': float(spectral_centroid),
        'spectral_bandwidth': float(spectral_bandwidth),
        'spectral_rolloff': float(spectral_rolloff),
        'spectral_flux': float(spectral_flux),
        'harmonic_ratio': float(harmonic_ratio)
    }

def extract_motion_primitives(joint_positions, fps=30):
    """Extract motion primitive features"""
    if joint_positions.shape[0] < 10:
        return {'primitives_count': 0, 'primitive_transitions': 0, 'primitive_complexity': 0}
    
    # Calculate overall motion energy
    velocities = np.gradient(joint_positions, axis=0) * fps
    motion_energy = np.mean(np.linalg.norm(velocities.reshape(velocities.shape[0], -1), axis=1))
    
    # Segment motion into primitives based on energy changes
    energy_per_frame = np.linalg.norm(velocities.reshape(velocities.shape[0], -1), axis=1)
    
    # Smooth energy signal
    if len(energy_per_frame) >= 5:
        energy_smooth = simple_savgol_filter(energy_per_frame, window_length=min(5, len(energy_per_frame)))
    else:
        energy_smooth = energy_per_frame
    
    # Find peaks and valleys (primitive boundaries)
    if len(energy_smooth) > 3:
        peaks, _ = signal.find_peaks(energy_smooth, height=np.mean(energy_smooth), distance=3)
        valleys, _ = signal.find_peaks(-energy_smooth, height=-np.mean(energy_smooth), distance=3)
        
        primitives_count = len(peaks) + len(valleys)
        primitive_transitions = len(peaks) + len(valleys) - 1 if len(peaks) + len(valleys) > 0 else 0
    else:
        primitives_count = 1
        primitive_transitions = 0
    
    # Primitive complexity (variance in primitive durations)
    if primitives_count > 1:
        boundary_points = sorted(list(peaks) + list(valleys))
        if len(boundary_points) > 1:
            primitive_durations = np.diff([0] + boundary_points + [len(energy_smooth)])
            primitive_complexity = np.var(primitive_durations) / (np.mean(primitive_durations) + 1e-8)
        else:
            primitive_complexity = 0
    else:
        primitive_complexity = 0
    
    return {
        'primitives_count': float(primitives_count),
        'primitive_transitions': float(primitive_transitions),
        'primitive_complexity': float(primitive_complexity)
    }

def extract_coordination_features(joint_positions, fps=30):
    """Extract advanced coordination features"""
    T = joint_positions.shape[0]
    if T < 4:
        return {'phase_coupling': 0, 'coordination_variability': 0, 'limb_synchrony': 0}
    
    # Calculate velocities for different body parts
    upper_vel = np.mean([
        np.linalg.norm(np.gradient(joint_positions[:, j, :], axis=0), axis=1)
        for j in UPPER_BODY_JOINTS
    ], axis=0)
    
    lower_vel = np.mean([
        np.linalg.norm(np.gradient(joint_positions[:, j, :], axis=0), axis=1)
        for j in LOWER_BODY_JOINTS
    ], axis=0)
    
    # Phase coupling analysis
    if len(upper_vel) > 3 and np.std(upper_vel) > 1e-6 and np.std(lower_vel) > 1e-6:
        # Cross-correlation for phase relationship
        correlation = np.correlate(upper_vel - np.mean(upper_vel), 
                                 lower_vel - np.mean(lower_vel), mode='full')
        max_corr_idx = np.argmax(np.abs(correlation))
        phase_coupling = np.abs(correlation[max_corr_idx]) / (np.std(upper_vel) * np.std(lower_vel) * len(upper_vel))
    else:
        phase_coupling = 0
    
    # Coordination variability over time
    window_size = max(3, T // 5)
    coord_variations = []
    
    for i in range(T - window_size + 1):
        window_upper = upper_vel[i:i+window_size]
        window_lower = lower_vel[i:i+window_size]
        
        if np.std(window_upper) > 1e-6 and np.std(window_lower) > 1e-6:
            window_corr = np.corrcoef(window_upper, window_lower)[0, 1]
            coord_variations.append(window_corr if not np.isnan(window_corr) else 0)
    
    coordination_variability = np.std(coord_variations) if coord_variations else 0
    
    # Limb synchrony (bilateral coordination)
    limb_synchronies = []
    for left_idx, right_idx in BILATERAL_PAIRS[:4]:  # Use first 4 pairs
        left_vel = np.linalg.norm(np.gradient(joint_positions[:, left_idx, :], axis=0), axis=1)
        right_vel = np.linalg.norm(np.gradient(joint_positions[:, right_idx, :], axis=0), axis=1)
        
        if np.std(left_vel) > 1e-6 and np.std(right_vel) > 1e-6:
            sync = np.corrcoef(left_vel, right_vel)[0, 1]
            limb_synchronies.append(sync if not np.isnan(sync) else 0)
    
    limb_synchrony = np.mean(np.abs(limb_synchronies)) if limb_synchronies else 0
    
    return {
        'phase_coupling': float(phase_coupling),
        'coordination_variability': float(coordination_variability),
        'limb_synchrony': float(limb_synchrony)
    }

def advanced_C1_foot_movement(joint_positions, foot_contacts, fps=30, **params):
    """Ultra-advanced C1 with maximum discriminative power"""
    T = joint_positions.shape[0]
    
    if T < 4:
        return {'complexity': 0.0, 'components': {}, 'weights': params}
    
    components = {}
    
    # 1. Multi-scale foot dynamics
    foot_pos = joint_positions[:, FOOT_JOINTS, :]
    foot_vel = np.gradient(foot_pos, axis=0) * fps
    foot_accel = np.gradient(foot_vel, axis=0) * fps if T > 2 else np.zeros_like(foot_vel)
    foot_jerk = np.gradient(foot_accel, axis=0) * fps if T > 3 else np.zeros_like(foot_accel)
    
    # Combine all foot movements  
    foot_vel_mag = np.linalg.norm(foot_vel.reshape(T, -1), axis=1)
    if T > 2:
        foot_accel_mag = np.linalg.norm(foot_accel.reshape(T, -1), axis=1)
    else:
        foot_accel_mag = np.zeros(T)
    if T > 3:
        foot_jerk_mag = np.linalg.norm(foot_jerk.reshape(T, -1), axis=1)
    else:
        foot_jerk_mag = np.zeros(T)
    
    # 2. Advanced frequency analysis
    freq_features = extract_advanced_frequency_features(foot_vel_mag, fps)
    components.update({f'foot_freq_{k}': v for k, v in freq_features.items()})
    
    # 3. Motion primitives for feet
    primitive_features = extract_motion_primitives(foot_pos.reshape(T, -1), fps)
    components.update({f'foot_primitive_{k}': v for k, v in primitive_features.items()})
    
    # 4. Advanced contact analysis
    contact_complexity_features = []
    
    for foot_idx in range(4):
        contacts = foot_contacts[:, foot_idx]
        
        # Contact phase analysis
        contact_phases = []
        current_phase = 'swing' if contacts[0] < 0.5 else 'stance'
        phase_duration = 1
        
        for i in range(1, len(contacts)):
            new_phase = 'swing' if contacts[i] < 0.5 else 'stance'
            if new_phase == current_phase:
                phase_duration += 1
            else:
                contact_phases.append((current_phase, phase_duration))
                current_phase = new_phase
                phase_duration = 1
        contact_phases.append((current_phase, phase_duration))
        
        # Phase variability
        stance_durations = [d for phase, d in contact_phases if phase == 'stance']
        swing_durations = [d for phase, d in contact_phases if phase == 'swing']
        
        stance_var = np.var(stance_durations) if len(stance_durations) > 1 else 0
        swing_var = np.var(swing_durations) if len(swing_durations) > 1 else 0
        
        contact_complexity_features.extend([stance_var, swing_var])
    
    components['contact_phase_complexity'] = np.mean(contact_complexity_features)
    
    # 5. Spatial trajectory complexity
    foot_trajectories = foot_pos.reshape(T, -1)
    if T > 5:
        # Calculate trajectory curvature
        traj_vel = np.gradient(foot_trajectories, axis=0)
        traj_accel = np.gradient(traj_vel, axis=0)
        
        # Curvature = |v × a| / |v|^3
        curvatures = []
        for t in range(1, T-1):
            v = traj_vel[t]
            a = traj_accel[t]
            v_mag = np.linalg.norm(v)
            if v_mag > 1e-6:
                # Approximate curvature in high-dim space
                curvature = np.linalg.norm(a - np.dot(a, v) * v / (v_mag**2)) / (v_mag**2)
                curvatures.append(curvature)
        
        components['trajectory_curvature'] = np.mean(curvatures) if curvatures else 0
    else:
        components['trajectory_curvature'] = 0
    
    # 6. Cross-foot coordination
    left_foot_vel = np.mean([np.linalg.norm(foot_vel[:, i, :], axis=1) for i in [0, 2]], axis=0)  # Left ankle + foot
    right_foot_vel = np.mean([np.linalg.norm(foot_vel[:, i, :], axis=1) for i in [1, 3]], axis=0)  # Right ankle + foot
    
    if np.std(left_foot_vel) > 1e-6 and np.std(right_foot_vel) > 1e-6:
        foot_coordination = np.abs(np.corrcoef(left_foot_vel, right_foot_vel)[0, 1])
        components['foot_coordination'] = foot_coordination if not np.isnan(foot_coordination) else 0
    else:
        components['foot_coordination'] = 0
    
    # Ultra-advanced combination with multiple non-linearities
    base_freq = (freq_features['spectral_bandwidth'] + 
                freq_features['spectral_flux'] + 
                freq_features['harmonic_ratio'])
    
    primitive_factor = (primitive_features['primitives_count'] * 
                       primitive_features['primitive_complexity'])
    
    contact_factor = components['contact_phase_complexity']
    spatial_factor = components['trajectory_curvature']
    coord_factor = 1 - components['foot_coordination']  # Lower coordination = higher complexity
    
    # Multi-level non-linear combination
    C1_ultra = (
        base_freq * (1 + 0.5 * np.tanh(primitive_factor)) * 
        (1 + 0.3 * np.log1p(contact_factor)) * 
        (1 + 0.4 * spatial_factor) * 
        (1 + 0.2 * coord_factor)
    )
    
    return {
        'complexity': float(C1_ultra),
        'components': components,
        'weights': params
    }

def advanced_C2_movement_density(joint_positions, fps=30, **params):
    """Ultra-advanced C2 with maximum discriminative power"""
    T = joint_positions.shape[0]
    
    if T < 4:
        return {'complexity': 0.0, 'components': {}, 'weights': params}
    
    components = {}
    
    # 1. Multi-scale density analysis for each limb
    limb_densities = []
    limb_burst_patterns = []
    
    for joint_idx in LIMB_JOINTS:
        joint_pos = joint_positions[:, joint_idx, :]
        joint_vel = np.gradient(joint_pos, axis=0) * fps
        joint_vel_mag = np.linalg.norm(joint_vel, axis=1)
        
        # Multi-scale analysis
        scales = [3, 5, 7] if T >= 7 else [3] if T >= 3 else [T]
        scale_densities = []
        
        for scale in scales:
            if T >= scale:
                local_densities = []
                for i in range(T - scale + 1):
                    window = joint_vel_mag[i:i+scale]
                    local_density = np.std(window) + np.mean(window)
                    local_densities.append(local_density)
                scale_densities.append(np.mean(local_densities))
        
        limb_densities.extend(scale_densities)
        
        # Burst pattern analysis
        if len(joint_vel_mag) > 5:
            threshold = np.percentile(joint_vel_mag, 70)
            burst_mask = joint_vel_mag > threshold
            
            # Find burst segments
            burst_segments = []
            in_burst = False
            burst_start = 0
            
            for i, is_burst in enumerate(burst_mask):
                if is_burst and not in_burst:
                    burst_start = i
                    in_burst = True
                elif not is_burst and in_burst:
                    burst_segments.append(i - burst_start)
                    in_burst = False
            
            if in_burst:
                burst_segments.append(len(burst_mask) - burst_start)
            
            # Burst pattern features
            burst_count = len(burst_segments)
            burst_duration_var = np.var(burst_segments) if len(burst_segments) > 1 else 0
            burst_intensity = np.mean(joint_vel_mag[burst_mask]) if np.any(burst_mask) else 0
            
            limb_burst_patterns.extend([burst_count, burst_duration_var, burst_intensity])
    
    components['limb_density_features'] = np.mean(limb_densities) + np.std(limb_densities)
    components['burst_pattern_complexity'] = np.mean(limb_burst_patterns)
    
    # 2. Global movement flow analysis
    all_limb_vel = []
    for joint_idx in LIMB_JOINTS:
        joint_vel = np.gradient(joint_positions[:, joint_idx, :], axis=0) * fps
        all_limb_vel.append(np.linalg.norm(joint_vel, axis=1))
    
    all_limb_vel = np.array(all_limb_vel).T  # (T, num_limbs)
    
    # Flow complexity (how movement flows between limbs)
    if T > 3:
        flow_transitions = []
        for t in range(T-1):
            current_dominant = np.argmax(all_limb_vel[t])
            next_dominant = np.argmax(all_limb_vel[t+1])
            flow_transitions.append(current_dominant != next_dominant)
        
        components['flow_transitions'] = np.mean(flow_transitions)
    else:
        components['flow_transitions'] = 0
    
    # 3. Advanced frequency analysis of overall movement
    overall_movement = np.mean(all_limb_vel, axis=1)
    freq_features = extract_advanced_frequency_features(overall_movement, fps)
    components.update({f'movement_freq_{k}': v for k, v in freq_features.items()})
    
    # 4. Motion primitives
    primitive_features = extract_motion_primitives(joint_positions[:, LIMB_JOINTS, :].reshape(T, -1), fps)
    components.update({f'movement_primitive_{k}': v for k, v in primitive_features.items()})
    
    # 5. Coordination between limb groups
    coord_features = extract_coordination_features(joint_positions, fps)
    components.update({f'movement_coord_{k}': v for k, v in coord_features.items()})
    
    # Ultra-advanced combination
    density_base = components['limb_density_features'] + components['burst_pattern_complexity']
    flow_factor = components['flow_transitions']
    freq_factor = (freq_features['spectral_bandwidth'] + 
                  freq_features['spectral_flux'] + 
                  freq_features['harmonic_ratio'])
    primitive_factor = (primitive_features['primitives_count'] + 
                       primitive_features['primitive_complexity'])
    coord_factor = (coord_features['coordination_variability'] + 
                   (1 - coord_features['limb_synchrony']))
    
    C2_ultra = (
        density_base * 
        (1 + 0.5 * flow_factor) * 
        (1 + 0.4 * np.tanh(freq_factor)) * 
        (1 + 0.3 * primitive_factor) * 
        (1 + 0.3 * coord_factor)
    )
    
    return {
        'complexity': float(C2_ultra),
        'components': components,
        'weights': params
    }

def advanced_C3_rotation(joint_positions, fps=30, **params):
    """Ultra-advanced C3 with enhanced rotation analysis"""
    T = joint_positions.shape[0]
    
    if T < 4:
        return {'complexity': 0.0, 'components': {}, 'weights': params}
    
    components = {}
    
    # 1. Multi-body rotation analysis with advanced features
    rotation_signals = []
    
    # Pelvis rotation (enhanced)
    left_hip = joint_positions[:, 1, :]
    right_hip = joint_positions[:, 2, :]
    pelvis_yaw = np.arctan2((right_hip - left_hip)[:, 2], (right_hip - left_hip)[:, 0])
    pelvis_yaw = np.unwrap(pelvis_yaw)
    rotation_signals.append(pelvis_yaw)
    
    # Shoulder rotation
    left_shoulder = joint_positions[:, 16, :]
    right_shoulder = joint_positions[:, 17, :]
    shoulder_yaw = np.arctan2((right_shoulder - left_shoulder)[:, 2], (right_shoulder - left_shoulder)[:, 0])
    shoulder_yaw = np.unwrap(shoulder_yaw)
    rotation_signals.append(shoulder_yaw)
    
    # Spine twist (relative rotation)
    spine_twist = pelvis_yaw - shoulder_yaw
    spine_twist = np.unwrap(spine_twist)
    rotation_signals.append(spine_twist)
    
    # Head rotation (if available)
    if joint_positions.shape[1] > 15:  # Check if head joint exists
        head_pos = joint_positions[:, 15, :]  # Head joint
        neck_pos = joint_positions[:, 12, :]  # Neck joint
        head_direction = head_pos - neck_pos
        head_yaw = np.arctan2(head_direction[:, 2], head_direction[:, 0])
        head_yaw = np.unwrap(head_yaw)
        rotation_signals.append(head_yaw)
    
    # 2. Advanced analysis for each rotation signal
    rotation_complexities = []
    
    for i, rotation_signal in enumerate(rotation_signals):
        if T > 1:
            # Velocity and acceleration
            rot_vel = np.gradient(rotation_signal) * fps
            rot_accel = np.gradient(rot_vel) * fps if T > 2 else np.zeros_like(rot_vel)
            
            # Basic features
            vel_complexity = np.std(rot_vel) + np.mean(np.abs(rot_vel))
            accel_complexity = np.std(rot_accel) + np.mean(np.abs(rot_accel))
            
            rotation_complexities.extend([vel_complexity, accel_complexity])
            
            # Advanced frequency analysis
            freq_features = extract_advanced_frequency_features(rotation_signal, fps)
            components.update({f'rot_{i}_freq_{k}': v for k, v in freq_features.items()})
            
            # Direction changes with temporal analysis
            direction_changes = np.sum(np.abs(np.diff(np.sign(rot_vel))))
            components[f'rot_{i}_direction_changes'] = direction_changes / T
            
            # Rotation bursts (sudden rotational movements)
            if len(rot_vel) > 5:
                rot_threshold = np.percentile(np.abs(rot_vel), 75)
                burst_mask = np.abs(rot_vel) > rot_threshold
                burst_ratio = np.mean(burst_mask)
                components[f'rot_{i}_burst_ratio'] = burst_ratio
    
    components['rotation_base_complexity'] = np.mean(rotation_complexities) + np.std(rotation_complexities)
    
    # 3. Cross-rotation coordination
    if len(rotation_signals) > 1:
        cross_correlations = []
        for i in range(len(rotation_signals)):
            for j in range(i+1, len(rotation_signals)):
                if np.std(rotation_signals[i]) > 1e-6 and np.std(rotation_signals[j]) > 1e-6:
                    corr = np.corrcoef(rotation_signals[i], rotation_signals[j])[0, 1]
                    cross_correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        components['rotation_coordination'] = 1 - np.mean(cross_correlations) if cross_correlations else 0
    else:
        components['rotation_coordination'] = 0
    
    # 4. Rotational momentum and dynamics
    pelvis_pos = joint_positions[:, 0, :]
    if T > 1:
        pelvis_vel = np.gradient(pelvis_pos, axis=0) * fps
        
        # Angular momentum approximation
        angular_momenta = []
        center_of_mass = np.mean(pelvis_pos, axis=0)
        
        for t in range(T):
            r = pelvis_pos[t] - center_of_mass
            v = pelvis_vel[t] if t < len(pelvis_vel) else np.zeros(3)
            angular_momentum = np.linalg.norm(np.cross(r, v))
            angular_momenta.append(angular_momentum)
        
        components['angular_momentum_complexity'] = np.std(angular_momenta) + np.mean(angular_momenta)
    
    # 5. Multi-scale rotation analysis
    if T >= 8:
        for scale in [4, 8]:
            if T >= scale:
                scale_complexities = []
                for i in range(T - scale + 1):
                    window_rotation = pelvis_yaw[i:i+scale]
                    window_complexity = np.var(np.gradient(window_rotation))
                    scale_complexities.append(window_complexity)
                components[f'rotation_scale_{scale}'] = np.mean(scale_complexities)
    
    # Ultra-advanced combination with sophisticated weighting
    base_rotation = components['rotation_base_complexity']
    coord_factor = components['rotation_coordination']
    momentum_factor = components.get('angular_momentum_complexity', 0)
    
    # Aggregate frequency features
    freq_keys = [k for k in components.keys() if 'freq_' in k]
    freq_factor = np.mean([components[k] for k in freq_keys]) if freq_keys else 0
    
    # Multi-scale factor
    scale_keys = [k for k in components.keys() if 'scale_' in k]
    scale_factor = np.mean([components[k] for k in scale_keys]) if scale_keys else 0
    
    C3_ultra = (
        base_rotation * 
        (1 + 0.4 * coord_factor) * 
        (1 + 0.3 * np.tanh(momentum_factor)) * 
        (1 + 0.3 * freq_factor) * 
        (1 + 0.2 * scale_factor)
    )
    
    return {
        'complexity': float(C3_ultra),
        'components': components,
        'weights': params
    }

def advanced_C4_coordination(joint_positions, fps=30, **params):
    """Ultra-advanced C4 with comprehensive coordination analysis"""
    T = joint_positions.shape[0]
    
    if T < 4:
        return {'complexity': 0.0, 'components': {}, 'weights': params}
    
    components = {}
    
    # 1. Multi-level coordination analysis
    velocities = np.gradient(joint_positions, axis=0) * fps
    
    # Body part velocities
    upper_vels = [np.linalg.norm(velocities[:, j, :], axis=1) for j in UPPER_BODY_JOINTS]
    lower_vels = [np.linalg.norm(velocities[:, j, :], axis=1) for j in LOWER_BODY_JOINTS]
    
    upper_vel = np.mean(upper_vels, axis=0)
    lower_vel = np.mean(lower_vels, axis=0)
    
    # 2. Advanced coordination features
    coord_features = extract_coordination_features(joint_positions, fps)
    components.update({f'coord_{k}': v for k, v in coord_features.items()})
    
    # 3. Phase relationships between body parts
    if np.std(upper_vel) > 1e-6 and np.std(lower_vel) > 1e-6:
        # Cross-correlation analysis
        correlation = np.correlate(upper_vel - np.mean(upper_vel), 
                                 lower_vel - np.mean(lower_vel), mode='full')
        max_corr = np.max(np.abs(correlation))
        phase_coupling = max_corr / (np.std(upper_vel) * np.std(lower_vel) * len(upper_vel))
        components['phase_coupling_strength'] = phase_coupling
        
        # Phase lag detection
        max_corr_lag = np.argmax(np.abs(correlation)) - len(upper_vel) + 1
        components['phase_lag'] = abs(max_corr_lag) / len(upper_vel)
    else:
        components['phase_coupling_strength'] = 0
        components['phase_lag'] = 0
    
    # 4. Individual limb coordination patterns
    limb_coord_patterns = []
    
    # Arm coordination
    if len(UPPER_BODY_JOINTS) >= 4:
        left_arm_joints = [j for j in UPPER_BODY_JOINTS[:4]]  # Approximate left arm
        right_arm_joints = [j for j in UPPER_BODY_JOINTS[4:]]  # Approximate right arm
        
        if left_arm_joints and right_arm_joints:
            left_arm_vel = np.mean([np.linalg.norm(velocities[:, j, :], axis=1) for j in left_arm_joints], axis=0)
            right_arm_vel = np.mean([np.linalg.norm(velocities[:, j, :], axis=1) for j in right_arm_joints], axis=0)
            
            if np.std(left_arm_vel) > 1e-6 and np.std(right_arm_vel) > 1e-6:
                arm_coord = np.abs(np.corrcoef(left_arm_vel, right_arm_vel)[0, 1])
                limb_coord_patterns.append(arm_coord if not np.isnan(arm_coord) else 0)
    
    # Leg coordination
    if len(LOWER_BODY_JOINTS) >= 4:
        left_leg_joints = [j for j in LOWER_BODY_JOINTS[:4]]  # Approximate left leg
        right_leg_joints = [j for j in LOWER_BODY_JOINTS[4:]]  # Approximate right leg
        
        if left_leg_joints and right_leg_joints:
            left_leg_vel = np.mean([np.linalg.norm(velocities[:, j, :], axis=1) for j in left_leg_joints], axis=0)
            right_leg_vel = np.mean([np.linalg.norm(velocities[:, j, :], axis=1) for j in right_leg_joints], axis=0)
            
            if np.std(left_leg_vel) > 1e-6 and np.std(right_leg_vel) > 1e-6:
                leg_coord = np.abs(np.corrcoef(left_leg_vel, right_leg_vel)[0, 1])
                limb_coord_patterns.append(leg_coord if not np.isnan(leg_coord) else 0)
    
    components['limb_coordination'] = np.mean(limb_coord_patterns) if limb_coord_patterns else 0
    
    # 5. Temporal coordination variability
    window_size = max(5, T // 4)
    coord_variations = []
    
    for i in range(T - window_size + 1):
        window_upper = upper_vel[i:i+window_size]
        window_lower = lower_vel[i:i+window_size]
        
        if np.std(window_upper) > 1e-6 and np.std(window_lower) > 1e-6:
            window_corr = np.corrcoef(window_upper, window_lower)[0, 1]
            coord_variations.append(window_corr if not np.isnan(window_corr) else 0)
    
    components['temporal_coord_variability'] = np.std(coord_variations) if coord_variations else 0
    
    # 6. Frequency-domain coordination analysis
    if T >= 8:
        upper_freq = extract_advanced_frequency_features(upper_vel, fps)
        lower_freq = extract_advanced_frequency_features(lower_vel, fps)
        
        # Frequency coupling (similar dominant frequencies indicate coordination)
        freq_coupling = abs(upper_freq['dominant_freq'] - lower_freq['dominant_freq'])
        components['frequency_coupling'] = freq_coupling
        
        # Spectral similarity
        spectral_sim = abs(upper_freq['spectral_centroid'] - lower_freq['spectral_centroid'])
        components['spectral_similarity'] = spectral_sim
    
    # 7. Motion primitives coordination
    primitive_features = extract_motion_primitives(joint_positions, fps)
    components.update({f'coord_primitive_{k}': v for k, v in primitive_features.items()})
    
    # Ultra-advanced combination for coordination complexity
    base_coord = (1 - components['limb_coordination']) + components['temporal_coord_variability']
    phase_factor = components['phase_coupling_strength'] * (1 + components['phase_lag'])
    freq_factor = components.get('frequency_coupling', 0) + components.get('spectral_similarity', 0)
    primitive_factor = primitive_features['primitive_complexity']
    advanced_coord = coord_features['coordination_variability'] + (1 - coord_features['limb_synchrony'])
    
    C4_ultra = (
        base_coord * 
        (1 + 0.4 * phase_factor) * 
        (1 + 0.3 * np.tanh(freq_factor)) * 
        (1 + 0.2 * primitive_factor) * 
        (1 + 0.3 * advanced_coord)
    )
    
    return {
        'complexity': float(C4_ultra),
        'components': components,
        'weights': params
    }

def advanced_C5_asymmetry(joint_positions, fps=30, **params):
    """Ultra-advanced C5 with comprehensive asymmetry analysis"""
    T = joint_positions.shape[0]
    
    if T < 4:
        return {'complexity': 0.0, 'components': {}, 'weights': params}
    
    components = {}
    velocities = np.gradient(joint_positions, axis=0) * fps
    
    # 1. Multi-scale bilateral asymmetry analysis
    asymmetry_features = []
    
    for left_idx, right_idx in BILATERAL_PAIRS:
        left_pos = joint_positions[:, left_idx, :]
        right_pos = joint_positions[:, right_idx, :]
        left_vel = velocities[:, left_idx, :]
        right_vel = velocities[:, right_idx, :]
        
        # Position asymmetry with multiple scales
        pos_asym_features = []
        for scale in [1, 3, 5] if T >= 5 else [1]:
            if T >= scale:
                # Multi-scale position differences
                for i in range(T - scale + 1):
                    left_window = left_pos[i:i+scale]
                    right_window = right_pos[i:i+scale]
                    
                    # Mirror right side for comparison
                    right_mirrored = right_window.copy()
                    right_mirrored[:, 0] = -right_mirrored[:, 0]  # Flip X
                    
                    pos_diff = np.mean(np.linalg.norm(left_window - right_mirrored, axis=1))
                    pos_asym_features.append(pos_diff)
        
        # Velocity asymmetry
        left_vel_mag = np.linalg.norm(left_vel, axis=1)
        right_vel_mag = np.linalg.norm(right_vel, axis=1)
        vel_asym = np.mean(np.abs(left_vel_mag - right_vel_mag))
        
        # Temporal pattern asymmetry
        if len(left_vel_mag) >= 8:
            left_freq = extract_advanced_frequency_features(left_vel_mag, fps)
            right_freq = extract_advanced_frequency_features(right_vel_mag, fps)
            
            freq_asym = (abs(left_freq['dominant_freq'] - right_freq['dominant_freq']) + 
                        abs(left_freq['spectral_centroid'] - right_freq['spectral_centroid']))
        else:
            freq_asym = 0
        
        # Phase relationship asymmetry
        if np.std(left_vel_mag) > 1e-6 and np.std(right_vel_mag) > 1e-6:
            phase_corr = np.corrcoef(left_vel_mag, right_vel_mag)[0, 1]
            phase_asym = 1 - abs(phase_corr) if not np.isnan(phase_corr) else 1
        else:
            phase_asym = 1
        
        asymmetry_features.extend([
            np.mean(pos_asym_features),
            vel_asym,
            freq_asym,
            phase_asym
        ])
    
    components['bilateral_asymmetry'] = np.mean(asymmetry_features) + np.std(asymmetry_features)
    
    # 2. Global body asymmetry
    left_side_joints = [pair[0] for pair in BILATERAL_PAIRS]
    right_side_joints = [pair[1] for pair in BILATERAL_PAIRS]
    
    left_side_movement = np.mean([
        np.linalg.norm(velocities[:, j, :], axis=1) for j in left_side_joints
    ], axis=0)
    
    right_side_movement = np.mean([
        np.linalg.norm(velocities[:, j, :], axis=1) for j in right_side_joints
    ], axis=0)
    
    # Global asymmetry metrics
    if np.std(left_side_movement) > 1e-6 and np.std(right_side_movement) > 1e-6:
        global_correlation = np.corrcoef(left_side_movement, right_side_movement)[0, 1]
        components['global_asymmetry'] = 1 - abs(global_correlation) if not np.isnan(global_correlation) else 1
    else:
        components['global_asymmetry'] = 1
    
    # 3. Asymmetric motion primitives
    left_primitives = extract_motion_primitives(
        joint_positions[:, left_side_joints, :].reshape(T, -1), fps
    )
    right_primitives = extract_motion_primitives(
        joint_positions[:, right_side_joints, :].reshape(T, -1), fps
    )
    
    primitive_asymmetry = (
        abs(left_primitives['primitives_count'] - right_primitives['primitives_count']) +
        abs(left_primitives['primitive_complexity'] - right_primitives['primitive_complexity'])
    )
    components['primitive_asymmetry'] = primitive_asymmetry
    
    # 4. Dynamic asymmetry (changes over time)
    window_size = max(5, T // 4)
    dynamic_asymmetries = []
    
    for i in range(T - window_size + 1):
        window_left = left_side_movement[i:i+window_size]
        window_right = right_side_movement[i:i+window_size]
        
        window_asym = np.mean(np.abs(window_left - window_right))
        dynamic_asymmetries.append(window_asym)
    
    components['dynamic_asymmetry'] = np.std(dynamic_asymmetries) if dynamic_asymmetries else 0
    
    # 5. Postural asymmetry
    # Center of mass for left and right sides
    left_positions = joint_positions[:, left_side_joints, :]
    right_positions = joint_positions[:, right_side_joints, :]
    
    left_com = np.mean(left_positions, axis=1)  # (T, 3)
    right_com = np.mean(right_positions, axis=1)  # (T, 3)
    
    # Mirror right COM for comparison
    right_com_mirrored = right_com.copy()
    right_com_mirrored[:, 0] = -right_com_mirrored[:, 0]
    
    postural_asym = np.mean(np.linalg.norm(left_com - right_com_mirrored, axis=1))
    components['postural_asymmetry'] = postural_asym
    
    # 6. Frequency-domain asymmetry analysis
    if T >= 8:
        left_freq_features = extract_advanced_frequency_features(left_side_movement, fps)
        right_freq_features = extract_advanced_frequency_features(right_side_movement, fps)
        
        spectral_asymmetry = (
            abs(left_freq_features['spectral_centroid'] - right_freq_features['spectral_centroid']) +
            abs(left_freq_features['spectral_bandwidth'] - right_freq_features['spectral_bandwidth']) +
            abs(left_freq_features['harmonic_ratio'] - right_freq_features['harmonic_ratio'])
        )
        components['spectral_asymmetry'] = spectral_asymmetry
    else:
        components['spectral_asymmetry'] = 0
    
    # Ultra-advanced combination for asymmetry complexity
    bilateral_factor = components['bilateral_asymmetry']
    global_factor = components['global_asymmetry']
    primitive_factor = components['primitive_asymmetry']
    dynamic_factor = components['dynamic_asymmetry']
    postural_factor = components['postural_asymmetry']
    spectral_factor = components['spectral_asymmetry']
    
    C5_ultra = (
        bilateral_factor * 
        (1 + 0.4 * global_factor) * 
        (1 + 0.3 * primitive_factor) * 
        (1 + 0.3 * dynamic_factor) * 
        (1 + 0.2 * postural_factor) * 
        (1 + 0.2 * spectral_factor)
    )
    
    return {
        'complexity': float(C5_ultra),
        'components': components,
        'weights': params
    }

def compute_all_advanced_complexities(joint_positions, foot_contacts, fps=30, params=None):
    """Compute all ultra-advanced complexity metrics"""
    if params is None:
        params = {}
    
    results = {}
    
    # All advanced complexity functions
    complexity_functions = [
        ('C1_advanced', advanced_C1_foot_movement, True),   # Needs foot contacts
        ('C2_advanced', advanced_C2_movement_density, False),
        ('C3_advanced', advanced_C3_rotation, False),
        ('C4_advanced', advanced_C4_coordination, False),
        ('C5_advanced', advanced_C5_asymmetry, False)
    ]
    
    for name, func, needs_contacts in complexity_functions:
        try:
            if needs_contacts:
                results[name] = func(joint_positions, foot_contacts, fps, **params.get(name, {}))
            else:
                results[name] = func(joint_positions, fps, **params.get(name, {}))
        except Exception as e:
            print(f"Error computing {name}: {e}")
            results[name] = {'complexity': 0, 'components': {}, 'weights': {}}
    
    return results

if __name__ == "__main__":
    print("=== Testing ULTRA-ADVANCED Complexity Functions ===")
    
    # Load a sample file
    sample_file = "/home/hyhan/project/MotionLCM_ControlNet_v2/npy_dataset_with_contact/1/gHO_sFM_c01_d19_mHO0_ch01_1-4_P7_segment_2_strategy_1_original.npy"
    
    if Path(sample_file).exists():
        # Load data
        data = np.load(sample_file)
        motion_tensor = torch.from_numpy(data[:, :-4]).float().unsqueeze(0)
        joint_positions = recover_from_ric(motion_tensor, 22).squeeze(0).numpy()
        foot_contacts = data[:, -4:]
        
        print(f"Sample file: {Path(sample_file).name}")
        print(f"Sequence length: {joint_positions.shape[0]} frames")
        
        # Test ultra-advanced complexity functions
        complexities = compute_all_advanced_complexities(joint_positions, foot_contacts)
        
        print(f"\n=== ULTRA-ADVANCED Complexity Results ===")
        for metric_name, result in complexities.items():
            print(f"\n{metric_name}:")
            print(f"  Overall complexity: {result['complexity']:.6f}")
            if 'components' in result:
                print(f"  Number of components: {len(result['components'])}")
                for comp_name, comp_value in list(result['components'].items())[:5]:  # Show first 5
                    if isinstance(comp_value, (int, float)):
                        print(f"  {comp_name}: {comp_value:.6f}")
        
        # Test on simplified version
        simplified_file = sample_file.replace("_original.npy", "_simplified.npy")
        if Path(simplified_file).exists():
            data_simplified = np.load(simplified_file)
            motion_tensor_simplified = torch.from_numpy(data_simplified[:, :-4]).float().unsqueeze(0)
            joint_positions_simplified = recover_from_ric(motion_tensor_simplified, 22).squeeze(0).numpy()
            foot_contacts_simplified = data_simplified[:, -4:]
            
            complexities_simplified = compute_all_advanced_complexities(joint_positions_simplified, foot_contacts_simplified)
            
            print(f"\n=== ULTRA-ADVANCED Complexity Comparison (Original vs Simplified) ===")
            for metric_name in complexities.keys():
                orig_score = complexities[metric_name]['complexity']
                simp_score = complexities_simplified[metric_name]['complexity']
                reduction = (orig_score - simp_score) / orig_score * 100 if orig_score > 0 else 0
                ratio = orig_score / simp_score if simp_score > 0 else float('inf')
                print(f"  {metric_name}: {orig_score:.6f} → {simp_score:.6f}")
                print(f"    Reduction: {reduction:+.1f}%, Ratio: {ratio:.2f}")
        
        print(f"\n{'='*80}")
        print("ULTRA-ADVANCED VERSION: Maximum discriminative power")
        print("- Multi-scale frequency analysis with 6+ spectral features")
        print("- Motion primitive detection and analysis")
        print("- Cross-body coordination with phase coupling")
        print("- Advanced bilateral asymmetry with postural analysis")
        print("- Temporal dynamics and burst pattern detection")
        print("- Multi-level non-linear feature combinations")
        print("="*80)
        
    else:
        print("❌ Sample file not found")