import numpy as np
import torch
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.append('/home/hyhan/project/MotionLCM_ControlNet_v2')
# from mld.data.humanml.scripts.motion_process import recover_from_ric

# Joint indices for HumanML3D skeleton
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 
    'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
]

# Joint groups
FOOT_JOINTS = [7, 8, 10, 11]  # left_ankle, right_ankle, left_foot, right_foot
LIMB_JOINTS = [4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21]  # arms and legs
UPPER_BODY_JOINTS = [13, 14, 16, 17, 18, 19, 20, 21]  # collars, shoulders, elbows, wrists  
LOWER_BODY_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]  # hips, knees, ankles, feet

# Bilateral joint pairs (left, right)
BILATERAL_PAIRS = [
    (1, 2),   # left_hip, right_hip
    (4, 5),   # left_knee, right_knee
    (7, 8),   # left_ankle, right_ankle
    (10, 11), # left_foot, right_foot
    (13, 14), # left_collar, right_collar
    (16, 17), # left_shoulder, right_shoulder
    (18, 19), # left_elbow, right_elbow
    (20, 21)  # left_wrist, right_wrist
]

# Revised parameters based on new formulas
DEFAULT_PARAMS = {
    'C1': {'alpha': 1.5, 'beta': 0.05, 'zeta': 15.0},
    'C2': {'gamma': 0.005},
    'C3': {'gamma': 0.3, 'alpha': 1.0, 'beta': 0.5},  # Weights for total rotation, velocity and acceleration
    'C4': {'delta': 0.01},  # Movement threshold for indicator function
    'C5': {'delta': 0.01, 'lambda_penalty': 0.5}   # Movement threshold and penalty weight
}

def simple_savgol_filter(data, window_length=5, polyorder=2):
    """Simple moving average filter"""
    if len(data.shape) == 1:
        if len(data) < window_length:
            return data
        
        filtered = np.zeros_like(data)
        half_window = window_length // 2
        
        for i in range(len(data)):
            start = max(0, i - half_window)
            end = min(len(data), i + half_window + 1)
            filtered[i] = np.mean(data[start:end])
        
        return filtered
    else:
        filtered = np.zeros_like(data)
        for i in range(data.shape[-1]):
            filtered[..., i] = simple_savgol_filter(data[..., i], window_length, polyorder)
        return filtered

def compute_entropy_improved(data, bins=10):
    """Improved entropy calculation"""
    try:
        positive_data = data[data > 1e-6]
        if len(positive_data) < 2:
            return 0
        
        hist, _ = np.histogram(positive_data, bins=bins)
        hist = hist.astype(float)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        
        if len(hist) <= 1:
            return 0
        
        entropy_value = -np.sum(hist * np.log(hist))
        return entropy_value
    except:
        return 0

def extract_pelvis_yaw_angles(joint_positions):
    """Extract pelvis y-axis rotation (yaw) angles"""
    T = joint_positions.shape[0]
    
    left_hip = joint_positions[:, 1, :]
    right_hip = joint_positions[:, 2, :]
    
    # Calculate hip vector (pointing from left to right hip)
    hip_vector = right_hip - left_hip
    
    # Calculate yaw angles from hip vector
    # Using atan2 to get angle in x-z plane
    yaw_angles = np.arctan2(hip_vector[:, 2], hip_vector[:, 0])
    
    # Unwrap angles to avoid discontinuities at ±π
    yaw_angles = np.unwrap(yaw_angles)
    
    return yaw_angles

def C1_foot_movement_complexity(joint_positions, foot_contacts, fps=30, alpha=1.5, beta=0.05, zeta=15.0):
    """C1: Foot Movement Complexity (unchanged - working excellently)"""
    T = joint_positions.shape[0]
    
    velocities = np.zeros_like(joint_positions)
    velocities[1:] = (joint_positions[1:] - joint_positions[:-1]) * fps
    
    left_foot_vel = np.linalg.norm(velocities[:, 11], axis=1)
    right_foot_vel = np.linalg.norm(velocities[:, 10], axis=1)
    mean_foot_velocity = np.mean(left_foot_vel + right_foot_vel)
    
    foot_vel_combined = np.concatenate([left_foot_vel, right_foot_vel])
    velocity_entropy = compute_entropy_improved(foot_vel_combined)
    
    foot_positions = joint_positions[:, FOOT_JOINTS, :]
    all_foot_pos = foot_positions.reshape(-1, 3)
    
    x_range = np.max(all_foot_pos[:, 0]) - np.min(all_foot_pos[:, 0])
    z_range = np.max(all_foot_pos[:, 2]) - np.min(all_foot_pos[:, 2])
    horizontal_range = np.sqrt(x_range**2 + z_range**2)
    
    contact_transitions = 0
    for foot_idx in range(4):
        transitions = np.sum(np.diff(foot_contacts[:, foot_idx]) != 0)
        contact_transitions += transitions
    contact_transition_rate = contact_transitions / T
    
    C1 = mean_foot_velocity + alpha * velocity_entropy + beta * horizontal_range + zeta * contact_transition_rate
    
    return {
        'complexity': float(C1),
        'components': {
            'mean_foot_velocity': float(mean_foot_velocity),
            'velocity_entropy': float(velocity_entropy),
            'horizontal_range': float(horizontal_range), 
            'contact_transition_rate': float(contact_transition_rate)
        },
        'weights': {'alpha': alpha, 'beta': beta, 'zeta': zeta}
    }

def C2_movement_density(joint_positions, fps=30, gamma=0.005):
    """C2: Movement Density (unchanged - working excellently)"""
    T = joint_positions.shape[0]
    
    velocities = np.zeros_like(joint_positions)
    velocities[1:] = (joint_positions[1:] - joint_positions[:-1]) * fps
    
    normalized_velocity_sum = 0
    for joint_idx in LIMB_JOINTS:
        joint_vel_magnitudes = np.linalg.norm(velocities[:, joint_idx], axis=1)
        sigma_j = np.std(joint_vel_magnitudes)
        if sigma_j > 1e-6:
            normalized_velocity_sum += np.mean(joint_vel_magnitudes / sigma_j)
        else:
            normalized_velocity_sum += np.mean(joint_vel_magnitudes)
    
    acceleration_magnitudes = []
    for joint_idx in LIMB_JOINTS:
        joint_vel_vec = velocities[:, joint_idx]
        smoothed_vel_vec = simple_savgol_filter(joint_vel_vec)
        acceleration_vec = np.zeros_like(smoothed_vel_vec)
        acceleration_vec[1:-1] = smoothed_vel_vec[2:] - 2*smoothed_vel_vec[1:-1] + smoothed_vel_vec[:-2]
        acceleration_vec *= fps**2
        acc_magnitudes = np.linalg.norm(acceleration_vec, axis=1)
        acceleration_magnitudes.extend(acc_magnitudes)
    
    acceleration_median = np.median(acceleration_magnitudes) if acceleration_magnitudes else 0
    C2 = normalized_velocity_sum + gamma * acceleration_median
    
    return {
        'complexity': float(C2),
        'components': {
            'normalized_velocity_sum': float(normalized_velocity_sum),
            'acceleration_median': float(acceleration_median)
        },
        'weights': {'gamma': gamma}
    }

def C3_rotation_complexity(joint_positions, fps=30, alpha=1.0, beta=0.5, gamma=0.3):
    """
    REVISED C3: Rotation Complexity with Normalized Total Rotation
    
    Formula:
    C3 = γ·|θ_T - θ_1|_norm + α·(1/(T-1))·Σ|Δθ_t| + β·(1/(T-2))·Σ|Δ²θ_t|
    
    Where:
    - First term: Normalized total rotation (modulo 2π, then normalized by π)
    - Second term: Mean rotation velocity (how often/strongly rotated)
    - Third term: Mean rotation acceleration (irregularity of rotation)
    - γ: Weight for total rotation term (default 0.3)
    """
    T = joint_positions.shape[0]
    
    # Extract pelvis yaw angles
    yaw_angles = extract_pelvis_yaw_angles(joint_positions)
    
    # 1. Total rotation: |θ_T - θ_1| with modulo 2π normalization
    raw_total_rotation = yaw_angles[-1] - yaw_angles[0]
    
    # Apply modulo 2π to handle multiple full rotations
    # This ensures the value is in [-2π, 2π] range
    total_rotation_mod = np.abs(raw_total_rotation) % (2 * np.pi)
    
    # If rotation is more than π, take the shorter path
    if total_rotation_mod > np.pi:
        total_rotation_mod = 2 * np.pi - total_rotation_mod
    
    # Normalize by π to get value in [0, 1] range
    total_rotation_normalized = total_rotation_mod / np.pi
    
    # 2. Rotation velocity: Δθ_t = θ_t - θ_{t-1}
    if T > 1:
        rotation_velocities = np.diff(yaw_angles)  # Frame-to-frame changes
        mean_rotation_velocity = np.mean(np.abs(rotation_velocities))
    else:
        rotation_velocities = np.array([0])
        mean_rotation_velocity = 0
    
    # 3. Rotation acceleration: Δ²θ_t = Δθ_{t+1} - Δθ_t
    if T > 2:
        rotation_accelerations = np.diff(rotation_velocities)
        mean_rotation_acceleration = np.mean(np.abs(rotation_accelerations))
    else:
        rotation_accelerations = np.array([0])
        mean_rotation_acceleration = 0
    
    # Final complexity score with weighted normalized total rotation
    C3 = gamma * total_rotation_normalized + alpha * mean_rotation_velocity + beta * mean_rotation_acceleration
    
    # Additional statistics for analysis
    # Count direction changes (sign changes in velocity)
    direction_changes = 0
    if len(rotation_velocities) > 1:
        for t in range(1, len(rotation_velocities)):
            if rotation_velocities[t] * rotation_velocities[t-1] < 0:  # Sign change
                direction_changes += 1
    
    # Convert to rates for interpretability
    rotation_velocity_rate = mean_rotation_velocity * fps if T > 1 else 0  # rad/s
    rotation_acceleration_rate = mean_rotation_acceleration * fps**2 if T > 2 else 0  # rad/s²
    
    # Calculate total accumulated rotation (sum of absolute changes)
    accumulated_rotation = np.sum(np.abs(rotation_velocities)) if T > 1 else 0
    
    return {
        'complexity': float(C3),
        'components': {
            'total_rotation_raw': float(np.abs(raw_total_rotation)),
            'total_rotation_normalized': float(total_rotation_normalized),
            'mean_rotation_velocity': float(mean_rotation_velocity),
            'mean_rotation_acceleration': float(mean_rotation_acceleration),
            'rotation_velocity_rate': float(rotation_velocity_rate),
            'rotation_acceleration_rate': float(rotation_acceleration_rate),
            'direction_changes': int(direction_changes),
            'direction_change_rate': float(direction_changes / (T-1)) if T > 1 else 0,
            'accumulated_rotation': float(accumulated_rotation)
        },
        'weights': {'gamma': gamma, 'alpha': alpha, 'beta': beta}
    }

def C4_coordination_revised(joint_positions, fps=30, delta=0.01):
    """
    REVISED C4: Multi-limb Coordination using LaTeX formula
    
    C₄ = Var(I_t^upper - I_t^lower) · I[min(μ_upper, μ_lower) > δ]
    
    Where:
    - I_t^upper, I_t^lower: Frame-wise average velocity (movement intensity)
    - μ_upper, μ_lower: Overall sequence average movement intensity
    - I[·]: Indicator function (1 if condition met, 0 otherwise)
    - δ: Threshold for "not moving" (e.g., 0.01)
    """
    T = joint_positions.shape[0]
    
    # Compute velocities
    velocities = np.zeros_like(joint_positions)
    velocities[1:] = (joint_positions[1:] - joint_positions[:-1]) * fps
    
    # Calculate movement intensities for each frame
    I_t_upper = np.zeros(T)  # Upper body intensity per frame
    I_t_lower = np.zeros(T)  # Lower body intensity per frame
    
    for t in range(T):
        # Upper body average movement intensity at frame t
        upper_movements = [np.linalg.norm(velocities[t, j]) for j in UPPER_BODY_JOINTS]
        I_t_upper[t] = np.mean(upper_movements)
        
        # Lower body average movement intensity at frame t
        lower_movements = [np.linalg.norm(velocities[t, j]) for j in LOWER_BODY_JOINTS]
        I_t_lower[t] = np.mean(lower_movements)
    
    # Calculate overall sequence averages (μ_upper, μ_lower)
    mu_upper = np.mean(I_t_upper)
    mu_lower = np.mean(I_t_lower)
    
    # Indicator function: I[min(μ_upper, μ_lower) > δ]
    indicator = 1 if min(mu_upper, mu_lower) > delta else 0
    
    # Calculate variance of intensity differences
    intensity_diff = I_t_upper - I_t_lower
    variance_diff = np.var(intensity_diff)
    
    # Final formula: C₄ = Var(I_t^upper - I_t^lower) · I[min(μ_upper, μ_lower) > δ]
    C4 = variance_diff * indicator
    
    return {
        'complexity': float(C4),
        'components': {
            'variance_diff': float(variance_diff),
            'indicator': float(indicator),
            'mu_upper': float(mu_upper),
            'mu_lower': float(mu_lower),
            'intensity_diff_mean': float(np.mean(intensity_diff)),
            'intensity_diff_std': float(np.std(intensity_diff)),
            'min_mu': float(min(mu_upper, mu_lower))
        },
        'weights': {'delta': delta}
    }

def C5_asymmetry_revised(joint_positions, fps=30, delta=0.01, lambda_penalty=0.5):
    """
    REVISED C5: Left-Right Asymmetry with One-sided Penalty
    
    Formula:
    - A_t = Σ w_j * (|‖v[j_L]‖ - ‖v[j_R]‖| + ‖p[j_L] - p[j_R]‖)
    - C5_asym = (1/T) * Σ A_t
    - B = min(V_L, V_R) / (max(V_L, V_R) + ε)
    - P_bias = I[B < δ]  (penalty when one side is inactive)
    - C5 = C5_asym * (1 + λ * P_bias)
    
    한쪽만 움직이는 동작 = 복잡 (penalty 적용)
    """
    T = joint_positions.shape[0]
    epsilon = 1e-8
    
    # Compute velocities
    velocities = np.zeros_like(joint_positions)
    velocities[1:] = (joint_positions[1:] - joint_positions[:-1]) * fps
    
    # Frame-wise asymmetry calculation
    A_t = np.zeros(T)
    
    # Collect per-pair statistics for bias calculation
    left_activities = []
    right_activities = []
    
    for pair_idx, (left_idx, right_idx) in enumerate(BILATERAL_PAIRS):
        # Weight based on joint importance (higher for extremities)
        if pair_idx in [3, 7]:  # foot, wrist
            w_j = 1.5
        elif pair_idx in [2, 6]:  # ankle, elbow
            w_j = 1.2
        else:
            w_j = 1.0
        
        # Get positions and velocities
        left_pos = joint_positions[:, left_idx]
        right_pos = joint_positions[:, right_idx]
        left_vel = velocities[:, left_idx]
        right_vel = velocities[:, right_idx]
        
        # Calculate velocity magnitude difference
        left_vel_mag = np.linalg.norm(left_vel, axis=1)
        right_vel_mag = np.linalg.norm(right_vel, axis=1)
        vel_diff = np.abs(left_vel_mag - right_vel_mag)
        
        # Calculate position difference (relative to pelvis)
        pelvis_pos = joint_positions[:, 0:1]  # Keep dims for broadcasting
        left_rel_pos = left_pos - pelvis_pos[:, 0]
        right_rel_pos = right_pos - pelvis_pos[:, 0]
        
        # Mirror right position for comparison
        right_rel_pos_mirrored = right_rel_pos.copy()
        right_rel_pos_mirrored[:, 0] = -right_rel_pos_mirrored[:, 0]  # Flip X
        
        # Position difference
        pos_diff = np.linalg.norm(left_rel_pos - right_rel_pos_mirrored, axis=1)
        
        # Add to frame-wise asymmetry
        A_t += w_j * (vel_diff + 0.5 * pos_diff)  # Position weighted less
        
        # Collect activity levels for bias calculation
        left_activity = np.mean(left_vel_mag)
        right_activity = np.mean(right_vel_mag)
        left_activities.append(left_activity)
        right_activities.append(right_activity)
    
    # Calculate mean asymmetry
    C5_asym = np.mean(A_t)
    
    # Calculate one-sided penalty
    # V_L and V_R are overall left/right activity levels
    V_L = np.mean(left_activities)
    V_R = np.mean(right_activities)
    
    # Bias ratio: B = min(V_L, V_R) / (max(V_L, V_R) + ε)
    B = min(V_L, V_R) / (max(V_L, V_R) + epsilon)
    
    # Penalty indicator: P_bias = I[B < δ]
    # When B is small, one side is much less active than the other
    P_bias = 1.0 if B < delta else 0.0
    
    # Additional check: if both sides are very inactive, reduce penalty
    if max(V_L, V_R) < delta:
        P_bias *= 0.5  # Reduce penalty when both sides are inactive
    
    # Final C5 with one-sided penalty
    C5_final = C5_asym * (1 + lambda_penalty * P_bias)
    
    # Calculate per-pair asymmetries for detailed analysis
    pair_asymmetries = []
    for left_idx, right_idx in BILATERAL_PAIRS:
        left_vel_mag = np.linalg.norm(velocities[:, left_idx], axis=1)
        right_vel_mag = np.linalg.norm(velocities[:, right_idx], axis=1)
        pair_asym = np.mean(np.abs(left_vel_mag - right_vel_mag))
        pair_asymmetries.append(float(pair_asym))
    
    return {
        'complexity': float(C5_final),
        'components': {
            'C5_asym': float(C5_asym),
            'V_L': float(V_L),
            'V_R': float(V_R),
            'B_ratio': float(B),
            'P_bias': float(P_bias),
            'one_sided_penalty': float(lambda_penalty * P_bias),
            'pair_asymmetries': pair_asymmetries,
            'mean_A_t': float(np.mean(A_t)),
            'std_A_t': float(np.std(A_t)),
            'max_A_t': float(np.max(A_t))
        },
        'weights': {
            'delta': delta,
            'lambda_penalty': lambda_penalty,
            'position_weight': 0.5
        }
    }

def compute_all_complexities_revised(joint_positions, foot_contacts, fps=30, params=None):
    """
    Compute all 5 complexity metrics with REVISED C4 and C5 implementations
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    results = {}
    
    try:
        c1_params = params.get('C1', DEFAULT_PARAMS['C1'])
        results['C1_foot_movement'] = C1_foot_movement_complexity(
            joint_positions, foot_contacts, fps, **c1_params)
    except Exception as e:
        print(f"Error computing C1: {e}")
        results['C1_foot_movement'] = {'complexity': 0, 'components': {}, 'weights': {}}
    
    try:
        c2_params = params.get('C2', DEFAULT_PARAMS['C2'])
        results['C2_movement_density'] = C2_movement_density(
            joint_positions, fps, **c2_params)
    except Exception as e:
        print(f"Error computing C2: {e}")
        results['C2_movement_density'] = {'complexity': 0, 'components': {}, 'weights': {}}
    
    try:
        c3_params = params.get('C3', DEFAULT_PARAMS['C3'])
        results['C3_rotation'] = C3_rotation_complexity(
            joint_positions, fps, **c3_params)
    except Exception as e:
        print(f"Error computing C3: {e}")
        results['C3_rotation'] = {'complexity': 0, 'components': {}, 'weights': {}}
    
    try:
        c4_params = params.get('C4', DEFAULT_PARAMS['C4'])
        results['C4_coordination'] = C4_coordination_revised(
            joint_positions, fps, **c4_params)
    except Exception as e:
        print(f"Error computing C4: {e}")
        results['C4_coordination'] = {'complexity': 0, 'components': {}, 'weights': {}}
    
    try:
        c5_params = params.get('C5', DEFAULT_PARAMS['C5'])
        results['C5_asymmetry'] = C5_asymmetry_revised(
            joint_positions, fps, **c5_params)
    except Exception as e:
        print(f"Error computing C5: {e}")
        results['C5_asymmetry'] = {'complexity': 0, 'components': {}, 'weights': {}}
    
    return results

if __name__ == "__main__":
    print("=== Testing REVISED Complexity Functions (C4 & C5) ===")
    
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
        
        # Test revised complexity functions
        complexities = compute_all_complexities_revised(joint_positions, foot_contacts)
        
        print(f"\n=== REVISED Complexity Results ===")
        for metric_name, result in complexities.items():
            print(f"\n{metric_name}:")
            print(f"  Overall complexity: {result['complexity']:.6f}")
            if 'weights' in result and result['weights']:
                print(f"  Weights used: {result['weights']}")
            if 'components' in result:
                for comp_name, comp_value in result['components'].items():
                    if isinstance(comp_value, (int, float)):
                        print(f"  {comp_name}: {comp_value:.6f}")
                    else:
                        print(f"  {comp_name}: {comp_value}")
        
        # Test on simplified version
        simplified_file = sample_file.replace("_original.npy", "_simplified.npy")
        if Path(simplified_file).exists():
            data_simplified = np.load(simplified_file)
            motion_tensor_simplified = torch.from_numpy(data_simplified[:, :-4]).float().unsqueeze(0)
            joint_positions_simplified = recover_from_ric(motion_tensor_simplified, 22).squeeze(0).numpy()
            foot_contacts_simplified = data_simplified[:, -4:]
            
            complexities_simplified = compute_all_complexities_revised(joint_positions_simplified, foot_contacts_simplified)
            
            print(f"\n=== REVISED Complexity Comparison (Original vs Simplified) ===")
            for metric_name in complexities.keys():
                orig_score = complexities[metric_name]['complexity']
                simp_score = complexities_simplified[metric_name]['complexity']
                reduction = (orig_score - simp_score) / orig_score * 100 if orig_score > 0 else 0
                print(f"  {metric_name}: {orig_score:.6f} → {simp_score:.6f} ({reduction:+.1f}%)")
                
                # Show detailed components for C4 and C5
                if metric_name in ['C4_coordination', 'C5_asymmetry']:
                    orig_comps = complexities[metric_name]['components']
                    simp_comps = complexities_simplified[metric_name]['components']
                    
                    if metric_name == 'C4_coordination':
                        print(f"    Indicator: {orig_comps['indicator']:.1f} → {simp_comps['indicator']:.1f}")
                        print(f"    Variance: {orig_comps['variance_diff']:.6f} → {simp_comps['variance_diff']:.6f}")
                        print(f"    Min μ: {orig_comps['min_mu']:.6f} → {simp_comps['min_mu']:.6f}")
                    
                    elif metric_name == 'C5_asymmetry':
                        print(f"    C5_asym: {orig_comps['C5_asym']:.6f} → {simp_comps['C5_asym']:.6f}")
                        print(f"    P_bias: {orig_comps['P_bias']:.1f} → {simp_comps['P_bias']:.1f}")
                        print(f"    B_ratio: {orig_comps['B_ratio']:.6f} → {simp_comps['B_ratio']:.6f}")
                        print(f"    One-sided penalty: {orig_comps['one_sided_penalty']:.6f} → {simp_comps['one_sided_penalty']:.6f}")
        
        print(f"\n{'='*80}")
        print("REVISED VERSION: C4 & C5 implemented according to LaTeX formulas")
        print("- C3: γ·|θ_T - θ_1|_norm + α·mean(|Δθ|) + β·mean(|Δ²θ|)")
        print("  (Normalized total rotation + velocity + acceleration)")
        print("- C4: Var(I_upper - I_lower) × Indicator[min(μ_upper, μ_lower) > δ]")
        print("- C5: C5_asym × (1 + λ × P_bias) where P_bias = I[B < δ]")
        print("  (One-sided penalty: 한쪽만 움직이는 동작 = 복잡)")
        print("="*80)
        
    else:
        print("❌ Sample file not found")