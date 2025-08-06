# Complexity Score Extraction Guide

This guide provides a comprehensive overview of the complexity score extraction codes in the current codebase for dance motion analysis and simplification.

## 📋 Overview

Our codebase implements 5 different complexity metrics for dance motion analysis:
- **C1**: Foot Movement Complexity (steps, velocity, patterns, range)
- **C2**: Movement Density (frequency of arm and foot movements)
- **C3**: Rotation Complexity (pelvis-based rotation patterns)
- **C4**: Multi-limb Coordination (upper/lower body coordination)
- **C5**: Left-Right Asymmetry (bilateral movement differences)

## 🔧 Core Function Files

### 1. Main Implementation: `complexity_functions_revised.py`

This is the primary file containing all complexity calculation functions:

```python
# Individual complexity functions
def C1_foot_movement_complexity(joint_positions, foot_contacts, fps=30):
    """Calculates foot movement complexity based on velocity, patterns, and range"""
    
def C2_movement_density(joint_positions, fps=30):
    """Calculates movement density across all limbs"""
    
def C3_rotation_complexity(joint_positions, fps=30):
    """Calculates rotation complexity based on pelvis rotations"""
    
def C4_coordination_revised(joint_positions, fps=30):
    """Calculates multi-limb coordination complexity"""
    
def C5_asymmetry_revised(joint_positions, fps=30):
    """Calculates left-right asymmetry in movements"""

# Unified analysis function
def compute_all_complexities(joint_positions, foot_contacts, fps=30, params=None):
    """Computes all 5 complexity metrics at once"""
```

### 2. Advanced Version: `advanced_complexity_functions.py`

Enhanced versions with frequency domain analysis:

```python
# Advanced frequency analysis
def extract_advanced_frequency_features(signal_data, fps=30):
    """Extracts spectral features like dominant frequency, centroid, bandwidth"""

# Enhanced complexity functions
def advanced_C1_foot_movement_complexity(joint_positions, foot_contacts, fps=30):
def advanced_C2_movement_density(joint_positions, fps=30):
# ... advanced versions of all complexity metrics
```

## 📊 Input/Output Format

### Input Requirements

All complexity functions expect:
- **joint_positions**: `(T, 22, 3)` numpy array - joint positions over time
- **foot_contacts**: `(T, 4)` numpy array - foot contact information (heel/toe for each foot)
- **fps**: Frame rate (default: 30)

### Output Format

Each function returns a dictionary:
```python
{
    'complexity': float,        # Main complexity score (0-1 normalized)
    'components': {...},        # Detailed component scores
    'weights': {...}           # Component weights used in calculation
}
```

## 🎯 Usage Examples

### Basic Usage - Individual Complexity

```python
from complexity_functions_revised import C1_foot_movement_complexity
from mld.data.humanml.scripts.motion_process import recover_from_ric
import numpy as np
import torch

# Load motion data
motion_data = np.load("path/to/motion.npy")  # Shape: (T, 263)

# Convert to joint positions
joints = recover_from_ric(torch.from_numpy(motion_data).float(), 22).numpy()

# Extract foot contact information (last 4 dimensions)
foot_contacts = motion_data[:, -4:]

# Calculate C1 complexity
c1_result = C1_foot_movement_complexity(joints, foot_contacts, fps=30)
print(f"C1 Steps Complexity: {c1_result['complexity']:.3f}")
```

### Comprehensive Analysis - All Complexities

```python
from complexity_functions_revised import compute_all_complexities

# Compute all complexities at once
all_complexities = compute_all_complexities(
    joint_positions=joints,
    foot_contacts=foot_contacts,
    fps=30
)

# Print results
print("=== Dance Motion Complexity Analysis ===")
print(f"C1 Steps:        {all_complexities['C1_foot_movement']['complexity']:.3f}")
print(f"C2 Density:      {all_complexities['C2_movement_density']['complexity']:.3f}")
print(f"C3 Rotation:     {all_complexities['C3_rotation']['complexity']:.3f}")
print(f"C4 Coordination: {all_complexities['C4_coordination']['complexity']:.3f}")
print(f"C5 Asymmetry:    {all_complexities['C5_asymmetry']['complexity']:.3f}")
```

## 🏗️ Integration in Codebase

### 1. Dataset Integration

```python
def complexity_based_mask(self, simplified_motion: np.ndarray):
    """Generates trajectory hints based on complexity type"""
    joints = recover_from_ric(torch.from_numpy(simplified_motion).float(), self.njoints).numpy()
    
    if self.complexity_type == 'C1' or self.complexity_type == 'C1_steps':
        # Select frames with high foot movement
        foot_joints = joints[:, [10, 11]]  # left and right feet
        foot_velocity = np.linalg.norm(np.diff(foot_joints, axis=0), axis=(1, 2))
        threshold = np.percentile(foot_velocity, 70)
        candidate_frames = np.where(foot_velocity > threshold)[0] + 1
        
    elif self.complexity_type == 'C2' or self.complexity_type == 'C2_density':
        # Select frames with high pelvis movement
        pelvis_velocity = np.linalg.norm(np.diff(joints[:, 0], axis=0), axis=-1)
        # ... similar logic for other complexity types
```

### 2. Demo Applications

```python
class DanceSimplificationDemo:
    def analyze_complexity(self, motion: np.ndarray) -> Dict[str, float]:
        """Analyzes C1 Steps complexity of motion"""
        joints = recover_from_ric(torch.from_numpy(motion).float(), 22).numpy()
        foot_contacts = motion[:, -4:]
        
        # Calculate C1 complexity
        c1_score = C1_foot_movement_complexity(joints, foot_contacts, fps=20)
        
        # Additional motion statistics
        foot_joints = joints[:, [10, 11]]
        foot_velocity = np.linalg.norm(np.diff(foot_joints, axis=0), axis=(1, 2))
        
        return {
            'C1_steps_complexity': c1_score,
            'avg_foot_velocity': np.mean(foot_velocity),
            'max_foot_velocity': np.max(foot_velocity),
            'foot_movement_frames': np.sum(foot_velocity > np.percentile(foot_velocity, 70))
        }
```

## 📁 File Structure

```
project_root/
├── complexity_functions_revised.py          # Main complexity functions
├── advanced_complexity_functions.py         # Enhanced versions with frequency analysis
```

## 🎨 Detailed Function Descriptions

### C1: Foot Movement Complexity
- **Purpose**: Analyzes foot step patterns, velocity, and movement range
- **Key Components**: Step velocity, direction patterns, movement frequency, range
- **Use Case**: Detecting complex dance steps and footwork

### C2: Movement Density  
- **Purpose**: Measures overall movement frequency across all limbs
- **Key Components**: Arm movement frequency, leg movement frequency
- **Use Case**: Identifying busy vs. calm dance sections

### C3: Rotation Complexity
- **Purpose**: Analyzes pelvis-based rotations and turning patterns
- **Key Components**: Total rotation amount, rotation velocity, direction changes
- **Use Case**: Detecting spins, turns, and rotational movements

### C4: Multi-limb Coordination
- **Purpose**: Measures coordination between upper and lower body
- **Key Components**: Upper body movement, lower body movement, coordination variance
- **Use Case**: Identifying complex choreographic coordination

### C5: Left-Right Asymmetry
- **Purpose**: Analyzes bilateral movement differences
- **Key Components**: Left vs. right limb movements, asymmetry patterns
- **Use Case**: Detecting unilateral vs. bilateral movement patterns

## 🔧 Parameter Customization

Each function accepts custom parameters for fine-tuning:

```python
# Default parameters
DEFAULT_PARAMS = {
    'C1': {'alpha': 1.5, 'beta': 0.05, 'zeta': 15.0},
    'C2': {'gamma': 0.005},
    'C3': {'gamma': 0.3, 'alpha': 1.0, 'beta': 0.5},
    'C4': {'delta': 0.01},
    'C5': {'delta': 0.01, 'lambda_penalty': 0.5}
}

# Custom usage
custom_params = {
    'C1': {'alpha': 2.0, 'beta': 0.1, 'zeta': 20.0}  # More sensitive to foot movements
}

results = compute_all_complexities(joints, foot_contacts, fps=30, params=custom_params)
```

## 📈 Output Analysis

### Interpreting Complexity Scores

- **0.0 - 0.3**: Low complexity (simple, repetitive movements)
- **0.3 - 0.7**: Medium complexity (moderate variation and coordination)
- **0.7 - 1.0**: High complexity (intricate, varied movements)

### Component Analysis

Each complexity function provides detailed component breakdowns:

```python
c1_result = C1_foot_movement_complexity(joints, foot_contacts)

print("C1 Components:")
print(f"  Step Velocity: {c1_result['components']['step_velocity']:.3f}")
print(f"  Direction Patterns: {c1_result['components']['direction_patterns']:.3f}")
print(f"  Movement Frequency: {c1_result['components']['movement_frequency']:.3f}")
print(f"  Movement Range: {c1_result['components']['movement_range']:.3f}")
```

## 🚀 Quick Start Example

Here's a complete example to get you started:

```python
import numpy as np
import torch
from complexity_functions_revised import compute_all_complexities
from mld.data.humanml.scripts.motion_process import recover_from_ric

# 1. Load your motion data
motion_file = "path/to/your/motion.npy"
motion_data = np.load(motion_file)  # Shape should be (T, 263)

# 2. Convert to joint positions
joints = recover_from_ric(torch.from_numpy(motion_data).float(), 22).numpy()

# 3. Extract foot contacts (if available in your data)
foot_contacts = motion_data[:, -4:]  # Last 4 dimensions

# 4. Compute all complexities
complexities = compute_all_complexities(joints, foot_contacts, fps=30)

# 5. Display results
print("=== Motion Complexity Analysis ===")
for metric_name, metric_data in complexities.items():
    complexity_score = metric_data['complexity']
    print(f"{metric_name}: {complexity_score:.3f}")

print("\nAnalysis complete!")
```

## 💡 Tips and Best Practices

1. **Frame Rate**: Ensure you use the correct fps parameter matching your motion data
2. **Data Format**: Always verify your motion data is in the expected (T, 263) format
3. **Joint Recovery**: Use `recover_from_ric()` to convert motion representation to joint positions
4. **Normalization**: All complexity scores are normalized to [0, 1] range for consistency
5. **Batch Processing**: For multiple motions, process them individually for accurate results

## 🔍 Troubleshooting

### Common Issues

1. **Dimension Mismatch**: Ensure motion data has shape (T, 263)
2. **Missing Foot Contacts**: If foot contact data is unavailable, you can estimate it or use zeros
3. **Frame Rate**: Incorrect fps can lead to inaccurate velocity calculations
4. **Memory Issues**: For very long sequences, consider processing in chunks

### Error Handling

The functions include robust error handling:

```python
try:
    complexities = compute_all_complexities(joints, foot_contacts)
except Exception as e:
    print(f"Error computing complexities: {e}")
    # Functions will return default values (complexity: 0) on error
```

---

This guide should help you understand and use the complexity extraction tools effectively. For more specific use cases or questions, refer to the individual function documentation in the source files.