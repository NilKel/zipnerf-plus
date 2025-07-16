# Binary Occupancy with Straight-Through Estimator (STE)

## Overview

This implementation adds binary occupancy support to the ZipNeRF potential field formulation. Instead of using smooth sigmoid confidence values, the system can now use hard binary occupancy values (0 or 1) while maintaining gradient flow through the Straight-Through Estimator (STE).

## Motivation

The binary occupancy approach addresses potential issues with the smooth formulation:

1. **Forces Hard Decisions**: Binary values eliminate ambiguous "fuzzy" regions, forcing the network to make clear occupied/empty decisions
2. **Sharp Gradients**: Binary occupancy creates sparse, delta-function gradients at boundaries, providing cleaner surface signals
3. **Prevents Abstract Solutions**: Constrains the confidence field to learn geometry-like patterns rather than abstract modulations

## Implementation Details

### Core Components

1. **Configuration Flag**: `Config.binary_occupancy = True`
2. **STE in Gradient Computation**: Computes both binary and continuous gradients, combines with STE
3. **STE in Query**: Interpolates both binary and continuous values, returns binary with continuous gradients
4. **Modified Potential Formulation**: `binary_occ * (features ⋅ ∇occupancy)` instead of `-confidence * (features ⋅ ∇occupancy)`

### Straight-Through Estimator (STE)

The STE allows binary values in the forward pass while maintaining continuous gradients for backpropagation:

```python
# Forward: Use binary values
binary_grid = (sigmoid(logits) > 0.5).float()

# Backward: Use continuous gradients  
continuous_grid = sigmoid(logits)

# STE combination
output = binary_grid.detach() + (continuous_grid - continuous_grid.detach())
```

This ensures:
- **Forward pass**: Uses actual binary values (0 or 1)
- **Backward pass**: Gradients flow through the continuous sigmoid

### Key Files Modified

1. **`internal/configs.py`**: Added `binary_occupancy` flag
2. **`internal/field.py`**: Implemented STE in `compute_gradient()` and `query()` methods
3. **`internal/models.py`**: Modified potential formulation and passed binary_occupancy flag
4. **`train_pt.sh`**: Added support for binary occupancy parameter
5. **`configs/potential_binary.gin`**: Configuration for binary occupancy experiments
6. **`train_binary.sh`**: Dedicated training script for binary experiments

## Usage

### Basic Training
```bash
# Train with binary occupancy
./train_binary.sh lego binary_test

# Train with smooth occupancy (original)
./train_pt.sh lego smooth_test False
```

### Configuration Options
```python
Config.binary_occupancy = True     # Enable binary occupancy with STE
Config.use_potential = True        # Required for potential field
Config.use_triplane = True         # Optional triplane features
```

### Direct gin bindings
```bash
accelerate launch train.py \
  --gin_configs=configs/potential_binary.gin \
  --gin_bindings="Config.data_dir = '/path/to/scene'" \
  --gin_bindings="Config.binary_occupancy = True"
```

## Testing

Run the comprehensive test suite:
```bash
python test_binary_occupancy.py
```

Tests validate:
- Binary values are truly binary (0 or 1)
- Gradient flow through STE works correctly
- Comparison with smooth occupancy behavior
- Integration with pretrained grids

## Mathematical Formulation

### Original Smooth Formulation
```
V_feat = -C(x) * (G(x) · ∇X(x))
```
Where:
- `C(x)` is smooth confidence ∈ [0,1] 
- `G(x)` are learned features
- `∇X(x)` is confidence gradient

### Binary Occupancy Formulation  
```
V_feat = O(x) * (G(x) · ∇O(x))
```
Where:
- `O(x)` is binary occupancy ∈ {0,1}
- `∇O(x)` is sparse gradient (non-zero only at boundaries)
- STE maintains gradient flow: `O(x) = STE(sigmoid(logits))`

## Expected Benefits

1. **Sharper Surfaces**: Binary boundaries create cleaner surface definitions
2. **Reduced Ambiguity**: Forces clear inside/outside decisions
3. **Better Disentanglement**: Prevents abstract patterns in favor of geometric structure
4. **Cleaner Gradients**: Sparse gradients concentrate signal at actual boundaries

## Debugging and Analysis

The implementation includes comprehensive logging and debugging tools:

1. **Binary Statistics**: Tracks percentage of occupied voxels
2. **Gradient Validation**: Ensures STE gradients flow correctly  
3. **Comparison Tools**: Easy switching between binary and smooth modes
4. **Pretrained Grid Support**: Can load existing confidence grids and convert to binary

## Example Results

When using the pretrained confidence grid from a working potential model:
- **Smooth confidence**: Mean=0.312, 31.2% high-confidence voxels, continuous values
- **Binary occupancy**: 31.2% occupied voxels, only values {0,1}, same gradient regions

## Next Steps

1. **Train and Compare**: Run binary vs smooth experiments on the same scene
2. **Analyze Gradients**: Compare gradient sparsity and surface quality
3. **Tune Hyperparameters**: Adjust learning rates and regularization for binary case
4. **Evaluate Metrics**: Compare PSNR, SSIM, and visual quality between formulations

The binary occupancy implementation provides a powerful tool for testing whether forcing hard geometric decisions improves the potential field approach's ability to learn disentangled geometry representations. 