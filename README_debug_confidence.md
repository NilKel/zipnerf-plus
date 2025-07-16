# Debug Confidence Grid Implementation

This document describes the implementation of debug confidence grid functionality for ZipNeRF potential field experiments. This enables loading pretrained confidence grids derived from well-trained baseline models for sanity check experiments.

## 🎯 Purpose

The debug confidence grid functionality implements the "sanity check" experiment described in the potential field approach:

> **Hypothesis**: If the volume integral feature formulation `V_feat = -C(x) * (G(x) · ∇X(x))` is correct, then using a frozen confidence grid from a well-trained baseline should yield **better** results than the baseline itself.

This tests whether the potential field formulation provides meaningful information to the MLP when given "perfect" geometry.

## 🏗️ Implementation Overview

### Modified Files

1. **`internal/field.py`** - Enhanced `ConfidenceField` class:
   - Added `pretrained_grid_path` parameter to load pretrained grids
   - Added `freeze_pretrained` parameter to control gradient flow
   - Automatic resolution matching and validation
   - Comprehensive logging and error handling

2. **`internal/configs.py`** - Added configuration parameters:
   - `debug_confidence_grid_path`: Path to pretrained confidence grid
   - `freeze_debug_confidence`: Whether to freeze the debug grid

3. **`internal/models.py`** - Updated model initialization:
   - Pass debug parameters to `ConfidenceField` constructor

### New Scripts

4. **`sample_density_to_confidence.py`** - Generate confidence grids:
   - Load trained ZipNeRF models
   - Sample density on regular 3D grids
   - Convert density to confidence logits
   - Save grids with metadata

5. **`verify_confidence_grids.py`** - Validation and visualization:
   - Verify grid integrity
   - Create slice visualizations
   - Test gradient computation

6. **`test_debug_confidence.py`** - Unit tests:
   - Test all functionality components
   - Verify configuration integration
   - Error handling tests

7. **`run_sanity_check_experiment.py`** - Automated experiment runner:
   - Complete sanity check workflow
   - Multiple experiment configurations
   - Automated command generation

## 📊 Generated Confidence Grids

For the lego scene, we have generated:

- **128³ grid**: `confidence_grids_lego/confidence_grid_128.pt` (8 MB)
  - 18,040 high-confidence voxels
  - Mean confidence: 0.015

- **256³ grid**: `confidence_grids_lego/confidence_grid_256.pt` (64 MB)
  - 149,863 high-confidence voxels  
  - Mean confidence: 0.015

## 🚀 Usage

### 1. Generate Confidence Grids

First, create confidence grids from a trained baseline model:

```bash
# Generate grids for both resolutions
python sample_density_to_confidence.py \
    --checkpoint_path exp/lego_baseline_25000_0704_2320/checkpoints/025000 \
    --output_dir ./confidence_grids_lego \
    --resolutions 128 256
```

### 2. Verify Generated Grids

```bash
# Verify and create visualizations
python verify_confidence_grids.py --create_viz
```

### 3. Run Sanity Check Experiment

#### Option A: Using the Automated Script (Recommended)

```bash
# Quick sanity check with frozen 128³ grid
python run_sanity_check_experiment.py --dry_run

# Actually run the experiment
python run_sanity_check_experiment.py

# High-resolution test with 256³ grid
python run_sanity_check_experiment.py --grid_resolution 256

# Run comprehensive comparison
python run_sanity_check_experiment.py --comparison
```

#### Option B: Manual Training Command

```bash
# Sanity check: frozen confidence grid
accelerate launch train.py \
    --gin_configs=configs/blender.gin \
    --gin_bindings="Config.data_dir = '/path/to/lego'" \
    --gin_bindings="Config.exp_name = 'lego_potential_sanity_frozen'" \
    --gin_bindings="Config.use_potential = True" \
    --gin_bindings="Config.debug_confidence_grid_path = 'confidence_grids_lego/confidence_grid_128.pt'" \
    --gin_bindings="Config.freeze_debug_confidence = True" \
    --gin_bindings="Config.max_steps = 25000"
```

### 4. Configuration Parameters

Add to your gin bindings:

```python
# Enable debug confidence grid
Config.debug_confidence_grid_path = 'confidence_grids_lego/confidence_grid_128.pt'
Config.freeze_debug_confidence = True  # For sanity check

# Standard potential field settings
Config.use_potential = True
Config.use_triplane = False  # Pure potential field test
```

## 🧪 Experiment Types

### 1. Sanity Check (Frozen Grid)
- **Purpose**: Test if volume integral features are meaningful
- **Configuration**: `freeze_debug_confidence = True`
- **Expected**: PSNR > baseline (~32+ vs ~31)
- **Interpretation**: If successful, formulation is sound

### 2. End-to-End Learning (Trainable Grid)
- **Purpose**: Test full learning capability
- **Configuration**: `freeze_debug_confidence = False`
- **Expected**: Similar or better than frozen
- **Interpretation**: Validates learning dynamics

### 3. Resolution Comparison
- **Purpose**: Test geometry detail requirements
- **Configurations**: 128³ vs 256³ grids
- **Expected**: Higher resolution may give better results
- **Interpretation**: Geometry detail importance

## 📈 Expected Results

### Successful Sanity Check
- **Frozen confidence grid**: PSNR ~32+ (better than baseline ~31)
- **Training stability**: Smooth convergence
- **Visual quality**: Sharp, detailed renders

### Failed Sanity Check
- **PSNR**: Similar to or worse than baseline
- **Possible causes**:
  - Bug in volume integral computation
  - Incorrect feature combination in MLP
  - Coordinate system mismatch
  - Gradient flow issues

## 🔧 Technical Details

### ConfidenceField Enhanced Constructor

```python
ConfidenceField(
    resolution=(128, 128, 128),
    device='cuda',
    pretrained_grid_path='confidence_grids_lego/confidence_grid_128.pt',
    freeze_pretrained=True  # For sanity check
)
```

### Volume Integral Feature Computation

The core computation remains:
```python
# Query confidence field
sampled_conf, sampled_grad = confidence_field.query(sample_coords)

# Compute volume integral features
dot_product = torch.sum(potential_features * sampled_grad_expanded, dim=-1)
volume_features = -sampled_conf_expanded * dot_product
```

### Coordinate Systems

- **Confidence grid**: `[-1, 1]³` coordinate space
- **Sample coordinates**: Transformed via scene contraction
- **Grid sampling**: Trilinear interpolation with `align_corners=True`

## 🐛 Troubleshooting

### Common Issues

1. **Grid not found**: Run `sample_density_to_confidence.py` first
2. **Resolution mismatch**: Grid resolution auto-updates to match pretrained
3. **Memory issues**: Use smaller batch sizes or 128³ instead of 256³
4. **Poor results**: Check coordinate transformations and feature computation

### Debug Commands

```bash
# Test basic functionality
python test_debug_confidence.py

# Verify specific grid
python verify_confidence_grids.py --grid_dir confidence_grids_lego

# Check model loading
python load_confidence_example.py
```

## 📚 File Structure

```
zipnerf-pytorch/
├── internal/
│   ├── field.py              # Enhanced ConfidenceField
│   ├── configs.py            # Added debug parameters
│   └── models.py             # Updated model init
├── confidence_grids_lego/
│   ├── confidence_grid_128.pt
│   ├── confidence_grid_256.pt
│   └── metadata files
├── sample_density_to_confidence.py
├── verify_confidence_grids.py
├── test_debug_confidence.py
├── run_sanity_check_experiment.py
└── README_debug_confidence.md
```

## 🎉 Next Steps

1. **Run sanity check**: Use frozen confidence grid
2. **Analyze results**: Compare PSNR with baseline
3. **Debug if needed**: Check feature computation if results are poor
4. **Scale experiments**: Try different scenes and resolutions
5. **Optimize**: Fine-tune regularization and learning rates

This implementation provides a robust framework for testing the potential field formulation with "perfect" geometry, enabling rapid validation of the core hypothesis. 