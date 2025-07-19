# Confidence Distortion Loss

This document explains the confidence distortion loss feature, which helps reduce floaters by encouraging compact occupancy distributions along rays in the confidence field.

## Overview

The confidence distortion loss (`L_conf_dist`) is analogous to the original mip-NeRF 360 distortion loss, but operates on the sampled confidence values rather than the density weights. This encourages the confidence field to have sharp, compact occupancy distributions, which helps eliminate floating artifacts.

## How It Works

### Core Concept

The original distortion loss in mip-NeRF 360 operates on the output weights from the MLP to encourage compact density distributions. However, your confidence field is an **input** to the MLP. The confidence distortion loss fills this gap by applying the same distortion penalty directly to the confidence values sampled along each ray.

### Mathematical Formulation

```
L_conf_dist = λ_conf_dist * mean(distortion_loss(s, c))
```

Where:
- `s` are the normalized distances along the ray (`sdist`)
- `c` are the sampled confidence values from the confidence field
- `λ_conf_dist` is the loss multiplier (configurable)
- `distortion_loss` is the same function used for the original distortion loss

## Implementation Details

### Where Confidence is Sampled

The confidence values are sampled in `MLP.predict_density()` where the confidence field is queried:

```python
# In predict_density() when using potential encoder:
means_for_conf = means.view(-1, 3)
sampled_conf_raw, sampled_grad = confidence_field.query(means_for_conf)

# Store for distortion loss (shaped like ray samples)
sampled_conf = sampled_conf_raw.view(*means.shape[:-1], 1)  # (..., num_samples, 1)
```

### Loss Computation

The loss is computed in `train_utils.confidence_distortion_loss()`:

```python
def confidence_distortion_loss(ray_history, config):
    """Computes distortion loss on sampled confidence values."""
    last_ray_results = ray_history[-1]  # Apply only to final NeRF level
    
    if 'sampled_confidence' not in last_ray_results:
        return torch.tensor(0.0, device=last_ray_results['sdist'].device)
        
    c = last_ray_results['sdist']  # Normalized distances
    w = last_ray_results['sampled_confidence']  # Confidence values
    
    loss = stepfun.lossfun_distortion(c, w).mean()
    return config.confidence_distortion_loss_mult * loss
```

## Configuration

### Required Settings

The confidence distortion loss requires:

1. **Potential encoder enabled**: `Config.use_potential = True`
2. **Confidence field**: The feature requires a confidence field to sample from
3. **Loss multiplier > 0**: `Config.confidence_distortion_loss_mult > 0`

### Example Configuration

```gin
# Enable confidence distortion loss
Config.confidence_distortion_loss_mult = 0.01

# Required: Use potential encoder
Config.use_potential = True

# Ensure confidence field is available
Config.confidence_grid_resolution = (128, 128, 128)
```

### Hyperparameter Tuning

- **Start with 0.01**: Similar magnitude to the original distortion loss
- **Range**: Typically between 0.001 and 0.1
- **Too high**: May over-constrain the confidence field, leading to overly sparse occupancy
- **Too low**: May not effectively reduce floaters

## Usage

### Basic Usage

1. **Enable the loss in your config**:
   ```gin
   Config.confidence_distortion_loss_mult = 0.01
   Config.use_potential = True
   ```

2. **Train normally**: The loss will be automatically computed and applied during training

3. **Monitor in logs**: The loss appears as `confidence_distortion` in your training logs

### Advanced Usage

#### Combining with ADMM Pruning

The confidence distortion loss works well with ADMM pruning:

```gin
# Confidence distortion for compactness
Config.confidence_distortion_loss_mult = 0.01

# ADMM pruning for sparsity
Config.use_admm_pruner = True
Config.admm_sparsity_constraint = 0.04
Config.admm_penalty_rho = 1e-4
```

#### Contraction-Aware Gradients

For outdoor scenes with spatial contraction:

```gin
Config.confidence_distortion_loss_mult = 0.01
Config.contraction_aware_gradients = True
```

## Expected Behavior

### Training Logs

When enabled, you'll see additional logging:
```
Step 1000: loss=0.0234 (data=0.0180|conf_dist=0.0021|distortion=0.0033)
```

### Effects on Rendering

- **Reduced floaters**: Confidence field becomes more compact along rays
- **Sharper geometry**: Occupancy transitions become more defined
- **Consistent with density**: Confidence field aligns better with density field geometry

## Troubleshooting

### Loss is Always Zero

**Cause**: Confidence field is not being sampled
**Solutions**:
- Ensure `Config.use_potential = True`
- Check that confidence field is initialized
- Verify MLP receives confidence_field parameter

### Loss is Too High/Low

**Cause**: Inappropriate loss multiplier
**Solutions**:
- Start with `0.01` and adjust based on other loss magnitudes
- Monitor the ratio of confidence_distortion to total loss
- Typical range: 5-20% of total loss

### No Improvement in Floaters

**Cause**: Loss weight too low or competing regularizers
**Solutions**:
- Increase `confidence_distortion_loss_mult`
- Check for conflicting regularization (high `confidence_reg_mult`)
- Ensure confidence field has sufficient capacity

## Related Features

- **Original Distortion Loss**: `Config.distortion_loss_mult`
- **ADMM Pruning**: `Config.use_admm_pruner` (for sparsity)
- **Confidence Regularization**: `Config.confidence_reg_mult` (for smoothness)
- **Contraction-Aware Gradients**: `Config.contraction_aware_gradients`

## Code Files Modified

- `internal/models.py`: MLP modifications to return sampled confidence
- `internal/train_utils.py`: Confidence distortion loss function
- `internal/configs.py`: Configuration parameter
- `train.py`: Integration into training loop 