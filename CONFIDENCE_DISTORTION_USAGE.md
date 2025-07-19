# Confidence Distortion Loss - Quick Usage Guide

## What It Does

The confidence distortion loss encourages compact occupancy distributions along rays in your confidence field, helping to reduce floating artifacts.

## Quick Setup

### 1. Enable in Config

Add these lines to your `.gin` config file:

```gin
Config.confidence_distortion_loss_mult = 0.01
Config.use_potential = True
```

### 2. Run Training

Train normally - the loss will be automatically applied:

```bash
python train.py --gin_configs=configs/your_config.gin
```

### 3. Monitor in Logs

Look for `confidence_distortion` in your training output:

```
Step 1000: loss=0.0234 (data=0.0180|conf_dist=0.0021|distortion=0.0033)
```

## Example Configurations

### Basic Usage
```gin
include 'configs/zip.gin'

Config.confidence_distortion_loss_mult = 0.01
Config.use_potential = True
```

### With ADMM Pruning
```gin
include 'configs/zip.gin'

# Confidence distortion for compactness
Config.confidence_distortion_loss_mult = 0.01

# ADMM pruning for sparsity  
Config.use_admm_pruner = True
Config.admm_sparsity_constraint = 0.04

Config.use_potential = True
```

### For Outdoor Scenes
```gin
include 'configs/zip.gin'

Config.confidence_distortion_loss_mult = 0.01
Config.contraction_aware_gradients = True
Config.use_potential = True
```

## Hyperparameter Guidelines

- **Start with**: `0.01` (same as regular distortion loss)
- **Typical range**: `0.001` to `0.1`
- **Too high**: Over-constrains confidence field
- **Too low**: Doesn't reduce floaters effectively

## Requirements

✅ **Must have**: `Config.use_potential = True`  
✅ **Must have**: `Config.confidence_distortion_loss_mult > 0`  
✅ **Must have**: Confidence field initialized  

## Expected Results

- 🎯 **Reduced floaters** in rendered images
- 🎯 **Sharper geometry** boundaries  
- 🎯 **More compact** confidence distributions
- 🎯 **Better alignment** between confidence and density fields

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Loss is always 0 | Enable `use_potential = True` |
| No floater reduction | Increase loss multiplier |
| Loss too high | Decrease multiplier or check for conflicts |

For detailed information, see `docs/CONFIDENCE_DISTORTION_LOSS.md` 