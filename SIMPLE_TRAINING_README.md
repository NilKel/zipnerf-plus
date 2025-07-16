# Simple ZipNeRF Potential Training

No more complex tmux scripts! This system provides a simple way to train ZipNeRF with potential fields.

## Quick Start

### Train from scratch (generates new confidence grid):
```bash
python train_potential.py my_experiment --scene lego
```

### Train with existing confidence grid:
```bash
python train_potential.py my_experiment --scene lego --load_grid
```

### Custom settings:
```bash
python train_potential.py my_experiment --scene lego --max_steps 50000 --batch_size 8192
```

## What This Does

1. **Unified Configuration**: All settings are in `configs/unified_potential.gin`
2. **Simple Command**: Just specify experiment name and scene
3. **Automatic Naming**: Creates unique names like `lego_potential_25000_0712_1845_my_experiment`
4. **Auto Grid Extraction**: Automatically saves confidence grid after training
5. **One Checkpoint**: Only keeps the final checkpoint to save disk space

## Key Features

- ✅ **No tmux complexity** - runs directly in your terminal
- ✅ **Automatic confidence grid handling** - saves and loads grids automatically  
- ✅ **Fixed checkpoint saving** - no more disk space crashes
- ✅ **Simple error handling** - clear error messages
- ✅ **Sensible defaults** - works out of the box

## Configuration

Edit `configs/unified_potential.gin` to change:
- Model architecture settings
- Learning rates
- Confidence field resolution
- Wandb project name
- Training hyperparameters

## Examples

```bash
# Basic training
python train_potential.py baseline_test --scene lego

# Train chair scene with existing grid
python train_potential.py chair_exp --scene chair --load_grid

# Quick test (fewer steps)
python train_potential.py quick_test --scene lego --max_steps 1000

# Large batch size for RTX 5090
python train_potential.py large_batch --scene lego --batch_size 16384
```

## Output

After training completes:
- Model saved to: `exp/lego_potential_25000_0712_1845_my_experiment/`
- Confidence grid saved to: `debug_grids/debug_confidence_grid_256.pt`
- Training logs: `exp/lego_potential_25000_0712_1845_my_experiment/log_train.txt`

## Troubleshooting

**Indentation Error**: If you see `IndentationError: expected an indented block` in checkpoints.py, run:
```bash
# Fix the indentation issue
sed -i 's/^    shutil.rmtree(folder)/                    shutil.rmtree(folder)/' internal/checkpoints.py
```

**CUDA Out of Memory**: Reduce batch size:
```bash
python train_potential.py my_exp --scene lego --batch_size 2048
```

**Scene not found**: Make sure the scene exists:
```bash
ls /home/nilkel/Projects/data/nerf_synthetic/
```

## Migration from tmux scripts

Instead of:
```bash
./train_pt_tmux.sh lego my_complex_experiment_name_with_many_options
```

Use:
```bash
python train_potential.py my_experiment --scene lego --load_grid
```

Much simpler! 