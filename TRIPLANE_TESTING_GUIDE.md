# ZipNeRF Triplane Models Testing Guide

This guide explains how to evaluate all your trained triplane models and generate comprehensive metrics reports.

## Overview

You have **8 ready triplane models** trained on different NeRF synthetic scenes:
- `lego` (2 models)
- `drums`, `ficus`, `hotdog`, `materials`, `mic`, `ship` (1 model each)

## Quick Start

### 1. Test All Models (Recommended)

```bash
# Quick dry run to see what will be evaluated
python test_all_triplane_models.py --dry_run

# Run full evaluation on all models
python test_all_triplane_models.py
```

### 2. Alternative: Use the Quick Script

```bash
# Dry run
./quick_test_all.sh dry_run

# Full evaluation
./quick_test_all.sh
```

## Available Options

### Basic Usage
```bash
python test_all_triplane_models.py [OPTIONS]
```

### Options:
- `--dry_run`: Preview commands without executing them
- `--scenes SCENE1 SCENE2`: Only evaluate specific scenes (e.g., `--scenes lego drums`)
- `--config PATH`: Use custom config file (default: `configs/blender.gin`)
- `--output FILENAME`: Set output report filename (default: `triplane_evaluation_report.json`)

### Examples:
```bash
# Test only lego and drums scenes
python test_all_triplane_models.py --scenes lego drums

# Use custom output filename
python test_all_triplane_models.py --output my_results.json

# Dry run for specific scenes
python test_all_triplane_models.py --dry_run --scenes ficus hotdog
```

## What the Script Does

1. **Discovery**: Automatically finds all `lego_triplane_relu_*` experiments
2. **Validation**: Checks which models have trained checkpoints ready
3. **Evaluation**: Runs `eval.py` on each ready model using the test set
4. **Metrics Extraction**: Parses PSNR, SSIM, and LPIPS metrics from results
5. **Reporting**: Generates comprehensive JSON and CSV reports

## Output Files

After running, you'll get:

### 1. JSON Report (`triplane_evaluation_report.json`)
Detailed results including:
- Experiment names and paths
- Success/failure status
- Full metrics for each scene
- Execution times
- Timestamps

### 2. CSV Summary (`triplane_evaluation_report.csv`)
Tabular format with:
- Scene names
- PSNR, SSIM, LPIPS values
- Evaluation status
- Duration

### 3. Console Summary
Real-time progress and final summary table with:
- Per-scene metrics
- Average metrics across all scenes
- Success/failure counts

## Expected Runtime

- **Per model**: ~5-15 minutes (depending on GPU and test set size)
- **All 8 models**: ~40-120 minutes total
- **Dry run**: <1 minute

## Your Current Models

Based on the discovery, here are your available models:

| Scene | Experiment | Status | Checkpoint |
|-------|------------|--------|------------|
| lego | `lego_triplane_relu_0703_2152` | ✅ Ready | 025000 |
| drums | `lego_triplane_relu_drums_0704_0014_0704_0014` | ✅ Ready | 050000 |
| ficus | `lego_triplane_relu_ficus_0704_0152_0704_0152` | ✅ Ready | 050000 |
| hotdog | `lego_triplane_relu_hotdog_0704_0325_0704_0325` | ✅ Ready | 050000 |
| lego | `lego_triplane_relu_lego_0703_2246_0703_2246` | ✅ Ready | 050000 |
| materials | `lego_triplane_relu_materials_0704_0452_0704_0452` | ✅ Ready | 050000 |
| mic | `lego_triplane_relu_mic_0704_0612_0704_0612` | ✅ Ready | 050000 |
| ship | `lego_triplane_relu_ship_0704_0731_0704_0731` | ✅ Ready | 050000 |

## Understanding the Results

### Metrics Explained:
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better, >30 is good)
- **SSIM**: Structural Similarity Index (0-1, higher is better, >0.9 is good)  
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better, <0.2 is good)

### Typical Good Results for NeRF Synthetic:
- PSNR: 30-35+ dB
- SSIM: 0.95-0.99
- LPIPS: 0.05-0.15

## Troubleshooting

### If evaluation fails:
1. Check that the experiment directory exists: `ls exp/EXPERIMENT_NAME/`
2. Verify checkpoints: `ls exp/EXPERIMENT_NAME/checkpoints/`
3. Check GPU memory: `nvidia-smi`
4. Review error logs in the script output

### Missing dependencies:
```bash
# If pandas is missing (optional, for better tables)
pip install pandas

# Core dependencies should already be installed
pip install torch accelerate
```

## Manual Single Model Testing

If you want to test just one model manually:

```bash
accelerate launch eval.py \
  --gin_configs=configs/blender.gin \
  --gin_bindings="Config.data_dir = '/home/nilkel/Projects/data/nerf_synthetic/SCENE'" \
  --gin_bindings="Config.exp_name = 'EXPERIMENT_NAME'" \
  --gin_bindings="Config.eval_only_once = True"
```

Replace `SCENE` and `EXPERIMENT_NAME` with your specific values.

## Next Steps

After evaluation:
1. Review the metrics in the generated reports
2. Compare performance across different scenes
3. Identify best-performing models
4. Use results for paper/presentation figures

The comprehensive reports will give you everything needed for analyzing your triplane integration performance! 