# ZipNeRF Triplane Models Evaluation Guide

This guide provides comprehensive instructions for evaluating all trained ZipNeRF triplane models on the full Blender synthetic dataset.

## Overview

The evaluation system automatically:
1. **Discovers** all triplane models in the `exp/` directory
2. **Organizes** them into a clean directory structure (removing "lego_" prefix)
3. **Evaluates** each model on its corresponding Blender scene
4. **Extracts** metrics (PSNR, SSIM, LPIPS) from evaluation results
5. **Generates** comprehensive reports in JSON, CSV, and text formats

## Found Models

The system discovered **7 triplane models** trained on different Blender scenes:

| Scene | Original Model Name | New Name | Checkpoint Steps |
|-------|---------------------|----------|------------------|
| drums | lego_triplane_relu_drums_0704_0014_0704_0014 | triplane_relu_drums | 50,000 |
| ship | lego_triplane_relu_ship_0704_0731_0704_0731 | triplane_relu_ship | 50,000 |
| mic | lego_triplane_relu_mic_0704_0612_0704_0612 | triplane_relu_mic | 50,000 |
| ficus | lego_triplane_relu_ficus_0704_0152_0704_0152 | triplane_relu_ficus | 50,000 |
| lego | lego_triplane_relu_lego_0703_2246_0703_2246 | triplane_relu_lego | 50,000 |
| hotdog | lego_triplane_relu_hotdog_0704_0325_0704_0325 | triplane_relu_hotdog | 50,000 |
| materials | lego_triplane_relu_materials_0704_0452_0704_0452 | triplane_relu_materials | 50,000 |

## Quick Start

### 1. Full Evaluation (Organize + Test)
```bash
# Run everything automatically
./run_triplane_evaluation.sh

# Or use Python directly
python organize_and_test_triplane_models.py
```

### 2. Dry Run (Preview Only)
```bash
# See what would happen without making changes
./run_triplane_evaluation.sh dry_run

# Or use Python directly
python organize_and_test_triplane_models.py --dry_run
```

### 3. Organize Only
```bash
# Only organize models, don't run evaluation
./run_triplane_evaluation.sh organize_only

# Or use Python directly
python organize_and_test_triplane_models.py --organize_only
```

### 4. Test Only
```bash
# Only run evaluation, don't organize
./run_triplane_evaluation.sh test_only

# Or use Python directly
python organize_and_test_triplane_models.py --test_only
```

## Directory Structure

### Before Organization
```
exp/
├── lego_triplane_relu_drums_0704_0014_0704_0014/
├── lego_triplane_relu_ship_0704_0731_0704_0731/
├── lego_triplane_relu_mic_0704_0612_0704_0612/
├── lego_triplane_relu_ficus_0704_0152_0704_0152/
├── lego_triplane_relu_lego_0703_2246_0703_2246/
├── lego_triplane_relu_hotdog_0704_0325_0704_0325/
└── lego_triplane_relu_materials_0704_0452_0704_0452/
```

### After Organization
```
exp/
├── triplane_models/
│   ├── triplane_relu_drums/
│   ├── triplane_relu_ship/
│   ├── triplane_relu_mic/
│   ├── triplane_relu_ficus/
│   ├── triplane_relu_lego/
│   ├── triplane_relu_hotdog/
│   └── triplane_relu_materials/
└── [original models remain untouched]
```

### Results Directory
```
triplane_results/
├── triplane_evaluation_YYYYMMDD_HHMMSS.json
├── triplane_evaluation_YYYYMMDD_HHMMSS.csv
└── triplane_summary_YYYYMMDD_HHMMSS.txt
```

## Evaluation Process

For each model, the system:

1. **Loads** the model checkpoint (50,000 steps)
2. **Runs** evaluation on the test set using `eval.py`
3. **Extracts** metrics from the evaluation output
4. **Stores** results with detailed logging

### Evaluation Command
```bash
source activate_zipnerf.sh && 
accelerate launch eval.py \
  --gin_configs=configs/blender.gin \
  --gin_bindings="Config.data_dir = '/path/to/scene'" \
  --gin_bindings="Config.exp_name = 'triplane_models/triplane_relu_scene'" \
  --gin_bindings="Config.eval_only_once = True" \
  --gin_bindings="Config.eval_save_output = True"
```

## Output Reports

### 1. JSON Report (`triplane_evaluation_*.json`)
Complete detailed results including:
- Model information
- Success/failure status
- Extracted metrics (PSNR, SSIM, LPIPS)
- Execution time
- Full stdout/stderr logs

### 2. CSV Report (`triplane_evaluation_*.csv`)
Tabular format with columns:
- `exp_name`: Model name
- `scene_name`: Blender scene
- `success`: Evaluation success status
- `checkpoint_step`: Training steps
- `psnr`: Peak Signal-to-Noise Ratio
- `ssim`: Structural Similarity Index
- `lpips`: Learned Perceptual Image Patch Similarity
- `duration`: Evaluation time
- `error`: Error message (if failed)

### 3. Summary Report (`triplane_summary_*.txt`)
Human-readable summary including:
- Successful evaluations with metrics
- Failed evaluations with error details
- Average metrics across all successful models

## Expected Metrics

Based on ZipNeRF triplane performance on Blender synthetic:
- **PSNR**: ~25-35 dB (higher is better)
- **SSIM**: ~0.85-0.95 (higher is better)
- **LPIPS**: ~0.05-0.20 (lower is better)

## Troubleshooting

### Common Issues

1. **Environment Not Activated**
   ```bash
   source activate_zipnerf.sh
   ```

2. **Data Directory Not Found**
   - Verify: `/home/nilkel/Projects/data/nerf_synthetic/`
   - Update path with: `--data_dir /path/to/your/data`

3. **Checkpoint Not Found**
   - Check: `exp/model_name/checkpoints/050000/`
   - Verify training completed successfully

4. **GPU Memory Issues**
   - Reduce batch size in gin config
   - Use smaller render chunk size

5. **Evaluation Timeout**
   - Default timeout: 1 hour per model
   - Increase in script if needed

### Debug Commands

```bash
# Test single model manually
source activate_zipnerf.sh
accelerate launch eval.py \
  --gin_configs=configs/blender.gin \
  --gin_bindings="Config.data_dir = '/home/nilkel/Projects/data/nerf_synthetic/lego'" \
  --gin_bindings="Config.exp_name = 'lego_triplane_relu_lego_0703_2246_0703_2246'"

# Check model structure
ls -la exp/lego_triplane_relu_lego_0703_2246_0703_2246/checkpoints/

# Verify data exists
ls -la /home/nilkel/Projects/data/nerf_synthetic/lego/
```

## Advanced Usage

### Custom Data Directory
```bash
python organize_and_test_triplane_models.py --data_dir /path/to/your/data
```

### Selective Evaluation
To evaluate specific models, modify the `BLENDER_SCENES` list in the script:
```python
BLENDER_SCENES = ["lego", "hotdog"]  # Only evaluate these scenes
```

### Metric Extraction Patterns
The script uses regex patterns to extract metrics from evaluation output:
- PSNR: `r'PSNR:\s*([0-9.]+)'`
- SSIM: `r'SSIM:\s*([0-9.]+)'`
- LPIPS: `r'LPIPS:\s*([0-9.]+)'`

## Performance Expectations

- **Evaluation Time**: ~10-30 minutes per model
- **Total Time**: ~2-4 hours for all 7 models
- **Disk Space**: ~50MB per evaluation result
- **Memory**: ~8-16GB GPU memory required

## Files Created

The evaluation process creates these files:
- `exp/triplane_models/` - Organized model directory
- `triplane_results/` - Evaluation reports
- Individual model `test_preds/` directories with rendered images

## Next Steps

After evaluation:
1. Review metrics in CSV/summary reports
2. Compare with baseline ZipNeRF results
3. Analyze per-scene performance
4. Generate visualizations from test predictions
5. Create comparison tables for paper/presentation

## Support

For issues or questions:
1. Check the detailed JSON report for full error logs
2. Verify environment setup with `activate_zipnerf.sh`
3. Test individual model evaluation manually
4. Review gin config settings in `configs/blender.gin` 