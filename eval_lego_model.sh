#!/bin/bash

# Evaluation script for the trained lego triplane model
# This script runs evaluation on the test set and reports PSNR, SSIM, LPIPS

set -e  # Exit on error

# Configuration
EXP_NAME="lego_triplane_25k_fixed"
DATA_DIR="/home/nilkel/Projects/data/nerf_synthetic/lego"
CHECKPOINT_DIR="exp/$EXP_NAME/checkpoints"

echo "🧪 Evaluating ZipNeRF Triplane Model: $EXP_NAME"
echo "==============================================="

# Activate conda environment
echo "🔧 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate zipnerf2

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Find the latest checkpoint
LATEST_CHECKPOINT=$(ls -1 "$CHECKPOINT_DIR" | sort -n | tail -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ Error: No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

echo "📁 Data directory: $DATA_DIR"
echo "🏷️  Experiment: $EXP_NAME"
echo "🔢 Latest checkpoint: $LATEST_CHECKPOINT"
echo "📊 Computing test set metrics..."
echo ""

# Run evaluation
echo "🏃 Running evaluation..."
accelerate launch eval.py \
    --gin_configs=configs/blender.gin \
    --gin_bindings="Config.data_dir = '$DATA_DIR'" \
    --gin_bindings="Config.exp_name = '$EXP_NAME'" \
    --gin_bindings="Config.use_triplane = True" \
    --gin_bindings="Config.eval_only_once = True" \
    --gin_bindings="Config.eval_save_output = True" \
    --gin_bindings="Config.eval_quantize_metrics = True"

echo ""
echo "✅ Evaluation completed!"
echo "📊 Check the evaluation log at: exp/$EXP_NAME/log_eval.txt"
echo "🖼️  Rendered images saved to: exp/$EXP_NAME/test_preds/" 