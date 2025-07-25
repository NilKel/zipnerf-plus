#!/bin/bash

# Evaluation script for lego baseline model
# Auto-detects the latest baseline experiment

set -e  # Exit on error

# Configuration
SCENE_NAME="lego"
DATA_DIR="/home/nilkel/Projects/data/nerf_synthetic/lego"
BASE_EXP_DIR="exp"

echo "🧪 Evaluating ZipNeRF Baseline Model for $SCENE_NAME"
echo "================================================="

# Find the latest baseline experiment
LATEST_BASELINE_EXP=$(ls -1t "$BASE_EXP_DIR" | grep "${SCENE_NAME}_baseline_" | head -1)

if [ -z "$LATEST_BASELINE_EXP" ]; then
    echo "❌ Error: No baseline experiments found for $SCENE_NAME"
    echo "Available experiments:"
    ls -1 "$BASE_EXP_DIR" | grep "$SCENE_NAME" || echo "  (none found)"
    exit 1
fi

EXP_NAME="$LATEST_BASELINE_EXP"
CHECKPOINT_DIR="$BASE_EXP_DIR/$EXP_NAME/checkpoints"

# Kill any existing evaluation process
echo "🛑 Stopping any existing evaluation processes..."
pkill -f "eval.py" || true

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
echo "📏 Factor: 4 (4x downsampling - matching training)"
echo "🔧 Mode: BASELINE (Standard ZipNeRF)"
echo "📊 Computing test set metrics..."
echo ""

# Run evaluation with matching configuration
echo "🏃 Running baseline evaluation..."
accelerate launch eval.py \
    --gin_configs=configs/blender.gin \
    --gin_bindings="Config.data_dir = '$DATA_DIR'" \
    --gin_bindings="Config.exp_name = '$EXP_NAME'" \
    --gin_bindings="Config.use_triplane = False" \
    --gin_bindings="Config.factor = 4" \
    --gin_bindings="Config.batch_size = 8192" \
    --gin_bindings="Config.eval_only_once = True" \
    --gin_bindings="Config.eval_save_output = True" \
    --gin_bindings="Config.eval_quantize_metrics = True"

echo ""
echo "✅ Baseline evaluation completed!"
echo "📊 Check the evaluation log at: $BASE_EXP_DIR/$EXP_NAME/log_eval.txt"
echo "🖼️  Rendered images saved to: $BASE_EXP_DIR/$EXP_NAME/test_preds/"
echo ""
echo "🎯 Expected PSNR: ~32-37 (standard ZipNeRF performance)" 