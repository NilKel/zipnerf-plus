#!/bin/bash

# ZipNeRF Potential Encoder Evaluation Script for Lego Scene
# Evaluates the trained potential encoder model: lego_potential_26000_0709_1628

set -e  # Exit on error

# Configuration for your specific experiment
SCENE_NAME="lego"
EXP_NAME="lego_potential_26000_0709_1628"
DATA_DIR="/home/nilkel/Projects/data/nerf_synthetic/lego"
CONFIG_FILE="configs/blender.gin"
BATCH_SIZE=8192
FACTOR=4
BASE_EXP_DIR="exp"
CHECKPOINT_DIR="$BASE_EXP_DIR/$EXP_NAME/checkpoints"

# Ensure we're in the correct directory
cd /home/nilkel/Projects/zipnerf-pytorch

# Kill any existing evaluation process
echo "🛑 Stopping any existing evaluation processes..."
pkill -f "eval.py" || true

# Activate conda environment
echo "🔧 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate zipnerf2

# Verify experiment directory exists
if [ ! -d "$BASE_EXP_DIR/$EXP_NAME" ]; then
    echo "❌ Error: Experiment directory not found: $BASE_EXP_DIR/$EXP_NAME"
    echo "Available experiments:"
    ls -1 "$BASE_EXP_DIR" 2>/dev/null | grep "lego" || echo "  (none found)"
    exit 1
fi

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "ℹ️  This usually means the training was interrupted or failed"
    exit 1
fi

# Find the latest checkpoint
LATEST_CHECKPOINT=$(ls -1 "$CHECKPOINT_DIR" | sort -n | tail -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ Error: No checkpoints found in $CHECKPOINT_DIR"
    echo "ℹ️  This usually means the training was interrupted or failed"
    exit 1
fi

# Check if evaluation already exists (optional skip)
if [ -f "$BASE_EXP_DIR/$EXP_NAME/log_eval.txt" ]; then
    echo "⚠️  Warning: Evaluation already exists for $EXP_NAME"
    echo "📄 Existing log: $BASE_EXP_DIR/$EXP_NAME/log_eval.txt"
    read -p "🤔 Do you want to re-run evaluation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "✅ Skipping evaluation (use existing results)"
        exit 0
    fi
fi

# Print configuration summary
echo ""
echo "🧪 ZipNeRF Potential Encoder Evaluation"
echo "======================================"
echo "  🎬 Scene: $SCENE_NAME"
echo "  📁 Data directory: $DATA_DIR"
echo "  🏷️  Experiment: $EXP_NAME"
echo "  🔢 Latest checkpoint: $LATEST_CHECKPOINT"
echo "  📏 Factor: $FACTOR (downsampling)"
echo "  🔧 Mode: POTENTIAL ENCODER (triplane=False)"
echo "  📦 Batch size: $BATCH_SIZE"
echo "  📊 Computing test set metrics..."
echo ""

# Run evaluation with potential encoder settings
echo "🏃 Running evaluation..."
accelerate launch eval.py \
    --gin_configs="$CONFIG_FILE" \
    --gin_bindings="Config.data_dir = '$DATA_DIR'" \
    --gin_bindings="Config.exp_name = '$EXP_NAME'" \
    --gin_bindings="Config.use_potential = True" \
    --gin_bindings="Config.use_triplane = False" \
    --gin_bindings="Config.factor = $FACTOR" \
    --gin_bindings="Config.batch_size = $BATCH_SIZE" \
    --gin_bindings="Config.eval_only_once = True" \
    --gin_bindings="Config.eval_save_output = True" \
    --gin_bindings="Config.eval_quantize_metrics = True"

echo ""
echo "✅ Potential encoder evaluation completed!"
echo "📊 Check the evaluation log at: $BASE_EXP_DIR/$EXP_NAME/log_eval.txt"
echo "🖼️  Rendered images saved to: $BASE_EXP_DIR/$EXP_NAME/test_preds/"
echo ""
echo "🎯 Expected PSNR: ~30-35 (potential encoder performance - experimental)"
echo ""
echo "📈 To compare results:"
echo "   tail $BASE_EXP_DIR/$EXP_NAME/log_eval.txt"
echo "" 