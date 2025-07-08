#!/bin/bash

# General ZipNeRF Evaluation Script
# Usage: ./eval_zipnerf_v2.sh <scene_name> [options]
# Example: ./eval_zipnerf_v2.sh lego --triplane true

set -e  # Exit on error

# Default configuration
DEFAULT_DATA_BASE="/home/nilkel/Projects/data/nerf_synthetic"
DEFAULT_CONFIG="configs/blender.gin"
DEFAULT_BATCH_SIZE=8192
DEFAULT_FACTOR=4
DEFAULT_TRIPLANE=true

# Help function
show_help() {
    echo "🧪 ZipNeRF Evaluation Script"
    echo "============================"
    echo ""
    echo "Usage: $0 <scene_name> [options]"
    echo ""
    echo "Required:"
    echo "  scene_name           Scene to evaluate (e.g., lego, chair, drums)"
    echo ""
    echo "Options:"
    echo "  --data_dir PATH      Base data directory (default: $DEFAULT_DATA_BASE)"
    echo "  --config PATH        Gin config file (default: $DEFAULT_CONFIG)"
    echo "  --batch_size NUM     Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  --factor NUM         Downsampling factor (default: $DEFAULT_FACTOR)"
    echo "  --triplane BOOL      Triplane mode: true/false (default: $DEFAULT_TRIPLANE)"
    echo "  --exp_name STR       Specific experiment name (default: auto-detect latest)"
    echo "  --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 lego --triplane true              # Evaluate latest triplane model"
    echo "  $0 lego --triplane false             # Evaluate latest baseline model"
    echo "  $0 chair --triplane true --factor 2  # Different factor"
    echo "  $0 drums --exp_name drums_custom_exp # Specific experiment"
}

# Parse arguments
if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

SCENE_NAME="$1"
shift

# Initialize variables with defaults
DATA_BASE="$DEFAULT_DATA_BASE"
CONFIG_FILE="$DEFAULT_CONFIG"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
FACTOR="$DEFAULT_FACTOR"
USE_TRIPLANE="$DEFAULT_TRIPLANE"
EXP_NAME=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_BASE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --factor)
            FACTOR="$2"
            shift 2
            ;;
        --triplane)
            case "$2" in
                true|True|TRUE|1|yes|Yes|YES)
                    USE_TRIPLANE=true
                    ;;
                false|False|FALSE|0|no|No|NO)
                    USE_TRIPLANE=false
                    ;;
                *)
                    echo "❌ Error: --triplane must be true or false, got: $2"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Construct full data path
DATA_DIR="$DATA_BASE/$SCENE_NAME"
BASE_EXP_DIR="exp"

# Auto-detect experiment if not specified
if [ -z "$EXP_NAME" ]; then
    if [ "$USE_TRIPLANE" = true ]; then
        SEARCH_PATTERN="${SCENE_NAME}_triplane_"
        MODE_NAME="triplane"
    else
        SEARCH_PATTERN="${SCENE_NAME}_baseline_"
        MODE_NAME="baseline"
    fi
    
    # Find the latest experiment matching the pattern
    LATEST_EXP=$(ls -1t "$BASE_EXP_DIR" 2>/dev/null | grep "$SEARCH_PATTERN" | head -1)
    
    if [ -z "$LATEST_EXP" ]; then
        echo "❌ Error: No $MODE_NAME experiments found for $SCENE_NAME"
        echo "Available experiments:"
        ls -1 "$BASE_EXP_DIR" 2>/dev/null | grep "$SCENE_NAME" || echo "  (none found)"
        exit 1
    fi
    
    EXP_NAME="$LATEST_EXP"
fi

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

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory does not exist: $DATA_DIR"
    echo "Available scenes in $DATA_BASE:"
    ls -1 "$DATA_BASE" 2>/dev/null || echo "  (directory not found)"
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

# Convert boolean to Python format for Gin
if [ "$USE_TRIPLANE" = true ]; then
    TRIPLANE_VALUE="True"
    MODE_NAME="TRIPLANE"
else
    TRIPLANE_VALUE="False"
    MODE_NAME="BASELINE"
fi

# Print configuration summary
echo ""
echo "🧪 ZipNeRF Evaluation Configuration"
echo "=================================="
echo "  🎬 Scene: $SCENE_NAME"
echo "  📁 Data directory: $DATA_DIR"
echo "  🏷️  Experiment: $EXP_NAME"
echo "  🔢 Latest checkpoint: $LATEST_CHECKPOINT"
echo "  📏 Factor: $FACTOR (downsampling)"
echo "  🔧 Mode: $MODE_NAME"
echo "  📦 Batch size: $BATCH_SIZE"
echo "  📊 Computing test set metrics..."
echo ""

# Run evaluation
echo "🏃 Running evaluation..."
accelerate launch eval.py \
    --gin_configs="$CONFIG_FILE" \
    --gin_bindings="Config.data_dir = '$DATA_DIR'" \
    --gin_bindings="Config.exp_name = '$EXP_NAME'" \
    --gin_bindings="Config.use_triplane = $TRIPLANE_VALUE" \
    --gin_bindings="Config.factor = $FACTOR" \
    --gin_bindings="Config.batch_size = $BATCH_SIZE" \
    --gin_bindings="Config.eval_only_once = True" \
    --gin_bindings="Config.eval_save_output = True" \
    --gin_bindings="Config.eval_quantize_metrics = True"

echo ""
echo "✅ Evaluation completed!"
echo "📊 Check the evaluation log at: $BASE_EXP_DIR/$EXP_NAME/log_eval.txt"
echo "🖼️  Rendered images saved to: $BASE_EXP_DIR/$EXP_NAME/test_preds/"
echo ""
if [ "$USE_TRIPLANE" = true ]; then
    echo "🎯 Expected PSNR: ~35-40 (triplane enhanced performance)"
else
    echo "🎯 Expected PSNR: ~30-35 (baseline ZipNeRF performance)"
fi 