#!/bin/bash

# General ZipNeRF Evaluation Script
# Usage: ./eval_general.sh <scene_name> [options]
# Example: ./eval_general.sh lego --potential true --triplane false

set -e  # Exit on error

# Default configuration
DEFAULT_DATA_BASE="/home/nilkel/Projects/data/nerf_synthetic"
DEFAULT_CONFIG="configs/blender.gin"
DEFAULT_BATCH_SIZE=8192
DEFAULT_FACTOR=4

# Help function
show_help() {
    echo "🧪 General ZipNeRF Evaluation Script"
    echo "===================================="
    echo ""
    echo "Usage: $0 <scene_name> [options]"
    echo ""
    echo "Required:"
    echo "  scene_name           Scene to evaluate (e.g., lego, chair, drums)"
    echo ""
    echo "Options:"
    echo "  --data_base PATH     Base data directory (default: $DEFAULT_DATA_BASE)"
    echo "  --config PATH        Gin config file (default: $DEFAULT_CONFIG)"
    echo "  --batch_size NUM     Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  --factor NUM         Downsampling factor (default: $DEFAULT_FACTOR)"
    echo "  --potential BOOL     Enable potential encoder: true/false (default: auto-detect)"
    echo "  --triplane BOOL      Enable triplane: true/false (default: auto-detect)"
    echo "  --exp_name STR       Specific experiment name (default: auto-detect latest)"
    echo "  --skip_existing      Skip if evaluation already exists"
    echo "  --force              Force re-evaluation even if results exist"
    echo "  --help               Show this help"
    echo ""
    echo "Mode Combinations:"
    echo "  --potential false --triplane false   # Baseline ZipNeRF"
    echo "  --potential false --triplane true    # ZipNeRF + Triplane"
    echo "  --potential true  --triplane false   # ZipNeRF + Potential"
    echo "  --potential true  --triplane true    # ZipNeRF + Potential + Triplane"
    echo ""
    echo "Automatic Detection (when not specified):"
    echo "  The script looks for experiments matching these patterns:"
    echo "  - *_baseline_* or *_zipnerf_* (baseline)"
    echo "  - *_triplane_* (triplane only)"
    echo "  - *_potential_* (potential only)"
    echo "  - *_potential_triplane_* (potential + triplane)"
    echo ""
    echo "Examples:"
    echo "  $0 lego                                    # Auto-detect latest experiment"
    echo "  $0 lego --potential false --triplane false # Baseline ZipNeRF"
    echo "  $0 lego --potential true --triplane true   # Full potential+triplane"
    echo "  $0 chair --data_base /path/to/data         # Custom data directory"
    echo "  $0 drums --config configs/llff.gin        # Different config file"
    echo "  $0 lego --exp_name lego_custom             # Specific experiment"
    echo "  $0 lego --skip_existing                    # Skip if already evaluated"
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
USE_POTENTIAL=""  # Empty means auto-detect
USE_TRIPLANE=""   # Empty means auto-detect
EXP_NAME=""
SKIP_EXISTING=false
FORCE_EVAL=false

# Function to parse boolean
parse_boolean() {
    case "$1" in
        true|True|TRUE|1|yes|Yes|YES)
            echo "true"
            ;;
        false|False|FALSE|0|no|No|NO)
            echo "false"
            ;;
        *)
            echo "❌ Error: Expected true or false, got: $1"
            exit 1
            ;;
    esac
}

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_base)
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
        --potential)
            USE_POTENTIAL=$(parse_boolean "$2")
            shift 2
            ;;
        --triplane)
            USE_TRIPLANE=$(parse_boolean "$2")
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --skip_existing)
            SKIP_EXISTING=true
            shift
            ;;
        --force)
            FORCE_EVAL=true
            shift
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Conflict check
if [ "$SKIP_EXISTING" = true ] && [ "$FORCE_EVAL" = true ]; then
    echo "❌ Error: --skip_existing and --force are mutually exclusive"
    exit 1
fi

# Construct full data path
DATA_DIR="$DATA_BASE/$SCENE_NAME"
BASE_EXP_DIR="exp"

# Auto-detect experiment and mode if not specified
if [ -z "$EXP_NAME" ]; then
    # If both potential and triplane are specified, use them for search
    if [ -n "$USE_POTENTIAL" ] && [ -n "$USE_TRIPLANE" ]; then
        if [ "$USE_POTENTIAL" = "true" ] && [ "$USE_TRIPLANE" = "true" ]; then
            SEARCH_PATTERNS=("${SCENE_NAME}_potential_triplane_" "potential_triplane_")
            MODE_NAME="potential+triplane"
        elif [ "$USE_POTENTIAL" = "true" ] && [ "$USE_TRIPLANE" = "false" ]; then
            SEARCH_PATTERNS=("${SCENE_NAME}_potential_" "potential_")
            MODE_NAME="potential"
        elif [ "$USE_POTENTIAL" = "false" ] && [ "$USE_TRIPLANE" = "true" ]; then
            SEARCH_PATTERNS=("${SCENE_NAME}_triplane_" "triplane_")
            MODE_NAME="triplane"
        else # both false
            SEARCH_PATTERNS=("${SCENE_NAME}_baseline_" "${SCENE_NAME}_zipnerf_" "baseline_" "zipnerf_")
            MODE_NAME="baseline"
        fi
    else
        # Auto-detect mode from available experiments
        SEARCH_PATTERNS=(
            "${SCENE_NAME}_potential_triplane_" "potential_triplane_"
            "${SCENE_NAME}_potential_" "potential_"
            "${SCENE_NAME}_triplane_" "triplane_"
            "${SCENE_NAME}_baseline_" "${SCENE_NAME}_zipnerf_" "baseline_" "zipnerf_"
            "${SCENE_NAME}_"
        )
        MODE_NAME="auto-detected"
    fi
    
    # Find the latest experiment matching any pattern
    LATEST_EXP=""
    LATEST_TIME=0
    for pattern in "${SEARCH_PATTERNS[@]}"; do
        for exp_path in "$BASE_EXP_DIR"/$pattern*; do
            if [ -d "$exp_path" ]; then
                exp_time=$(stat -c %Y "$exp_path" 2>/dev/null || echo "0")
                if [ "$exp_time" -gt "$LATEST_TIME" ]; then
                    LATEST_TIME="$exp_time"
                    LATEST_EXP=$(basename "$exp_path")
                fi
            fi
        done
        # If we found something, break (respect priority order)
        if [ -n "$LATEST_EXP" ]; then
            break
        fi
    done
    
    if [ -z "$LATEST_EXP" ]; then
        echo "❌ Error: No experiments found for $SCENE_NAME"
        echo ""
        echo "🔍 Available experiments for $SCENE_NAME:"
        ls -1 "$BASE_EXP_DIR" 2>/dev/null | grep "$SCENE_NAME" || echo "  (none found)"
        echo ""
        echo "🔍 All available experiments:"
        ls -1 "$BASE_EXP_DIR" 2>/dev/null | head -10 || echo "  (none found)"
        exit 1
    fi
    
    EXP_NAME="$LATEST_EXP"
    
    # Auto-detect settings from experiment name if not specified
    if [ -z "$USE_POTENTIAL" ] || [ -z "$USE_TRIPLANE" ]; then
        if [[ "$EXP_NAME" == *"potential_triplane"* ]]; then
            [ -z "$USE_POTENTIAL" ] && USE_POTENTIAL="true"
            [ -z "$USE_TRIPLANE" ] && USE_TRIPLANE="true"
            MODE_NAME="potential+triplane"
        elif [[ "$EXP_NAME" == *"potential"* ]]; then
            [ -z "$USE_POTENTIAL" ] && USE_POTENTIAL="true"
            [ -z "$USE_TRIPLANE" ] && USE_TRIPLANE="false"
            MODE_NAME="potential"
        elif [[ "$EXP_NAME" == *"triplane"* ]]; then
            [ -z "$USE_POTENTIAL" ] && USE_POTENTIAL="false"
            [ -z "$USE_TRIPLANE" ] && USE_TRIPLANE="true"
            MODE_NAME="triplane"
        else
            # Default to baseline
            [ -z "$USE_POTENTIAL" ] && USE_POTENTIAL="false"
            [ -z "$USE_TRIPLANE" ] && USE_TRIPLANE="false"
            MODE_NAME="baseline"
        fi
    fi
fi

# Set default values if still not specified
[ -z "$USE_POTENTIAL" ] && USE_POTENTIAL="false"
[ -z "$USE_TRIPLANE" ] && USE_TRIPLANE="false"

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
    echo ""
    echo "🔍 Available scenes in $DATA_BASE:"
    ls -1 "$DATA_BASE" 2>/dev/null || echo "  (directory not found)"
    echo ""
    echo "💡 Tip: Use --data_base to specify a different base directory"
    exit 1
fi

# Verify experiment directory exists
if [ ! -d "$BASE_EXP_DIR/$EXP_NAME" ]; then
    echo "❌ Error: Experiment directory not found: $BASE_EXP_DIR/$EXP_NAME"
    echo ""
    echo "🔍 Available experiments:"
    ls -1 "$BASE_EXP_DIR" 2>/dev/null | head -10 || echo "  (none found)"
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

# Check if evaluation already exists
EVAL_LOG="$BASE_EXP_DIR/$EXP_NAME/log_eval.txt"
if [ -f "$EVAL_LOG" ]; then
    if [ "$SKIP_EXISTING" = true ]; then
        echo "✅ Evaluation already exists for $EXP_NAME, skipping..."
        echo "📄 Existing log: $EVAL_LOG"
        exit 0
    elif [ "$FORCE_EVAL" = false ]; then
        echo "⚠️  Warning: Evaluation already exists for $EXP_NAME"
        echo "📄 Existing log: $EVAL_LOG"
        read -p "🤔 Do you want to re-run evaluation? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "✅ Skipping evaluation (use existing results)"
            exit 0
        fi
    fi
fi

# Determine mode description and expected PSNR
if [ "$USE_POTENTIAL" = "true" ] && [ "$USE_TRIPLANE" = "true" ]; then
    MODE_DISPLAY="POTENTIAL + TRIPLANE"
    EXPECTED_PSNR="~30-40"
elif [ "$USE_POTENTIAL" = "true" ] && [ "$USE_TRIPLANE" = "false" ]; then
    MODE_DISPLAY="POTENTIAL ENCODER"
    EXPECTED_PSNR="~30-35"
elif [ "$USE_POTENTIAL" = "false" ] && [ "$USE_TRIPLANE" = "true" ]; then
    MODE_DISPLAY="TRIPLANE ONLY"
    EXPECTED_PSNR="~32-38"
else
    MODE_DISPLAY="BASELINE ZIPNERF"
    EXPECTED_PSNR="~30-36"
fi

# Print configuration summary
echo ""
echo "🧪 General ZipNeRF Evaluation"
echo "============================="
echo "  🎬 Scene: $SCENE_NAME"
echo "  📁 Data directory: $DATA_DIR"
echo "  🏷️  Experiment: $EXP_NAME"
echo "  🔢 Latest checkpoint: $LATEST_CHECKPOINT"
echo "  📏 Factor: $FACTOR (downsampling)"
echo "  🔧 Mode: $MODE_DISPLAY"
echo "  🧬 Potential encoder: $USE_POTENTIAL"
echo "  📐 Triplane: $USE_TRIPLANE"
echo "  📦 Batch size: $BATCH_SIZE"
echo "  📊 Computing test set metrics..."
echo ""

# Convert boolean to Python format for Gin
if [ "$USE_POTENTIAL" = "true" ]; then
    POTENTIAL_VALUE="True"
else
    POTENTIAL_VALUE="False"
fi

if [ "$USE_TRIPLANE" = "true" ]; then
    TRIPLANE_VALUE="True"
else
    TRIPLANE_VALUE="False"
fi

# Run evaluation with specified settings
echo "🏃 Running evaluation..."
accelerate launch eval.py \
    --gin_configs="$CONFIG_FILE" \
    --gin_bindings="Config.data_dir = '$DATA_DIR'" \
    --gin_bindings="Config.exp_name = '$EXP_NAME'" \
    --gin_bindings="Config.use_potential = $POTENTIAL_VALUE" \
    --gin_bindings="Config.use_triplane = $TRIPLANE_VALUE" \
    --gin_bindings="Config.factor = $FACTOR" \
    --gin_bindings="Config.batch_size = $BATCH_SIZE" \
    --gin_bindings="Config.eval_only_once = True" \
    --gin_bindings="Config.eval_save_output = True" \
    --gin_bindings="Config.eval_quantize_metrics = True"

echo ""
echo "✅ General ZipNeRF evaluation completed!"
echo "📊 Check the evaluation log at: $EVAL_LOG"
echo "🖼️  Rendered images saved to: $BASE_EXP_DIR/$EXP_NAME/test_preds/"
echo ""
echo "🎯 Expected PSNR: $EXPECTED_PSNR ($MODE_DISPLAY)"
echo ""
echo "📈 Quick results view:"
echo "   tail $EVAL_LOG"
echo ""
echo "🔍 Compare with baseline:"
echo "   grep -A5 -B5 'PSNR\\|SSIM\\|LPIPS' $EVAL_LOG" 