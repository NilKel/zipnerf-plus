#!/bin/bash

# Batch ZipNeRF Potential Encoder Evaluation Script for All Scenes
# Usage: ./eval_all_potential.sh [options]
# Example: ./eval_all_potential.sh --triplane true --scenes "lego chair drums"

set -e  # Exit on error

# Default configuration
DEFAULT_DATA_BASE="/home/nilkel/Projects/data/nerf_synthetic"
DEFAULT_CONFIG="configs/blender.gin"
DEFAULT_BATCH_SIZE=8192
DEFAULT_FACTOR=4
DEFAULT_TRIPLANE=false

# Define all scenes (including lego)
ALL_SCENES=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

# Help function
show_help() {
    echo "🧪 ZipNeRF Potential Encoder Batch Evaluation Script"
    echo "===================================================="
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --data_base PATH     Base data directory (default: $DEFAULT_DATA_BASE)"
    echo "  --config PATH        Gin config file (default: $DEFAULT_CONFIG)"
    echo "  --batch_size NUM     Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  --factor NUM         Downsampling factor (default: $DEFAULT_FACTOR)"
    echo "  --triplane BOOL      Potential+triplane mode: true/false (default: $DEFAULT_TRIPLANE)"
    echo "  --scenes \"s1 s2\"     Custom scene list (default: all scenes)"
    echo "  --skip_existing      Skip scenes that already have evaluation results"
    echo "  --force              Force re-evaluation even if results exist"
    echo "  --help               Show this help"
    echo ""
    echo "Default scenes: ${ALL_SCENES[@]}"
    echo ""
    echo "Examples:"
    echo "  $0                                         # Evaluate all scenes with potential encoder"
    echo "  $0 --triplane true                        # Evaluate with potential+triplane"
    echo "  $0 --scenes \"lego chair drums\"            # Evaluate specific scenes"
    echo "  $0 --skip_existing                        # Skip already evaluated scenes"
    echo "  $0 --triplane true --force                # Force re-evaluation with triplane"
}

# Parse arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Initialize variables with defaults
DATA_BASE="$DEFAULT_DATA_BASE"
CONFIG_FILE="$DEFAULT_CONFIG"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
FACTOR="$DEFAULT_FACTOR"
USE_TRIPLANE="$DEFAULT_TRIPLANE"
SCENES=(${ALL_SCENES[@]})
SKIP_EXISTING=false
FORCE_EVAL=false

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
        --scenes)
            read -a SCENES <<< "$2"
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

# Set mode name for display
if [ "$USE_TRIPLANE" = true ]; then
    MODE_NAME="POTENTIAL+TRIPLANE"
    SEARCH_PATTERN="_potential_triplane_"
else
    MODE_NAME="POTENTIAL"
    SEARCH_PATTERN="_potential_"
fi

# Ensure we're in the correct directory
cd /home/nilkel/Projects/zipnerf-pytorch

# Validate data directory
if [ ! -d "$DATA_BASE" ]; then
    echo "❌ Error: Data directory does not exist: $DATA_BASE"
    exit 1
fi

# Kill any existing evaluation process
echo "🛑 Stopping any existing evaluation processes..."
pkill -f "eval.py" || true

# Check if conda environment is activated
echo "🔧 Checking conda environment..."
if [[ "$CONDA_DEFAULT_ENV" != "zipnerf2" ]]; then
    echo "⚠️  Warning: zipnerf2 environment not activated. Attempting to activate..."
    eval "$(conda shell.bash hook)"
    conda activate zipnerf2
else
    echo "✅ zipnerf2 environment is already activated"
fi

# Function to check if evaluation already exists
check_evaluation_exists() {
    local scene_name=$1
    local exp_dir="exp"
    
    # Find the latest experiment directory for this scene
    local latest_exp=""
    local latest_time=0
    
    for exp_path in "$exp_dir"/${scene_name}${SEARCH_PATTERN}*; do
        if [ -d "$exp_path" ]; then
            local exp_time=$(stat -c %Y "$exp_path" 2>/dev/null || echo "0")
            if [ "$exp_time" -gt "$latest_time" ]; then
                latest_time="$exp_time"
                latest_exp=$(basename "$exp_path")
            fi
        fi
    done
    
    if [ -z "$latest_exp" ]; then
        return 1  # No experiment found
    fi
    
    # Check if log_eval.txt exists
    if [ -f "$exp_dir/$latest_exp/log_eval.txt" ]; then
        echo "$latest_exp"
        return 0  # Evaluation exists
    fi
    
    return 1  # Evaluation not complete
}

# Function to check if experiment has checkpoints (completed training)
check_experiment_has_checkpoints() {
    local scene_name=$1
    local exp_dir="exp"
    
    # Find the latest experiment directory for this scene
    local latest_exp=""
    local latest_time=0
    
    for exp_path in "$exp_dir"/${scene_name}${SEARCH_PATTERN}*; do
        if [ -d "$exp_path" ]; then
            local exp_time=$(stat -c %Y "$exp_path" 2>/dev/null || echo "0")
            if [ "$exp_time" -gt "$latest_time" ]; then
                latest_time="$exp_time"
                latest_exp=$(basename "$exp_path")
            fi
        fi
    done
    
    if [ -z "$latest_exp" ]; then
        return 1  # No experiment found
    fi
    
    # Check for checkpoints
    if [ -d "$exp_dir/$latest_exp/checkpoints" ] && ls "$exp_dir/$latest_exp/checkpoints"/* >/dev/null 2>&1; then
        echo "$latest_exp"
        return 0  # Training completed
    fi
    
    return 1  # Training not complete
}

# Print configuration summary
echo ""
echo "🧪 ZipNeRF Potential Encoder Batch Evaluation"
echo "============================================="
echo "  🔧 Mode: $MODE_NAME"
echo "  📁 Data directory: $DATA_BASE"
echo "  📄 Config file: $CONFIG_FILE"
echo "  📏 Factor: $FACTOR (downsampling)"
echo "  📦 Batch size: $BATCH_SIZE"
echo "  🎬 Scenes: ${SCENES[@]}"
echo "  ⏭️  Skip existing: $SKIP_EXISTING"
echo "  🔄 Force evaluation: $FORCE_EVAL"
echo ""

# Initialize counters
TOTAL_SCENES=${#SCENES[@]}
EVALUATED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0
SUCCESSFUL_SCENES=()
SKIPPED_SCENES=()
FAILED_SCENES=()

# Start evaluation loop
echo "🏃 Starting batch evaluation of $TOTAL_SCENES scenes..."
echo "=================================================="

for i in "${!SCENES[@]}"; do
    SCENE="${SCENES[$i]}"
    SCENE_NUM=$((i + 1))
    
    echo ""
    echo "🎬 Scene [$SCENE_NUM/$TOTAL_SCENES]: $SCENE"
    echo "$(printf '=%.0s' {1..50})"
    
    # Check if scene data exists
    echo "🔍 Checking scene data directory..."
    if [ ! -d "$DATA_BASE/$SCENE" ]; then
        echo "⚠️  Warning: Scene data not found at $DATA_BASE/$SCENE, skipping..."
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        SKIPPED_SCENES+=("$SCENE (no data)")
        continue
    fi
    echo "✅ Scene data found at $DATA_BASE/$SCENE"
    
    # Check if experiment has completed training
    echo "🔍 Checking if potential training completed for $SCENE..."
    TRAINED_EXP=$(check_experiment_has_checkpoints "$SCENE")
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: No completed potential training found for $SCENE"
        echo "   This usually means the training was interrupted or failed"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        SKIPPED_SCENES+=("$SCENE (training incomplete)")
        continue
    fi
    echo "✅ Potential training completed for $SCENE ($TRAINED_EXP)"
    
    # Check if evaluation already exists
    echo "🔍 Checking if evaluation already exists..."
    if [ "$FORCE_EVAL" = false ]; then
        if EXISTING_EXP=$(check_evaluation_exists "$SCENE"); then
            if [ "$SKIP_EXISTING" = true ]; then
                echo "✅ Evaluation already exists for $SCENE ($EXISTING_EXP), skipping..."
                SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
                SKIPPED_SCENES+=("$SCENE (already evaluated)")
                continue
            else
                echo "⚠️  Warning: Evaluation already exists for $SCENE ($EXISTING_EXP)"
                echo "   Proceeding anyway (use --skip_existing to skip)"
            fi
        else
            echo "📝 No existing evaluation found for $SCENE"
        fi
    else
        echo "🔄 Force evaluation enabled, skipping existing check"
    fi
    
    # Run evaluation for this scene
    echo "🏃 Evaluating $SCENE in $MODE_NAME mode..."
    
    # Build the eval command arguments
    EVAL_ARGS=""
    if [ "$SKIP_EXISTING" = true ]; then
        EVAL_ARGS="$EVAL_ARGS --skip_existing"
    fi
    if [ "$FORCE_EVAL" = true ]; then
        EVAL_ARGS="$EVAL_ARGS --force"
    fi
    
    if ./eval_potential.sh "$SCENE" \
        --data_base "$DATA_BASE" \
        --config "$CONFIG_FILE" \
        --batch_size "$BATCH_SIZE" \
        --factor "$FACTOR" \
        --triplane "$USE_TRIPLANE" \
        $EVAL_ARGS; then
        
        echo "✅ Successfully evaluated $SCENE"
        EVALUATED_COUNT=$((EVALUATED_COUNT + 1))
        SUCCESSFUL_SCENES+=("$SCENE")
    else
        echo "❌ Failed to evaluate $SCENE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_SCENES+=("$SCENE")
    fi
done

# Print final summary
echo ""
echo "🏁 Batch Potential Encoder Evaluation Complete!"
echo "==============================================="
echo "  📊 Total scenes: $TOTAL_SCENES"
echo "  ✅ Successfully evaluated: $EVALUATED_COUNT"
echo "  ⏭️  Skipped: $SKIPPED_COUNT"
echo "  ❌ Failed: $FAILED_COUNT"
echo ""

if [ ${#SUCCESSFUL_SCENES[@]} -gt 0 ]; then
    echo "✅ Successfully evaluated scenes:"
    for scene in "${SUCCESSFUL_SCENES[@]}"; do
        echo "   - $scene"
    done
    echo ""
fi

if [ ${#SKIPPED_SCENES[@]} -gt 0 ]; then
    echo "⏭️  Skipped scenes:"
    for scene in "${SKIPPED_SCENES[@]}"; do
        echo "   - $scene"
    done
    echo ""
fi

if [ ${#FAILED_SCENES[@]} -gt 0 ]; then
    echo "❌ Failed scenes:"
    for scene in "${FAILED_SCENES[@]}"; do
        echo "   - $scene"
    done
    echo ""
fi

echo "🎯 Mode: $MODE_NAME"
if [ "$USE_TRIPLANE" = true ]; then
    echo "📈 Expected PSNR: ~30-40 (potential+triplane experimental performance)"
else
    echo "📈 Expected PSNR: ~30-35 (potential encoder experimental performance)"
fi
echo ""
echo "📊 Check individual evaluation logs in: exp/*/log_eval.txt"
echo "🖼️  Rendered images saved to: exp/*/test_preds/" 