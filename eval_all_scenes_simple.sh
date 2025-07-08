#!/bin/bash

# Simple ZipNeRF Batch Evaluation Script
# Usage: ./eval_all_scenes_simple.sh [--triplane true/false] [--scenes "scene1 scene2"]

set -e

# Default configuration
DEFAULT_TRIPLANE=true
ALL_SCENES=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

# Parse arguments
USE_TRIPLANE="$DEFAULT_TRIPLANE"
SCENES=(${ALL_SCENES[@]})

while [[ $# -gt 0 ]]; do
    case $1 in
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
        *)
            echo "❌ Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set mode name
if [ "$USE_TRIPLANE" = true ]; then
    MODE_NAME="TRIPLANE"
else
    MODE_NAME="BASELINE"
fi

# Ensure we're in the correct directory
cd /home/nilkel/Projects/zipnerf-pytorch

echo "🧪 Simple ZipNeRF Batch Evaluation"
echo "=================================="
echo "  🔧 Mode: $MODE_NAME"
echo "  🎬 Scenes: ${SCENES[@]}"
echo ""

# Initialize counters
TOTAL_SCENES=${#SCENES[@]}
SUCCESSFUL_SCENES=()
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
    
    # Run evaluation for this scene
    echo "🏃 Evaluating $SCENE in $MODE_NAME mode..."
    
    if ./eval_zipnerf_v2.sh "$SCENE" --triplane "$USE_TRIPLANE"; then
        echo "✅ Successfully evaluated $SCENE"
        SUCCESSFUL_SCENES+=("$SCENE")
    else
        echo "❌ Failed to evaluate $SCENE"
        FAILED_SCENES+=("$SCENE")
    fi
done

# Print final summary
echo ""
echo "🏁 Simple Batch Evaluation Complete!"
echo "===================================="
echo "  📊 Total scenes: $TOTAL_SCENES"
echo "  ✅ Successfully evaluated: ${#SUCCESSFUL_SCENES[@]}"
echo "  ❌ Failed: ${#FAILED_SCENES[@]}"
echo ""

if [ ${#SUCCESSFUL_SCENES[@]} -gt 0 ]; then
    echo "✅ Successfully evaluated scenes:"
    for scene in "${SUCCESSFUL_SCENES[@]}"; do
        echo "   - $scene"
    done
fi

if [ ${#FAILED_SCENES[@]} -gt 0 ]; then
    echo "❌ Failed scenes:"
    for scene in "${FAILED_SCENES[@]}"; do
        echo "   - $scene"
    done
fi

echo ""
echo "🎯 Mode: $MODE_NAME"
echo "📊 Check individual evaluation logs in: exp/*/log_eval.txt"
echo "🖼️  Rendered images saved to: exp/*/test_preds/" 