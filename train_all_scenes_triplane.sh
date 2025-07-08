#!/bin/bash

# Train all NeRF synthetic scenes with triplane or baseline
# Usage: ./train_all_scenes_triplane.sh [options]

set -e  # Exit on error

# Default configuration
DEFAULT_DATA_BASE="/home/nilkel/Projects/data/nerf_synthetic"
DEFAULT_WANDB_PROJECT="zipnerf-comparison"
DEFAULT_STEPS=25000
DEFAULT_BATCH_SIZE=8192
DEFAULT_TRIPLANE=true

# Help function
show_help() {
    echo "🚀 ZipNeRF Training - All Scenes"
    echo "================================"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --data_dir PATH      Base data directory (default: $DEFAULT_DATA_BASE)"
    echo "  --wandb_project STR  Wandb project name (default: $DEFAULT_WANDB_PROJECT)"
    echo "  --steps NUM          Training steps per scene (default: $DEFAULT_STEPS)"
    echo "  --batch_size NUM     Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  --triplane BOOL      Enable triplane: true/false (default: $DEFAULT_TRIPLANE)"
    echo "  --skip_existing      Skip scenes that already have matching experiments"
    echo "  --scenes LIST        Comma-separated list of scenes to train (default: all)"
    echo "  --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Train all scenes with triplane"
    echo "  $0 --triplane false                  # Train all scenes baseline"
    echo "  $0 --wandb_project my-comparison      # Custom wandb project"
    echo "  $0 --scenes lego,chair,drums          # Train only specific scenes"
    echo "  $0 --triplane false --skip_existing   # Skip already trained baselines"
}

# Parse arguments
WANDB_PROJECT="$DEFAULT_WANDB_PROJECT"
DATA_BASE="$DEFAULT_DATA_BASE"
MAX_STEPS="$DEFAULT_STEPS"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
USE_TRIPLANE="$DEFAULT_TRIPLANE"
SKIP_EXISTING=false
CUSTOM_SCENES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_BASE="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
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
        --skip_existing)
            SKIP_EXISTING=true
            shift
            ;;
        --scenes)
            CUSTOM_SCENES="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Define all scenes (excluding lego - already trained)
ALL_SCENES=(chair drums ficus hotdog materials mic ship)

# Use custom scenes if provided
if [ -n "$CUSTOM_SCENES" ]; then
    IFS=',' read -ra SCENES <<< "$CUSTOM_SCENES"
else
    SCENES=("${ALL_SCENES[@]}")
fi

# Ensure we're in the correct directory
cd /home/nilkel/Projects/zipnerf-pytorch

# Verify training script exists
if [ ! -f "train_zipnerf_v2.sh" ]; then
    echo "❌ Error: train_zipnerf_v2.sh not found in current directory"
    exit 1
fi

# Make sure training script is executable
chmod +x train_zipnerf_v2.sh

# Print configuration
echo ""
if [ "$USE_TRIPLANE" = true ]; then
    echo "🚀 ZipNeRF Triplane Training - All Scenes"
    echo "========================================="
    MODE_LABEL="TRIPLANE (enhanced)"
else
    echo "🚀 ZipNeRF Baseline Training - All Scenes"
    echo "========================================="
    MODE_LABEL="BASELINE (standard)"
fi
echo "  📁 Data directory: $DATA_BASE"
echo "  📈 Wandb project: $WANDB_PROJECT"
echo "  🎯 Steps per scene: $MAX_STEPS"
echo "  📦 Batch size: $BATCH_SIZE"
echo "  🔧 Mode: $MODE_LABEL"
echo "  ⏭️  Skip existing: $SKIP_EXISTING"
echo "  🎬 Scenes to train: ${SCENES[*]}"
echo ""

# Initialize tracking
TOTAL_SCENES=${#SCENES[@]}
COMPLETED_SCENES=0
FAILED_SCENES=()
SUCCESS_SCENES=()
START_TIME=$(date +%s)

echo "📊 Training Progress: 0/$TOTAL_SCENES scenes completed"
echo "⏰ Started at: $(date)"
echo ""

# Train each scene
for scene in "${SCENES[@]}"; do
    echo "🎬 [$((COMPLETED_SCENES + 1))/$TOTAL_SCENES] Processing scene: $scene"
    echo "=================================================="
    
    # Check if scene directory exists
    if [ ! -d "$DATA_BASE/$scene" ]; then
        echo "⚠️  Warning: Scene directory not found: $DATA_BASE/$scene"
        echo "   Skipping $scene..."
        FAILED_SCENES+=("$scene (directory not found)")
        echo ""
        continue
    fi
    
    # Check if we should skip existing experiments
    if [ "$SKIP_EXISTING" = true ]; then
        if [ "$USE_TRIPLANE" = true ]; then
            SEARCH_PATTERN="${scene}_triplane_"
        else
            SEARCH_PATTERN="${scene}_baseline_"
        fi
        EXISTING_EXP=$(ls -1 exp/ 2>/dev/null | grep "$SEARCH_PATTERN" | head -1)
        if [ -n "$EXISTING_EXP" ]; then
            echo "⏭️  Skipping $scene (existing experiment found: $EXISTING_EXP)"
            SUCCESS_SCENES+=("$scene (skipped - existing)")
            COMPLETED_SCENES=$((COMPLETED_SCENES + 1))
            echo ""
            continue
        fi
    fi
    
    # Run training
    if [ "$USE_TRIPLANE" = true ]; then
        echo "🏃 Starting triplane training for $scene..."
    else
        echo "🏃 Starting baseline training for $scene..."
    fi
    SCENE_START_TIME=$(date +%s)
    
    if ./train_zipnerf_v2.sh "$scene" \
        --triplane "$USE_TRIPLANE" \
        --wandb_project "$WANDB_PROJECT" \
        --steps "$MAX_STEPS" \
        --batch_size "$BATCH_SIZE"; then
        
        SCENE_END_TIME=$(date +%s)
        SCENE_DURATION=$((SCENE_END_TIME - SCENE_START_TIME))
        SCENE_DURATION_MIN=$((SCENE_DURATION / 60))
        
        echo "✅ Successfully completed $scene in ${SCENE_DURATION_MIN} minutes"
        SUCCESS_SCENES+=("$scene")
        COMPLETED_SCENES=$((COMPLETED_SCENES + 1))
    else
        echo "❌ Failed to train $scene"
        FAILED_SCENES+=("$scene")
        
        # Ask if user wants to continue
        echo ""
        echo "⚠️  Training failed for $scene. Continue with remaining scenes? (y/N)"
        read -r -n 1 CONTINUE_CHOICE
        echo ""
        if [[ ! "$CONTINUE_CHOICE" =~ ^[Yy]$ ]]; then
            echo "🛑 Stopping training process..."
            break
        fi
    fi
    
    echo ""
    echo "📊 Progress: $COMPLETED_SCENES/$TOTAL_SCENES scenes completed"
    echo ""
done

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_DURATION_MIN=$((TOTAL_DURATION / 60))
TOTAL_DURATION_HOUR=$((TOTAL_DURATION / 3600))

echo ""
echo "🏁 Training Summary"
echo "=================="
echo "  ⏰ Total time: ${TOTAL_DURATION_HOUR}h ${TOTAL_DURATION_MIN}m"
echo "  ✅ Successful: ${#SUCCESS_SCENES[@]} scenes"
echo "  ❌ Failed: ${#FAILED_SCENES[@]} scenes"
echo ""

if [ ${#SUCCESS_SCENES[@]} -gt 0 ]; then
    echo "✅ Successfully trained scenes:"
    for scene in "${SUCCESS_SCENES[@]}"; do
        echo "  • $scene"
    done
    echo ""
fi

if [ ${#FAILED_SCENES[@]} -gt 0 ]; then
    echo "❌ Failed scenes:"
    for scene in "${FAILED_SCENES[@]}"; do
        echo "  • $scene"
    done
    echo ""
fi

echo "📈 Check all results in Wandb project: $WANDB_PROJECT"
if [ "$USE_TRIPLANE" = true ]; then
    echo "📁 Model checkpoints saved in: exp/SCENE_triplane_${MAX_STEPS}_*/"
else
    echo "📁 Model checkpoints saved in: exp/SCENE_baseline_${MAX_STEPS}_*/"
fi
echo ""

# Return appropriate exit code
if [ ${#FAILED_SCENES[@]} -eq 0 ]; then
    echo "🎉 All scenes completed successfully!"
    exit 0
else
    echo "⚠️  Some scenes failed. Check the summary above."
    exit 1
fi 