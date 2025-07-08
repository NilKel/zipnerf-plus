#!/bin/bash

# General ZipNeRF Training Script
# Usage: ./train_zipnerf_v2.sh <scene_name> [options]
# Example: ./train_zipnerf_v2.sh lego --triplane true --steps 25000

set -e  # Exit on error

# Default configuration
DEFAULT_DATA_BASE="/home/nilkel/Projects/data/nerf_synthetic"
DEFAULT_CONFIG="configs/blender.gin"
DEFAULT_WANDB_PROJECT="zipnerf-experiments"
DEFAULT_STEPS=25000
DEFAULT_BATCH_SIZE=8192
DEFAULT_FACTOR=4
DEFAULT_TRIPLANE=true

# Help function
show_help() {
    echo "🚀 ZipNeRF Training Script"
    echo "=========================="
    echo ""
    echo "Usage: $0 <scene_name> [options]"
    echo ""
    echo "Required:"
    echo "  scene_name           Scene to train on (e.g., lego, chair, drums)"
    echo ""
    echo "Options:"
    echo "  --data_dir PATH      Base data directory (default: $DEFAULT_DATA_BASE)"
    echo "  --config PATH        Gin config file (default: $DEFAULT_CONFIG)"
    echo "  --steps NUM          Training steps (default: $DEFAULT_STEPS)"
    echo "  --batch_size NUM     Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  --factor NUM         Downsampling factor (default: $DEFAULT_FACTOR)"
    echo "  --triplane BOOL      Enable triplane integration: true/false (default: $DEFAULT_TRIPLANE)"
    echo "  --wandb_project STR  Wandb project name (default: $DEFAULT_WANDB_PROJECT)"
    echo "  --exp_name STR       Custom experiment name (default: auto-generated)"
    echo "  --disable_wandb      Disable wandb logging"
    echo "  --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 lego                                    # Triplane enabled (default)"
    echo "  $0 lego --triplane false                   # Baseline ZipNeRF"
    echo "  $0 chair --triplane true --steps 50000     # Long triplane training"
    echo "  $0 drums --triplane false --wandb_project baseline-experiments"
    echo "  $0 ficus --exp_name ficus_comparison_v1"
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
MAX_STEPS="$DEFAULT_STEPS"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
FACTOR="$DEFAULT_FACTOR"
WANDB_PROJECT="$DEFAULT_WANDB_PROJECT"
EXP_NAME=""
USE_TRIPLANE="$DEFAULT_TRIPLANE"
USE_WANDB=true

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
        --steps)
            MAX_STEPS="$2"
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
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --disable_wandb)
            USE_WANDB=false
            shift
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Generate experiment name if not provided
if [ -z "$EXP_NAME" ]; then
    TIMESTAMP=$(date +"%m%d_%H%M")
    if [ "$USE_TRIPLANE" = true ]; then
        EXP_NAME="${SCENE_NAME}_triplane_${MAX_STEPS}_${TIMESTAMP}"
    else
        EXP_NAME="${SCENE_NAME}_baseline_${MAX_STEPS}_${TIMESTAMP}"
    fi
fi

# Construct full data path
DATA_DIR="$DATA_BASE/$SCENE_NAME"

# Ensure we're in the correct directory
cd /home/nilkel/Projects/zipnerf-pytorch

# Activate the environment
echo "🔧 Activating ZipNeRF environment..."
eval "$(conda shell.bash hook)"
conda activate zipnerf2

# Clear GPU cache
echo "🧹 Clearing GPU cache..."
python -c "import torch; torch.cuda.empty_cache()"

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory does not exist: $DATA_DIR"
    echo "Available scenes in $DATA_BASE:"
    ls -1 "$DATA_BASE" 2>/dev/null || echo "  (directory not found)"
    exit 1
fi

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check GPU memory
echo "🖥️  GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Print configuration summary
echo ""
echo "🚀 ZipNeRF Training Configuration"
echo "================================="
echo "  🎬 Scene: $SCENE_NAME"
echo "  📁 Data directory: $DATA_DIR"
echo "  🏷️  Experiment name: $EXP_NAME"
echo "  📋 Config file: $CONFIG_FILE"
echo "  🎯 Max steps: $MAX_STEPS"
echo "  📦 Batch size: $BATCH_SIZE"
echo "  📏 Downsampling factor: $FACTOR"
if [ "$USE_TRIPLANE" = true ]; then
    echo "  🔧 Mode: TRIPLANE (enhanced)"
else
    echo "  🔧 Mode: BASELINE (standard ZipNeRF)"
fi
echo "  📊 Wandb enabled: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "  📈 Wandb project: $WANDB_PROJECT"
fi
echo ""

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Convert boolean to Python format for Gin
if [ "$USE_TRIPLANE" = true ]; then
    TRIPLANE_VALUE="True"
else
    TRIPLANE_VALUE="False"
fi

# Build gin bindings
GIN_BINDINGS=(
    "--gin_bindings=Config.data_dir = '$DATA_DIR'"
    "--gin_bindings=Config.exp_name = '$EXP_NAME'"
    "--gin_bindings=Config.max_steps = $MAX_STEPS"
    "--gin_bindings=Config.batch_size = $BATCH_SIZE"
    "--gin_bindings=Config.factor = $FACTOR"
    "--gin_bindings=Config.use_triplane = $TRIPLANE_VALUE"
    "--gin_bindings=Config.gradient_scaling = True"
    "--gin_bindings=Config.train_render_every = 1000"
)

if [ "$USE_WANDB" = true ]; then
    GIN_BINDINGS+=(
        "--gin_bindings=Config.use_wandb = True"
        "--gin_bindings=Config.wandb_project = '$WANDB_PROJECT'"
        "--gin_bindings=Config.wandb_name = '$EXP_NAME'"
    )
else
    GIN_BINDINGS+=(
        "--gin_bindings=Config.use_wandb = False"
    )
fi

# Run training
echo "🏃 Starting training..."
accelerate launch train.py \
    --gin_configs="$CONFIG_FILE" \
    "${GIN_BINDINGS[@]}"

echo ""
echo "✅ Training completed!"
echo "📊 Check results:"
echo "  📁 Experiment directory: exp/$EXP_NAME"
echo "  📈 Training log: exp/$EXP_NAME/log_train.txt"
if [ "$USE_WANDB" = true ]; then
    echo "  🌐 Wandb: https://wandb.ai/YOUR_USERNAME/$WANDB_PROJECT"
fi 