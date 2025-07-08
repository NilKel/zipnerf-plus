#!/bin/bash

# ZipNeRF Training Script for Lego Scene - Memory Optimized
# This script uses smaller batch size to prevent CUDA out of memory errors

set -e  # Exit on error

# Configuration - Memory optimized
SCENE="lego"
DATA_DIR="/home/nilkel/Projects/data/nerf_synthetic/lego"
EXP_NAME="lego_triplane_25k_fixed"
WANDB_PROJECT="my-blender-experiments"
MAX_STEPS=25000
BATCH_SIZE=8192  # Reduced from 65536 to prevent OOM
FACTOR=4

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
    exit 1
fi

# Check GPU memory
echo "🖥️  GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Print configuration
echo ""
echo "🚀 Starting ZipNeRF training with MEMORY-OPTIMIZED configuration:"
echo "  📁 Data directory: $DATA_DIR"
echo "  🏷️  Experiment name: $EXP_NAME"
echo "  📊 Wandb project: $WANDB_PROJECT"
echo "  🎯 Max steps: $MAX_STEPS"
echo "  📦 Batch size: $BATCH_SIZE (reduced for memory efficiency)"
echo "  📏 Factor: $FACTOR"
echo "  🔬 Wandb enabled: YES"
echo "  🔧 Triplane enabled: YES"
echo "  ⚡ Memory optimizations: ENABLED"
echo ""

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# Run training with memory optimizations
echo "🏃 Starting training..."
accelerate launch train.py \
    --gin_configs=configs/blender.gin \
    --gin_bindings="Config.data_dir = '$DATA_DIR'" \
    --gin_bindings="Config.exp_name = '$EXP_NAME'" \
    --gin_bindings="Config.wandb_project = '$WANDB_PROJECT'" \
    --gin_bindings="Config.max_steps = $MAX_STEPS" \
    --gin_bindings="Config.batch_size = $BATCH_SIZE" \
    --gin_bindings="Config.factor = $FACTOR" \
    --gin_bindings="Config.use_wandb = True" \
    --gin_bindings="Config.use_triplane = True" \
    --gin_bindings="Config.gradient_scaling = True" \
    --gin_bindings="Config.train_render_every = 1000"

echo "✅ Training completed!" 