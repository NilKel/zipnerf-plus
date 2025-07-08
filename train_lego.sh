#!/bin/bash

# ZipNeRF Training Script for Lego Scene
# This script sets up the environment and runs training with wandb enabled

set -e  # Exit on error

# Configuration
SCENE="lego"
DATA_DIR="/home/nilkel/Projects/data/nerf_synthetic/lego"
EXP_NAME="lego_triplane_25k"
WANDB_PROJECT="my-blender-experiments"
MAX_STEPS=25000
BATCH_SIZE=65536
FACTOR=4

# Ensure we're in the correct directory
cd /home/nilkel/Projects/zipnerf-pytorch

# Activate the environment
echo "🔧 Activating ZipNeRF environment..."
eval "$(conda shell.bash hook)"
conda activate zipnerf2

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Print configuration
echo "🚀 Starting ZipNeRF training with the following configuration:"
echo "  📁 Data directory: $DATA_DIR"
echo "  🏷️  Experiment name: $EXP_NAME"
echo "  📊 Wandb project: $WANDB_PROJECT"
echo "  🎯 Max steps: $MAX_STEPS"
echo "  📦 Batch size: $BATCH_SIZE"
echo "  📏 Factor: $FACTOR"
echo "  🔬 Wandb enabled: YES"
echo ""

# Run training
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
    --gin_bindings="Config.use_triplane = True"

echo "✅ Training completed!" 