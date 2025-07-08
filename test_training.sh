#!/bin/bash

# Quick test script to verify ZipNeRF training setup
# Runs just 10 steps to verify everything is working

set -e  # Exit on error

# Configuration
DATA_DIR="/home/nilkel/Projects/data/nerf_synthetic/lego"
EXP_NAME="test_lego_setup"
WANDB_PROJECT="my-blender-experiments"
MAX_STEPS=10  # Just 10 steps for testing
BATCH_SIZE=1024  # Smaller batch for testing

echo "🧪 Testing ZipNeRF training setup..."
echo "  📁 Data directory: $DATA_DIR"
echo "  🏷️  Experiment name: $EXP_NAME"
echo "  📊 Wandb project: $WANDB_PROJECT"
echo "  🎯 Max steps: $MAX_STEPS (test mode)"
echo "  📦 Batch size: $BATCH_SIZE"
echo ""

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory does not exist: $DATA_DIR"
    echo "Please check the path and try again."
    exit 1
fi

# Check if we have the required files
if [ ! -f "train.py" ]; then
    echo "❌ Error: train.py not found in current directory"
    exit 1
fi

if [ ! -f "configs/blender.gin" ]; then
    echo "❌ Error: configs/blender.gin not found"
    exit 1
fi

echo "✅ All files and directories found!"
echo ""

# Activate conda environment
echo "🔧 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate zipnerf2

echo "🔍 Checking Python environment..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits
fi

echo ""
echo "🏃 Running test training (10 steps)..."
echo "⏱️  This should take about 1-2 minutes..."
echo ""

# Run test training
accelerate launch train.py \
    --gin_configs=configs/blender.gin \
    --gin_bindings="Config.data_dir = '$DATA_DIR'" \
    --gin_bindings="Config.exp_name = '$EXP_NAME'" \
    --gin_bindings="Config.wandb_project = '$WANDB_PROJECT'" \
    --gin_bindings="Config.max_steps = $MAX_STEPS" \
    --gin_bindings="Config.batch_size = $BATCH_SIZE" \
    --gin_bindings="Config.factor = 4" \
    --gin_bindings="Config.use_wandb = True" \
    --gin_bindings="Config.use_triplane = True" \
    --gin_bindings="Config.print_every = 1" \
    --gin_bindings="Config.checkpoint_every = 1000"

echo ""
echo "✅ Test completed successfully!"
echo "🎉 Your setup is working correctly!"
echo ""
echo "📊 Check your wandb project at: https://wandb.ai/YOUR_USERNAME/$WANDB_PROJECT"
echo "📁 Check experiment folder: exp/$EXP_NAME"
echo ""
echo "🚀 To run full training (25k steps), use: ./train_lego.sh" 