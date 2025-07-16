#!/bin/bash

# Simple test script for ZipNeRF training
# Run this directly from terminal: bash test_training_direct.sh

echo "🧪 Testing ZipNeRF training setup..."

# Activate environment
echo "🔧 Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zipnerf2

# Verify environment
echo "📋 Environment check:"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Test simple training command (just a few steps)
echo "🚀 Testing training command..."
echo "Scene: lego"
echo "Comment: test_run_256res"

# Simple training command with minimal steps for testing
accelerate launch train.py \
    --gin_configs=configs/potential_triplane.gin \
    --gin_bindings="Config.data_dir = '/home/nilkel/Projects/data/nerf_synthetic/lego'" \
    --gin_bindings="Config.debug_confidence_grid_path = ''" \
    --gin_bindings="Config.freeze_debug_confidence = False" \
    --gin_bindings="Config.binary_occupancy = False" \
    --gin_bindings="Config.confidence_grid_resolution = (256, 256, 256)" \
    --gin_bindings="Config.comment = 'test_run_256res'" \
    --gin_bindings="Config.max_steps = 100" \
    --gin_bindings="Config.checkpoint_every = 50"

echo "✅ Test completed!" 