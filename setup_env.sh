#!/bin/bash

# ZipNeRF Environment Setup Script for RTX 5090
# This script activates the virtual environment and sets up the required library paths

echo "🚀 Activating ZipNeRF environment for RTX 5090..."

# Activate the virtual environment
source zipnerf-env/bin/activate

# Set up library paths for PyTorch CUDA extensions
export LD_LIBRARY_PATH="/home/nilkel/Projects/zipnerf-pytorch/zipnerf-env/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Add CUDA extensions to Python path
export PYTHONPATH="/home/nilkel/Projects/zipnerf-pytorch/extensions/cuda:$PYTHONPATH"

# Set CUDA architecture for compilation (in case needed later)
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

echo "✅ Environment activated!"
echo "📍 Current directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "🔥 PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "🎮 GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"

echo ""
echo "🎯 Ready to use ZipNeRF! Example commands:"
echo "   • Configure accelerate: accelerate config"
echo "   • Train on 360 data: accelerate launch train.py --gin_configs=configs/360.gin --gin_bindings=\"Config.data_dir='data/360_v2/bicycle'\" --gin_bindings=\"Config.exp_name='bicycle'\""
echo "   • Render: accelerate launch render.py --gin_configs=configs/360.gin --gin_bindings=\"Config.data_dir='data/360_v2/bicycle'\" --gin_bindings=\"Config.exp_name='bicycle'\""
echo "" 