#!/bin/bash

# ZipNeRF Environment Activation Script for RTX 5090 (Conda version)
echo "🚀 Activating ZipNeRF conda environment for RTX 5090..."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zipconda

# Set up library paths for PyTorch CUDA extensions
export LD_LIBRARY_PATH="/home/nilkel/miniconda3/envs/zipconda/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Add CUDA extensions to Python path
export PYTHONPATH="/home/nilkel/Projects/zipnerf-pytorch/extensions/cuda:$PYTHONPATH"

# RTX 5090 compatibility fixes
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export CUDA_ARCH="90"
export CUDA_VISIBLE_DEVICES=0

echo "✅ Environment activated!"
echo "📍 Current directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "🔥 PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "🎮 GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo ""
echo "Ready to train! Use the following commands:"
echo "• accelerate config  (first time setup)"  
echo "• accelerate launch train.py [args]  (training)"
echo "" 