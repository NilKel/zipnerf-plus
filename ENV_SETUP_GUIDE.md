# Zip-NeRF PyTorch Environment Setup Guide

## Overview
This guide provides step-by-step instructions to set up the zipnerf-pytorch repository with occupancy gradients, distortion loss, and ADMM pruning features. The setup has been tested on NVIDIA RTX 5090 with CUDA 12.8.

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 5090)
- **RAM**: Minimum 32GB system RAM (64GB recommended for large datasets)
- **Storage**: SSD recommended for faster data loading

### Software
- **OS**: Linux (tested on Ubuntu 24.04.2)
- **CUDA**: 12.8 (driver version 570.144)
- **Python**: 3.9+ (tested with Python 3.9)

## Step 1: System Setup

### Install CUDA and NVIDIA Drivers
```bash
# Check current NVIDIA driver and CUDA version
nvidia-smi

# If CUDA is not installed, follow NVIDIA's official installation guide:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
```

### Install Miniconda
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Restart shell or source conda
source ~/.bashrc
```

## Step 2: Environment Setup

### Create Conda Environment
```bash
# Create new environment
conda create -n zipnerf2 python=3.9
conda activate zipnerf2

# Update conda
conda update conda
```

### Install PyTorch with CUDA Support
```bash
# Install PyTorch 2.7.1 with CUDA 12.8
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### Install Additional Dependencies
```bash
# Core ML dependencies
pip install numpy==2.0.2
pip install scipy==1.13.1
pip install matplotlib==3.9.4
pip install pillow==11.3.0

# Computer vision
pip install opencv-python==4.11.0.86
pip install opencv-contrib-python==4.11.0.86

# Deep learning utilities
pip install accelerate==1.8.1
pip install transformers
pip install tensorboard==2.19.0
pip install tensorboardx==2.6.4

# Logging and monitoring
pip install wandb==0.21.0

# Additional utilities
pip install tqdm
pip install gin-config
pip install imageio
pip install lpips
pip install pytorch3d
pip install pycolmap
```

### Install CUDA Extensions (if needed)
```bash
# The gridencoder extension should work out of the box with PyTorch 2.7.1
# If you encounter issues, you may need to rebuild:

# Navigate to the gridencoder directory
cd gridencoder

# If there are compilation issues, try:
# pip install ninja
# python setup.py build_ext --inplace
```

## Step 3: Repository Setup

### Clone and Setup Repository
```bash
# Clone the repository
git clone <your-repo-url>
cd zipnerf-pytorch

# Checkout the occupancy_grad branch
git checkout occupancy_grad

# Verify the gridencoder extension is present
ls -la gridencoder/
# Should show: grid.py, __init__.py, __pycache__/
```

### Verify Installation
```bash
# Test basic imports
python -c "
import torch
import numpy as np
import cv2
import accelerate
import wandb
import tensorboard
from gridencoder import grid
print('All imports successful!')
"
```

## Step 4: Dataset Preparation

### For LLFF Datasets (360° Scenes)
```bash
# Create data directory
mkdir -p /path/to/your/data

# Download and prepare LLFF dataset (e.g., bicycle)
# The dataset should have the following structure:
# /path/to/your/data/bicycle/
# ├── images/          # Original high-res images
# ├── images_2/        # 2x downsampled
# ├── images_4/        # 4x downsampled  
# ├── images_8/        # 8x downsampled
# └── sparse/0/        # COLMAP reconstruction
```

### For Blender Synthetic Datasets
```bash
# Download from https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
# Extract to /path/to/your/data/lego/ (or other scene names)
```

## Step 5: Configuration

### Basic Training Configuration
Create a config file (e.g., `configs/my_config.gin`):
```gin
# Basic settings
Config.data_dir = "/path/to/your/data/bicycle"
Config.factor = 8  # Use 8x downsampled images for memory efficiency
Config.batch_size = 4096
Config.max_steps = 25000

# Multiscale settings
Config.multiscale = True
Config.multiscale_levels = 2

# Confidence field settings
Config.use_potential = False
Config.confidence_grid_resolution = (128, 128, 128)
Config.confidence_distortion_loss_mult = 0.005
Config.use_admm_pruner = False

# Model settings
NerfMLP.use_positional_encoder = False
NerfMLP.grid_level_dim = 4
PropMLP.disable_density_normals = True
PropMLP.disable_rgb = True
```

## Step 6: Training

### Basic Training Command
```bash
# Activate environment
conda activate zipnerf2

# Start training
accelerate launch train.py \
  --gin_configs=configs/my_config.gin \
  --gin_bindings="Config.comment = 'my_experiment'"
```

### Memory-Efficient Training
```bash
# For large datasets or limited memory, use:
accelerate launch train.py \
  --gin_configs=configs/my_config.gin \
  --gin_bindings="Config.factor = 8" \
  --gin_bindings="Config.batch_size = 2048" \
  --gin_bindings="Config.multiscale_levels = 1"
```

## Step 7: Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
--gin_bindings="Config.batch_size = 1024"

# Use smaller factor (more downsampling)
--gin_bindings="Config.factor = 8"

# Reduce multiscale levels
--gin_bindings="Config.multiscale_levels = 1"
```

#### 2. Dataset Loading Killed
```bash
# This usually means system RAM is exhausted
# Use factor=8 instead of factor=4 or factor=0
--gin_bindings="Config.factor = 8"
```

#### 3. Gridencoder Import Error
```bash
# Check if gridencoder extension is properly installed
python -c "from gridencoder import grid; print('Gridencoder OK')"

# If failed, try rebuilding:
cd gridencoder
python setup.py build_ext --inplace
```

#### 4. PyTorch CUDA Version Mismatch
```bash
# Verify CUDA versions match
nvidia-smi  # Should show CUDA 12.8
python -c "import torch; print(torch.version.cuda)"  # Should show 12.8

# If mismatch, reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### Performance Optimization

#### For RTX 5090 (32GB VRAM)
```bash
# Optimal settings for RTX 5090
Config.batch_size = 4096
Config.factor = 8
Config.multiscale_levels = 2
```

#### For Smaller GPUs (8-16GB VRAM)
```bash
# Conservative settings
Config.batch_size = 1024
Config.factor = 8
Config.multiscale_levels = 1
```

## Step 8: Monitoring and Logging

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir=exp/

# Access at http://localhost:6006
```

### Weights & Biases
```bash
# Login to W&B
wandb login

# Training will automatically log to W&B if configured
```

## Environment Summary

### Current Working Setup
- **OS**: Ubuntu 24.04.2 (Linux 6.11.0-29-generic)
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **CUDA**: 12.8 (Driver 570.144)
- **Python**: 3.9
- **PyTorch**: 2.7.1+cu128
- **Conda Environment**: zipnerf2

### Key Features Implemented
- ✅ Occupancy gradients with confidence field
- ✅ Distortion loss for confidence field
- ✅ ADMM pruning for sparsity
- ✅ Multiscale training support
- ✅ Contraction-aware gradients
- ✅ Binary occupancy with STE
- ✅ Analytical gradient computation

### Memory Requirements
- **System RAM**: 32GB minimum, 64GB recommended
- **GPU VRAM**: 8GB minimum, 32GB recommended for full features
- **Storage**: SSD recommended for dataset loading

## Quick Start Checklist

- [ ] Install CUDA 12.8 and NVIDIA drivers
- [ ] Install Miniconda
- [ ] Create conda environment: `conda create -n zipnerf2 python=3.9`
- [ ] Install PyTorch 2.7.1+cu128
- [ ] Install all dependencies
- [ ] Clone repository and checkout occupancy_grad branch
- [ ] Prepare dataset with proper directory structure
- [ ] Create config file with appropriate settings
- [ ] Test training with small batch size first
- [ ] Monitor with TensorBoard or W&B

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify CUDA and PyTorch versions match
3. Ensure sufficient system RAM and GPU VRAM
4. Use `factor=8` for memory-efficient training
5. Start with smaller batch sizes and increase gradually

---

**Last Updated**: July 19, 2025  
**Tested On**: NVIDIA RTX 5090, CUDA 12.8, Ubuntu 24.04.2 