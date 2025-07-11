# ZipNeRF with Vector Potential Optimization - Setup Guide

This repository contains a modified version of ZipNeRF with vector potential optimization, binary occupancy implementation, and separate confidence field learning rates.

## Features

- **Vector Potential Optimization**: Enhanced NeRF training with potential field optimization
- **Binary Occupancy**: Optional hard 0/1 occupancy decisions with Straight-Through Estimator (STE)
- **Separate Confidence Learning Rates**: Independent learning rate control for confidence fields
- **Triplane Features**: Advanced feature encoding with triplane representations

## Environment Setup

### Prerequisites

- Linux system (tested on Ubuntu with kernel 6.11.0)
- NVIDIA GPU with CUDA support
- Miniconda or Anaconda installed

### 1. Create Conda Environment

```bash
# Create the environment
conda create -n zipnerf2 python=3.9.23

# Activate the environment
conda activate zipnerf2
```

### 2. Install Core Dependencies

```bash
# Add required channels
conda config --add channels pytorch
conda config --add channels nvidia
conda config --add channels conda-forge

# Install CUDA and PyTorch dependencies
conda install pytorch-cuda=12.4 cuda-cudart=12.4.127 cuda-libraries=12.4.1 -c nvidia -c pytorch

# Install core packages
conda install numpy=2.0.2 matplotlib scipy scikit-learn scikit-image
conda install ffmpeg=4.4.2 opencv-python=4.11.0.86
conda install mkl=2022.1.0 intel-openmp=2022.0.1
```

### 3. Install Python Packages via pip

```bash
# Core ML packages
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==0.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# NVIDIA CUDA packages
pip install nvidia-cublas-cu12==12.8.3.14
pip install nvidia-cuda-cupti-cu12==12.8.57
pip install nvidia-cuda-nvrtc-cu12==12.8.61
pip install nvidia-cuda-runtime-cu12==12.8.57
pip install nvidia-cudnn-cu12==9.7.1.26
pip install nvidia-cufft-cu12==11.3.3.41
pip install nvidia-curand-cu12==10.3.9.55
pip install nvidia-cusolver-cu12==11.7.2.55
pip install nvidia-cusparse-cu12==12.5.7.53
pip install nvidia-nccl-cu12==2.26.2
pip install nvidia-nvjitlink-cu12==12.8.61

# Graphics and geometry
pip install nvdiffrast==0.3.3
pip install trimesh==4.6.13
pip install pymeshlab==2023.12.post3
pip install xatlas==0.0.10

# Computer vision and imaging
pip install opencv-contrib-python==4.11.0.86
pip install imageio==2.37.0
pip install imageio-ffmpeg==0.6.0
pip install rawpy==0.25.0
pip install pillow==11.3.0
pip install tifffile==2024.8.30

# Machine learning utilities
pip install accelerate==1.8.1
pip install huggingface-hub==0.33.2
pip install safetensors==0.5.3
pip install torch-scatter==2.1.2+pt27cu128 --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Configuration and logging
pip install gin-config==0.5.0
pip install tensorboard==2.19.0
pip install tensorboardx==2.6.4
pip install wandb==0.21.0

# Utilities
pip install tqdm==4.67.1
pip install mediapy==1.2.4
pip install lpips==0.1.4
pip install plyfile==1.1.2
pip install psutil==7.0.0
pip install ninja==1.11.1.4

# Development tools
pip install ipython==8.18.1
pip install jupyter
```

### 4. Alternative: Use Environment File

If you prefer to use the exact environment file:

```bash
# Save the environment file (environment.yml)
conda env create -f environment.yml

# Or import from exported environment
conda env export --name zipnerf2 --no-builds > environment.yml
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import nvdiffrast; print('nvdiffrast: OK')"
python -c "import gin; print('gin-config: OK')"
```

## Usage

### Basic Training

```bash
# Activate environment
conda activate zipnerf2

# Train standard ZipNeRF
python train.py --gin_configs configs/zipnerf/360.gin --gin_bindings "Config.data_dir = 'path/to/your/data'" --gin_bindings "Config.checkpoint_dir = 'path/to/checkpoints'"

# Train with potential field optimization
python train.py --gin_configs configs/potential_triplane.gin --gin_bindings "Config.data_dir = 'path/to/your/data'"
```

### Binary Occupancy Training

```bash
# Train with binary occupancy (hard 0/1 decisions)
python train.py --gin_configs configs/potential_binary.gin --gin_bindings "Config.data_dir = 'path/to/your/data'"
```

### Separate Confidence Learning Rates

```bash
# Train with higher confidence field learning rate
python train.py --gin_configs configs/potential_binary_high_conf_lr.gin --gin_bindings "Config.data_dir = 'path/to/your/data'"

# Train with lower confidence field learning rate  
python train.py --gin_configs configs/potential_binary_low_conf_lr.gin --gin_bindings "Config.data_dir = 'path/to/your/data'"

# Train with confidence LR multiplier
python train.py --gin_configs configs/potential_binary_conf_multiplier.gin --gin_bindings "Config.data_dir = 'path/to/your/data'"
```

### Evaluation

```bash
# Evaluate trained model
python eval.py --gin_configs configs/zipnerf/360.gin --gin_bindings "Config.checkpoint_dir = 'path/to/checkpoints'" --gin_bindings "Config.data_dir = 'path/to/your/data'"
```

## Configuration Options

### Binary Occupancy
- `Config.binary_occupancy = True/False`: Enable/disable binary occupancy
- Uses Straight-Through Estimator (STE) for gradient flow

### Confidence Learning Rates
- `Config.confidence_lr_multiplier = 2.0`: Simple multiplier approach
- `Config.confidence_lr_init = 0.05`: Explicit initial learning rate
- `Config.confidence_lr_final = 0.005`: Explicit final learning rate
- `Config.confidence_lr_delay_steps = 5000`: Warmup steps for confidence field
- `Config.confidence_lr_delay_mult = 0.01`: Warmup multiplier

## Project Structure

```
zipnerf-pytorch/
├── internal/
│   ├── configs.py          # Configuration parameters
│   ├── models.py           # Main NeRF models
│   ├── field.py            # Confidence field with STE
│   ├── train_utils.py      # Training utilities with separate LR
│   └── ...
├── configs/
│   ├── potential_binary.gin                    # Binary occupancy config
│   ├── potential_binary_high_conf_lr.gin      # High confidence LR
│   ├── potential_binary_low_conf_lr.gin       # Low confidence LR
│   ├── potential_binary_conf_multiplier.gin   # LR multiplier
│   └── potential_triplane.gin                 # Triplane features
├── test_binary_occupancy.py                   # Binary occupancy tests
├── test_separate_confidence_lr.py             # LR separation tests
└── train.py                                   # Main training script
```

## Testing

Run the included test scripts to verify functionality:

```bash
# Test binary occupancy implementation
python test_binary_occupancy.py

# Test separate confidence learning rates
python test_separate_confidence_lr.py
```

## Troubleshooting

### CUDA Issues
If you encounter CUDA-related errors:
1. Verify NVIDIA driver compatibility with CUDA 12.4
2. Check that `torch.cuda.is_available()` returns `True`
3. Ensure all CUDA packages are from the same version (12.4/12.8)

### Memory Issues
- Reduce batch size in config files
- Use gradient checkpointing: `Config.gradient_checkpointing = True`
- Monitor GPU memory usage with `nvidia-smi`

### Package Conflicts
- Use a clean conda environment
- Install packages in the exact order specified above
- Avoid mixing conda and pip for the same package

## Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 3080/4070 or better recommended)
- **RAM**: 32GB system RAM recommended for large scenes
- **Storage**: SSD recommended for fast data loading

## Citation

If you use this work, please cite the original ZipNeRF paper and acknowledge the vector potential optimization extensions.

## License

This project maintains the same license as the original ZipNeRF implementation. 