# ZipNeRF Setup for RTX 5090

This repository has been successfully configured to work with the RTX 5090 GPU using `uv` and `uv pip install` only.

## ✅ What's Installed

- **Python 3.11.13** via uv
- **PyTorch 2.1.2+cu121** (compatible with RTX 5090)
- **NumPy 1.26.4** (downgraded from 2.x for compatibility)
- **All ZipNeRF dependencies** from requirements.txt
- **Custom CUDA extensions** compiled with C++17
- **nerfstudio** for the ZipNeRF plugin
- **torch_scatter** for enhanced functionality
- **nvdiffrast** for textured mesh support

## 🚀 Quick Start

1. **Activate the environment:**
   ```bash
   source setup_env.sh
   ```

2. **Configure accelerate (first time only):**
   ```bash
   accelerate config
   ```

3. **Download a dataset (example with mipnerf360):**
   ```bash
   mkdir data && cd data
   wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
   unzip 360_v2.zip
   cd ..
   ```

4. **Train ZipNeRF:**
   ```bash
   accelerate launch train.py \
       --gin_configs=configs/360.gin \
       --gin_bindings="Config.data_dir='data/360_v2/bicycle'" \
       --gin_bindings="Config.exp_name='bicycle'" \
       --gin_bindings="Config.factor=4"
   ```

## 🔧 Technical Details

### Compatibility Issues Resolved:
- **RTX 5090 CUDA Capability**: Uses sm_90 architecture compilation for compatibility
- **PyTorch Version**: Downgraded from 2.6 to 2.1.2 for code compatibility
- **NumPy Version**: Downgraded to 1.x to avoid extension compilation issues
- **C++ Standard**: Updated to C++17 to resolve compilation warnings

### Key Environment Variables:
```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export LD_LIBRARY_PATH="/path/to/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/path/to/extensions/cuda:$PYTHONPATH"
```

## 📊 Performance Notes

The RTX 5090 works despite PyTorch warnings about sm_120 capability not being officially supported. The GPU runs using sm_90 compilation targets and performs excellently.

## 🎯 Available Commands

- **Training**: `accelerate launch train.py --gin_configs=configs/360.gin`
- **Rendering**: `accelerate launch render.py --gin_configs=configs/360.gin`
- **Evaluation**: `accelerate launch eval.py --gin_configs=configs/360.gin`
- **Mesh Extraction**: `python extract.py --gin_configs=configs/360.gin`

## 🛠️ Manual Activation

If you don't want to use the setup script:
```bash
source zipnerf-env/bin/activate
export LD_LIBRARY_PATH="/home/nilkel/Projects/zipnerf-pytorch/zipnerf-env/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/home/nilkel/Projects/zipnerf-pytorch/extensions/cuda:$PYTHONPATH"
```

## ✨ Successfully Tested Components

- ✅ PyTorch 2.1.2 with CUDA 12.1
- ✅ Custom CUDA extensions (`_cuda_backend`)
- ✅ GridEncoder functionality
- ✅ Training script accessibility
- ✅ nvdiffrast for mesh rendering
- ✅ torch_scatter for point operations

Happy training! 🎉 