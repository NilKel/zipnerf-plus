#!/bin/bash
# Training Examples for ZipNeRF with Triplane Integration
# Make this file executable: chmod +x training_examples.sh

echo "🚀 ZipNeRF Training Examples"
echo "================================"

# Set your data directory here
DATA_DIR="/SSD_DISK/datasets/360_v2"  # Change this to your data path

echo "📁 Using data directory: $DATA_DIR"
echo ""

echo "💡 Example Commands:"
echo ""

echo "1️⃣  Train with triplane (recommended):"
echo "python run_training.py --exp_name 'lego_triplane' --data_dir '$DATA_DIR' --scene 'lego'"
echo ""

echo "2️⃣  Baseline ZipNeRF (no triplane):"
echo "python run_training.py --exp_name 'lego_baseline' --data_dir '$DATA_DIR' --scene 'lego' --no_triplane"
echo ""

echo "3️⃣  Quick test run (smaller batch):"
echo "python run_training.py --exp_name 'lego_test' --data_dir '$DATA_DIR' --scene 'lego' --batch_size 16384 --max_steps 5000"
echo ""

echo "4️⃣  High-res training (factor=2):"
echo "python run_training.py --exp_name 'lego_highres' --data_dir '$DATA_DIR' --scene 'lego' --factor 2 --batch_size 16384"
echo ""

echo "5️⃣  Multi-GPU training on specific GPU:"
echo "python run_training.py --exp_name 'lego_gpu1' --data_dir '$DATA_DIR' --scene 'lego' --gpu 1"
echo ""

echo "6️⃣  Dry run (see command without executing):"
echo "python run_training.py --exp_name 'lego_test' --data_dir '$DATA_DIR' --scene 'lego' --dry_run"
echo ""

echo "📋 Common Blender scenes: lego, chair, drums, ficus, hotdog, materials, mic, ship"
echo "📋 Common 360_v2 scenes: bicycle, flowers, garden, stump, treehill, room, counter, kitchen, bonsai"
echo ""

echo "🔧 Quick Commands (copy and paste):"
echo "===================================="

# Quick training commands for common scenes
scenes=("lego" "chair" "hotdog" "ship")

for scene in "${scenes[@]}"; do
    echo "# $scene with triplane"
    echo "python run_training.py --exp_name '${scene}_triplane' --data_dir '$DATA_DIR' --scene '$scene'"
    echo ""
done

echo "💡 Tips:"
echo "- Use --no_wandb to disable Weights & Biases logging"
echo "- Use --resume to continue from existing checkpoint"
echo "- Experiment names automatically get timestamps (e.g., lego_triplane_0703_1530)"
echo "- Default config is configs/blender.gin (change with --config)"
echo ""

echo "🚨 Remember to:"
echo "1. Activate your conda environment: conda activate zipnerf2"
echo "2. Set your correct data directory in this script or use --data_dir"
echo "3. Check GPU availability: nvidia-smi"
echo ""

echo "For help: python run_training.py --help" 