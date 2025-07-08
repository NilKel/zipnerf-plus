#!/usr/bin/env python3
"""
Convenient training script for ZipNeRF with triplane integration
Usage examples:
  python run_training.py --exp_name "lego_triplane" --data_dir "/path/to/lego"
  python run_training.py --exp_name "chair_baseline" --data_dir "/path/to/chair" --triplane false
  python run_training.py --exp_name "hotdog_test" --data_dir "/path/to/hotdog" --batch_size 16384
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def get_default_data_dir():
    """Get default data directory if available"""
    # Common data directory locations
    possible_dirs = [
        "/SSD_DISK/datasets/360_v2",
        "/data/nerf_synthetic", 
        "./data",
        "../data"
    ]
    
    for data_dir in possible_dirs:
        if os.path.exists(data_dir):
            return data_dir
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Train ZipNeRF with triplane integration")
    
    # Required arguments
    parser.add_argument("--exp_name", type=str, required=True,
                       help="Experiment name (will be used for logging and checkpoints)")
    parser.add_argument("--data_dir", type=str, 
                       help="Path to dataset directory")
    parser.add_argument("--scene", type=str,
                       help="Scene name within data_dir (e.g., 'lego', 'chair')")
    
    # Training configuration
    parser.add_argument("--config", type=str, default="configs/blender.gin",
                       help="Gin config file (default: configs/blender.gin)")
    parser.add_argument("--batch_size", type=int, default=32768,
                       help="Training batch size (default: 32768)")
    parser.add_argument("--factor", type=int, default=4,
                       help="Image downsampling factor (default: 4)")
    parser.add_argument("--max_steps", type=int, default=25000,
                       help="Maximum training steps (default: 25000)")
    
    # Triplane options
    parser.add_argument("--triplane", choices=["true", "false"], default="true",
                       help="Enable triplane integration: true/false (default: true)")
    
    # Wandb options
    parser.add_argument("--wandb_project", type=str, default="zipnerf-triplane",
                       help="Wandb project name (default: zipnerf-triplane)")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    
    # Hardware options
    parser.add_argument("--gpu", type=int, default=None,
                       help="GPU ID to use (default: auto)")
    
    # Advanced options
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if available")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print command without executing")
    
    args = parser.parse_args()
    
    # Validate and setup paths
    if args.data_dir is None:
        default_dir = get_default_data_dir()
        if default_dir is None:
            print("Error: --data_dir is required (no default data directory found)")
            sys.exit(1)
        args.data_dir = default_dir
        print(f"Using default data directory: {args.data_dir}")
    
    # Construct full data path
    if args.scene:
        full_data_path = os.path.join(args.data_dir, args.scene)
    else:
        full_data_path = args.data_dir
    
    if not os.path.exists(full_data_path):
        print(f"Error: Data directory does not exist: {full_data_path}")
        sys.exit(1)
    
    # Generate timestamp for unique naming
    timestamp = datetime.now().strftime("%m%d_%H%M")
    exp_name_with_timestamp = f"{args.exp_name}_{timestamp}"
    
    # Setup environment variables
    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Build training command
    cmd = [
        "accelerate", "launch", "train.py",
        f"--gin_configs={args.config}",
        f"--gin_bindings=Config.data_dir = '{full_data_path}'",
        f"--gin_bindings=Config.exp_name = '{exp_name_with_timestamp}'",
        f"--gin_bindings=Config.factor = {args.factor}",
        f"--gin_bindings=Config.batch_size = {args.batch_size}",
        f"--gin_bindings=Config.max_steps = {args.max_steps}",
    ]
    
    # Triplane configuration
    if args.triplane.lower() == "false":
        cmd.append("--gin_bindings=Config.use_triplane = False")
        print("🔧 Triplane integration: DISABLED (baseline ZipNeRF)")
    else:
        cmd.append("--gin_bindings=Config.use_triplane = True")
        print("🔧 Triplane integration: ENABLED")
    
    # Wandb configuration
    if not args.no_wandb:
        cmd.extend([
            "--gin_bindings=Config.use_wandb = True",
            f"--gin_bindings=Config.wandb_project = '{args.wandb_project}'",
            f"--gin_bindings=Config.wandb_name = '{exp_name_with_timestamp}'"
        ])
        print(f"📊 Wandb logging: ENABLED (project: {args.wandb_project})")
    else:
        cmd.append("--gin_bindings=Config.use_wandb = False")
        print("📊 Wandb logging: DISABLED")
    
    # Resume configuration
    if not args.resume:
        cmd.append("--gin_bindings=Config.resume_from_checkpoint = False")
    
    # Print configuration summary
    print("\n" + "="*60)
    print("🚀 ZIPNERF TRAINING CONFIGURATION")
    print("="*60)
    print(f"📁 Data directory: {full_data_path}")
    print(f"🏷️  Experiment name: {exp_name_with_timestamp}")
    print(f"📋 Config file: {args.config}")
    print(f"🎯 Batch size: {args.batch_size}")
    print(f"📏 Downsampling factor: {args.factor}")
    print(f"🔄 Max steps: {args.max_steps}")
    if args.gpu is not None:
        print(f"🖥️  GPU: {args.gpu}")
    print("="*60)
    
    # Print and optionally execute command
    print("\n💻 Training command:")
    print(" ".join(cmd))
    print()
    
    if args.dry_run:
        print("🔍 Dry run mode - command not executed")
        return
    
    # Ask for confirmation
    response = input("Start training? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    print("🏃 Starting training...")
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Experiment: {exp_name_with_timestamp}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main() 