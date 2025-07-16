#!/usr/bin/env python3
"""
Sanity Check Experiment: ZipNeRF with Pretrained Confidence Grid

This script demonstrates how to run the sanity check experiment using a pretrained
confidence grid derived from a well-trained baseline model. This tests whether
the potential field formulation V_feat = -C(x) * (G(x) · ∇X(x)) is effective
when provided with "perfect" geometry.

The hypothesis: If the volume integral feature formulation is correct, then using
a frozen confidence grid from a well-trained baseline should yield BETTER results
than the baseline itself.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def check_requirements():
    """Check that required files exist."""
    print("🔍 Checking requirements...")
    
    # Check for confidence grids
    confidence_dir = Path("confidence_grids_lego")
    if not confidence_dir.exists():
        print(f"❌ Confidence grids directory not found: {confidence_dir}")
        print("   Please run sample_density_to_confidence.py first!")
        return False
    
    # Check for specific grid files
    required_grids = ["confidence_grid_128.pt", "confidence_grid_256.pt"]
    for grid_file in required_grids:
        grid_path = confidence_dir / grid_file
        if not grid_path.exists():
            print(f"❌ Required confidence grid not found: {grid_path}")
            return False
        else:
            print(f"✅ Found confidence grid: {grid_path}")
    
    # Check baseline model exists
    baseline_checkpoint = Path("exp/lego_baseline_25000_0704_2320/checkpoints/025000")
    if baseline_checkpoint.exists():
        print(f"✅ Found baseline checkpoint: {baseline_checkpoint}")
    else:
        print(f"⚠️  Baseline checkpoint not found: {baseline_checkpoint}")
        print("   The sanity check will still work, but you won't have a direct comparison")
    
    return True


def generate_experiment_name(grid_resolution=128, frozen=True):
    """Generate experiment name for sanity check."""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    freeze_suffix = "frozen" if frozen else "trainable"
    return f"lego_potential_sanity_{grid_resolution}_{freeze_suffix}_{timestamp}"


def run_sanity_check_experiment(grid_resolution=128, freeze_grid=True, max_steps=25000, 
                               batch_size=32768, use_wandb=True, dry_run=False):
    """
    Run the sanity check experiment.
    
    Args:
        grid_resolution: Resolution of confidence grid to use (128 or 256)
        freeze_grid: Whether to freeze the confidence grid
        max_steps: Maximum training steps
        batch_size: Training batch size
        use_wandb: Whether to use wandb logging
        dry_run: If True, print command without executing
    """
    print(f"\n🧪 Running Sanity Check Experiment")
    print(f"{'='*60}")
    
    # Configuration
    data_dir = "/home/nilkel/Projects/data/nerf_synthetic/lego"
    config_file = "configs/blender.gin"
    confidence_grid_path = f"confidence_grids_lego/confidence_grid_{grid_resolution}.pt"
    exp_name = generate_experiment_name(grid_resolution, freeze_grid)
    
    print(f"📊 Experiment Configuration:")
    print(f"   🎬 Scene: lego")
    print(f"   📁 Data: {data_dir}")
    print(f"   🏷️  Experiment: {exp_name}")
    print(f"   🔧 Confidence grid: {confidence_grid_path} ({grid_resolution}³)")
    print(f"   🔒 Grid frozen: {freeze_grid}")
    print(f"   🎯 Max steps: {max_steps}")
    print(f"   📦 Batch size: {batch_size}")
    print(f"   📊 Wandb: {use_wandb}")
    
    # Build training command
    gin_bindings = [
        f"Config.data_dir = '{data_dir}'",
        f"Config.exp_name = '{exp_name}'",
        f"Config.max_steps = {max_steps}",
        f"Config.batch_size = {batch_size}",
        "Config.factor = 4",
        "Config.use_potential = True",
        "Config.use_triplane = False",  # Pure potential field test
        f"Config.debug_confidence_grid_path = '{confidence_grid_path}'",
        f"Config.freeze_debug_confidence = {str(freeze_grid)}",
        "Config.gradient_scaling = True",
        "Config.train_render_every = 1000",
    ]
    
    if use_wandb:
        gin_bindings.extend([
            "Config.use_wandb = True",
            f"Config.wandb_project = 'zipnerf-sanity-check'",
            f"Config.wandb_name = '{exp_name}'",
        ])
    else:
        gin_bindings.append("Config.use_wandb = False")
    
    # Build command
    cmd = [
        "accelerate", "launch", "train.py",
        f"--gin_configs={config_file}"
    ]
    
    for binding in gin_bindings:
        cmd.append(f"--gin_bindings={binding}")
    
    # Print command
    print(f"\n💻 Training command:")
    cmd_str = " \\\n    ".join(cmd)
    print(f"    {cmd_str}")
    
    if dry_run:
        print(f"\n🔍 Dry run mode - command not executed")
        return exp_name
    
    # Ask for confirmation
    print(f"\n🤔 This will start training with the sanity check configuration.")
    print(f"   Expected behavior: Should achieve HIGHER PSNR than baseline (~32+ vs ~31)")
    response = input("Start training? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("Training cancelled.")
        return exp_name
    
    # Run training
    print(f"\n🏃 Starting sanity check training...")
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
    
    return exp_name


def run_comparison_experiments():
    """Run multiple sanity check experiments for comparison."""
    print(f"\n🔬 Running Comprehensive Sanity Check")
    print(f"{'='*80}")
    print(f"This will run multiple experiments to test the potential field formulation:")
    print(f"1. Frozen 128³ confidence grid (sanity check)")
    print(f"2. Trainable 128³ confidence grid (end-to-end learning)")
    print(f"3. Frozen 256³ confidence grid (higher resolution test)")
    
    response = input(f"\nRun all experiments? This will take several hours. [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Comparison experiments cancelled.")
        return
    
    experiments = [
        {"grid_resolution": 128, "freeze_grid": True, "name": "Sanity Check (Frozen 128³)"},
        {"grid_resolution": 128, "freeze_grid": False, "name": "End-to-End (Trainable 128³)"},
        {"grid_resolution": 256, "freeze_grid": True, "name": "High-Res Sanity (Frozen 256³)"},
    ]
    
    completed_experiments = []
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Experiment {i}/3: {exp_config['name']}")
        print(f"{'='*60}")
        
        exp_name = run_sanity_check_experiment(
            grid_resolution=exp_config['grid_resolution'],
            freeze_grid=exp_config['freeze_grid'],
            max_steps=25000,
            batch_size=32768,
            use_wandb=True,
            dry_run=False
        )
        
        completed_experiments.append({
            'name': exp_config['name'],
            'exp_name': exp_name,
            'config': exp_config
        })
        
        print(f"\n⏸️  Experiment {i} completed. Press Enter to continue to next experiment...")
        input()
    
    print(f"\n🎉 All sanity check experiments completed!")
    print(f"\n📊 Experiment Summary:")
    for exp in completed_experiments:
        print(f"   {exp['name']}: {exp['exp_name']}")
    
    print(f"\n📈 Next steps:")
    print(f"1. Monitor training progress in wandb: https://wandb.ai/")
    print(f"2. Compare PSNR results when training completes")
    print(f"3. Expected results:")
    print(f"   - Frozen grids should achieve ~32+ PSNR (better than baseline ~31)")
    print(f"   - If not, debug the volume integral feature computation")
    print(f"   - Trainable grids should achieve similar or slightly better results")


def main():
    parser = argparse.ArgumentParser(description="Run ZipNeRF sanity check experiment with pretrained confidence grid")
    parser.add_argument("--grid_resolution", type=int, choices=[128, 256], default=128,
                       help="Resolution of confidence grid to use")
    parser.add_argument("--freeze_grid", action="store_true", default=True,
                       help="Freeze the confidence grid (recommended for sanity check)")
    parser.add_argument("--trainable_grid", action="store_true",
                       help="Make confidence grid trainable (overrides --freeze_grid)")
    parser.add_argument("--max_steps", type=int, default=25000,
                       help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=32768,
                       help="Training batch size")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print command without executing")
    parser.add_argument("--comparison", action="store_true",
                       help="Run multiple comparison experiments")
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Handle freeze/trainable logic
    freeze_grid = args.freeze_grid and not args.trainable_grid
    
    if args.comparison:
        run_comparison_experiments()
    else:
        run_sanity_check_experiment(
            grid_resolution=args.grid_resolution,
            freeze_grid=freeze_grid,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            use_wandb=not args.no_wandb,
            dry_run=args.dry_run
        )


if __name__ == "__main__":
    main() 