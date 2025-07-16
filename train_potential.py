#!/usr/bin/env python3
"""
Simple ZipNeRF Training Script

Usage:
    python train_potential.py <experiment_comment> [--scene <scene>] [--model_type <type>] [--load_grid]

Examples:
    python train_potential.py my_experiment --scene lego --model_type potential
    python train_potential.py my_experiment --scene lego --model_type triplane --load_grid
    python train_potential.py baseline_test --scene lego --model_type baseline
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

def get_model_config(model_type):
    """Return the appropriate config file for the model type"""
    configs = {
        'potential': 'configs/unified_potential.gin',
        'triplane': 'configs/potential_triplane.gin', 
        'baseline': 'configs/blender.gin'
    }
    
    if model_type not in configs:
        print(f"❌ Error: Unknown model type '{model_type}'. Available: {list(configs.keys())}")
        sys.exit(1)
        
    config_file = configs[model_type]
    if not os.path.exists(config_file):
        print(f"❌ Error: Config file not found: {config_file}")
        sys.exit(1)
        
    return config_file

def main():
    parser = argparse.ArgumentParser(description="Train ZipNeRF")
    parser.add_argument("comment", help="Experiment comment/name to append")
    parser.add_argument("--scene", default="lego", help="Scene name (default: lego)")
    parser.add_argument("--model_type", default="potential", 
                       choices=['potential', 'triplane', 'baseline'],
                       help="Model type (default: potential)")
    parser.add_argument("--load_grid", action="store_true", 
                       help="Load existing confidence grid if available (potential/triplane only)")
    parser.add_argument("--max_steps", type=int, default=25000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = f"/home/nilkel/Projects/data/nerf_synthetic/{args.scene}"
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Get config file for model type
    config_file = get_model_config(args.model_type)
    
    # Check if we should load existing confidence grid (only for potential/triplane)
    debug_grid_path = ""
    if args.load_grid and args.model_type in ['potential', 'triplane']:
        grid_path = "debug_grids/debug_confidence_grid_256.pt"
        if os.path.exists(grid_path):
            debug_grid_path = grid_path
            print(f"✅ Will load confidence grid: {grid_path}")
        else:
            print(f"⚠️  Confidence grid not found: {grid_path} (training from scratch)")
    elif args.load_grid and args.model_type == 'baseline':
        print("⚠️  --load_grid ignored for baseline model (no confidence grid)")
    
    # Create unique experiment name with timestamp
    timestamp = datetime.now().strftime("%m%d_%H%M")
    full_exp_name = f"{args.scene}_{args.model_type}_{timestamp}_{args.comment}"
    
    # Print configuration
    print("\n" + "="*60)
    print("🚀 ZIPNERF TRAINING")
    print("="*60)
    print(f"🎬 Scene: {args.scene}")
    print(f"🤖 Model: {args.model_type}")
    print(f"📁 Data directory: {data_dir}")
    print(f"🏷️  Experiment name: {full_exp_name}")
    print(f"⚙️  Config file: {config_file}")
    print(f"🎯 Max steps: {args.max_steps}")
    print(f"📦 Batch size: {args.batch_size}")
    if args.model_type in ['potential', 'triplane']:
        print(f"🔧 Confidence grid: {'Load existing' if debug_grid_path else 'Train from scratch'}")
    print("="*60)
    
    # Build gin bindings
    gin_bindings = [
        f"Config.data_dir = '{data_dir}'",
        f"Config.exp_name = '{full_exp_name}'",
        f"Config.max_steps = {args.max_steps}",
        f"Config.batch_size = {args.batch_size}",
    ]
    
    # Add confidence grid binding only for potential/triplane models
    if args.model_type in ['potential', 'triplane']:
        gin_bindings.append(f"Config.debug_confidence_grid_path = '{debug_grid_path}'")
    
    # Build command - use conda run to ensure proper environment
    cmd = [
        "conda", "run", "-n", "zipnerf2", "--no-capture-output",
        "accelerate", "launch", "train.py",
        f"--gin_configs={config_file}"
    ]
    
    # Add gin bindings
    for binding in gin_bindings:
        cmd.append(f"--gin_bindings={binding}")
    
    print("\n💻 Training command:")
    print(" ".join(cmd))
    print()
    
    # Ask for confirmation
    response = input("Start training? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    # Set environment variables for better error handling
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print("🏃 Starting training...")
    try:
        # Run training
        result = subprocess.run(cmd, env=env, check=True)
        
        print("\n✅ Training completed successfully!")
        
        # Extract confidence grid if training completed and using potential/triplane
        if args.model_type in ['potential', 'triplane']:
            extract_confidence_grid(full_exp_name)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        print("Check the logs for details.")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def extract_confidence_grid(exp_name):
    """Extract confidence grid from trained model"""
    try:
        print("\n🔍 Extracting confidence grid...")
        
        # Find the experiment directory
        exp_dir = None
        for item in os.listdir("exp"):
            if item.startswith(exp_name):
                exp_dir = f"exp/{item}"
                break
        
        if not exp_dir or not os.path.exists(exp_dir):
            print(f"❌ Could not find experiment directory for {exp_name}")
            return
        
        # Find the latest checkpoint
        checkpoints_dir = f"{exp_dir}/checkpoints"
        if not os.path.exists(checkpoints_dir):
            print(f"❌ No checkpoints directory found in {exp_dir}")
            return
        
        checkpoints = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(f"{checkpoints_dir}/{d}")]
        if not checkpoints:
            print(f"❌ No checkpoints found in {checkpoints_dir}")
            return
        
        latest_checkpoint = sorted(checkpoints)[-1]
        checkpoint_path = f"{checkpoints_dir}/{latest_checkpoint}"
        
        print(f"📦 Using checkpoint: {checkpoint_path}")
        
        # Python script to extract confidence grid
        extract_script = f"""
import torch
import os
import sys
sys.path.append('.')

# Try different checkpoint file formats
checkpoint_files = [
    '{checkpoint_path}/model.safetensors',
    '{checkpoint_path}/pytorch_model.bin',
    '{checkpoint_path}/pytorch_model.pt'
]

checkpoint = None
for checkpoint_file in checkpoint_files:
    if os.path.exists(checkpoint_file):
        try:
            if checkpoint_file.endswith('.safetensors'):
                from safetensors import safe_open
                checkpoint = {{}}
                with safe_open(checkpoint_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        checkpoint[key] = f.get_tensor(key)
            else:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
            print(f"✅ Loaded checkpoint from: {{checkpoint_file}}")
            break
        except Exception as e:
            print(f"❌ Failed to load {{checkpoint_file}}: {{e}}")
            continue

if checkpoint is None:
    print("❌ No valid checkpoint file found")
    print(f"Checked: {{checkpoint_files}}")
    sys.exit(1)

# Look for confidence grid in different possible keys
confidence_keys = [k for k in checkpoint.keys() if 'confidence' in k.lower()]
print(f"Available confidence keys: {{confidence_keys}}")

confidence_grid = None
for key in ['confidence_field.c_grid', 'confidence_field.confidence', 'model.confidence_field.confidence']:
    if key in checkpoint:
        confidence_grid = checkpoint[key]
        print(f"✅ Found confidence grid at key: {{key}}")
        break

if confidence_grid is not None:
    print(f"✅ Extracted confidence grid with shape: {{confidence_grid.shape}}")
    print(f"✅ Value range: {{confidence_grid.min():.2f}} to {{confidence_grid.max():.2f}}")
    
    # Create debug_grids directory
    os.makedirs('debug_grids', exist_ok=True)
    
    # Save to debug_grids
    torch.save(confidence_grid, 'debug_grids/debug_confidence_grid_256.pt')
    torch.save(confidence_grid, 'debug_grids/confidence_grid_256_learned.pt')
    
    print("✅ Saved confidence grids to debug_grids/")
else:
    print("❌ No confidence grid found in checkpoint")
    if confidence_keys:
        print(f"Available confidence keys: {{confidence_keys}}")
    else:
        print("No keys containing 'confidence' found")
"""
        
        # Run extraction using conda
        extract_cmd = [
            "conda", "run", "-n", "zipnerf2", "--no-capture-output",
            "python", "-c", extract_script
        ]
        
        result = subprocess.run(extract_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ Failed to extract confidence grid:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error extracting confidence grid: {e}")

if __name__ == "__main__":
    main() 