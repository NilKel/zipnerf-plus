#!/usr/bin/env python3
"""
Setup Learned Confidence Grid as Debug Grid

This script sets up the ultimate sanity check: use the actual learned confidence grid
from the working potential model as the debug confidence grid. 

If the potential field formulation V_feat = -C(x) * (G(x) · ∇X(x)) is correct,
then using this "perfect" confidence should yield excellent results.
"""

import torch
import json
from pathlib import Path
import shutil
import argparse


def setup_learned_debug_grid(learned_grid_path, output_dir, resolution=128):
    """Setup the learned confidence grid as a debug grid."""
    
    print(f"🔧 Setting up learned confidence grid as debug grid")
    print(f"   Source: {learned_grid_path}")
    print(f"   Output: {output_dir}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load learned confidence grid
    if not Path(learned_grid_path).exists():
        raise FileNotFoundError(f"Learned grid not found: {learned_grid_path}")
    
    learned_logits = torch.load(learned_grid_path, map_location='cpu')
    learned_conf = torch.sigmoid(learned_logits)
    
    print(f"✅ Loaded learned confidence grid:")
    print(f"   Shape: {learned_logits.shape}")
    print(f"   Logits range: [{learned_logits.min():.3f}, {learned_logits.max():.3f}]")
    print(f"   Confidence range: [{learned_conf.min():.6f}, {learned_conf.max():.6f}]")
    print(f"   Mean confidence: {learned_conf.mean():.6f}")
    print(f"   High conf voxels (>0.5): {(learned_conf > 0.5).sum().item()}/{learned_conf.numel()}")
    
    # Copy the learned grid as debug grid
    debug_grid_path = output_dir / f"debug_confidence_grid_{resolution}.pt"
    shutil.copy2(learned_grid_path, debug_grid_path)
    print(f"💾 Copied learned grid to: {debug_grid_path}")
    
    # Create metadata
    metadata = {
        'source': 'learned_confidence_grid',
        'source_path': str(learned_grid_path),
        'resolution': list(learned_logits.shape),
        'description': 'Learned confidence grid from working potential model - ultimate sanity check',
        'use_case': 'Debug confidence grid for potential field sanity check experiment',
        'logits_stats': {
            'min': learned_logits.min().item(),
            'max': learned_logits.max().item(),
            'mean': learned_logits.mean().item(),
            'std': learned_logits.std().item()
        },
        'confidence_stats': {
            'min': learned_conf.min().item(),
            'max': learned_conf.max().item(),
            'mean': learned_conf.mean().item(),
            'std': learned_conf.std().item(),
            'high_conf_ratio': (learned_conf > 0.5).sum().item() / learned_conf.numel()
        },
        'experiment_rationale': (
            "If the potential field formulation V_feat = -C(x) * (G(x) · ∇X(x)) is correct, "
            "then using this learned confidence grid should yield excellent results since it "
            "represents the 'perfect' geometry learned by a working potential model."
        )
    }
    
    metadata_path = output_dir / f"debug_confidence_grid_{resolution}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata to: {metadata_path}")
    
    return debug_grid_path, metadata


def create_experiment_config(debug_grid_path, output_dir):
    """Create a config snippet for running the sanity check experiment."""
    
    config_text = f"""
# Sanity Check Experiment Configuration
# Use this config to run ZipNeRF with the learned confidence grid as debug grid

# Enable potential field with debug confidence grid
Model.use_potential = True
Model.confidence_grid_resolution = (128, 128, 128)

# CRITICAL: Enable debug confidence grid
Config.debug_confidence_grid_path = "{debug_grid_path.absolute()}"
Config.freeze_debug_confidence = True

# Disable other experimental features for clean comparison
Model.use_triplane = False
Config.confidence_reg_mult = 0.0

# Training parameters (can be adjusted)
Config.max_steps = 25000
Config.lr_init = 5e-4
Config.lr_final = 5e-6

# Output settings
Config.exp_path = "exp/lego_sanity_check_learned_grid"
"""
    
    config_path = output_dir / "sanity_check_config.gin"
    with open(config_path, 'w') as f:
        f.write(config_text.strip())
    
    print(f"📝 Created experiment config: {config_path}")
    
    return config_path


def create_run_script(config_path, output_dir):
    """Create a script to run the sanity check experiment."""
    
    script_text = f"""#!/bin/bash

# Sanity Check Experiment: ZipNeRF with Learned Confidence Grid
# This is the ultimate test of the potential field formulation

echo "🧪 Starting Sanity Check Experiment"
echo "Using learned confidence grid as debug grid"
echo "If potential field formulation is correct, this should yield excellent results!"

# Activate environment
conda activate zipnerf2

# Run training with debug confidence grid
python train.py \\
    --gin_configs={config_path.name} \\
    --gin_bindings="Config.data_dir = '/home/nilkel/Projects/data/nerf_synthetic/lego'" \\
    --gin_bindings="Config.exp_path = 'exp/lego_sanity_check_learned_grid_$(date +%m%d_%H%M)'"

echo "✅ Sanity check experiment completed!"
echo "Compare results with baseline to validate potential field formulation"
"""
    
    script_path = output_dir / "run_sanity_check.sh"
    with open(script_path, 'w') as f:
        f.write(script_text.strip())
    
    # Make executable
    script_path.chmod(0o755)
    
    print(f"📜 Created run script: {script_path}")
    print(f"   Run with: bash {script_path}")
    
    return script_path


def main():
    parser = argparse.ArgumentParser(description="Setup learned confidence grid as debug grid for sanity check")
    parser.add_argument("--learned_grid_path", type=str, 
                       default="confidence_comparison/learned_confidence_grid.pt",
                       help="Path to learned confidence grid")
    parser.add_argument("--output_dir", type=str, default="./debug_grids",
                       help="Output directory for debug grids and configs")
    parser.add_argument("--resolution", type=int, default=128,
                       help="Grid resolution")
    
    args = parser.parse_args()
    
    print("🎯 Setting Up Ultimate Sanity Check")
    print("=" * 60)
    print("This experiment uses the learned confidence grid from the working")
    print("potential model as the debug confidence grid. If the potential field")
    print("formulation is correct, this should give EXCELLENT results!")
    print("=" * 60)
    
    try:
        # Setup debug grid
        debug_grid_path, metadata = setup_learned_debug_grid(
            args.learned_grid_path, args.output_dir, args.resolution
        )
        
        # Create experiment config
        config_path = create_experiment_config(debug_grid_path, Path(args.output_dir))
        
        # Create run script
        script_path = create_run_script(config_path, Path(args.output_dir))
        
        print(f"\n🎯 **Ready for Ultimate Sanity Check!**")
        print(f"\n📋 **How to run the experiment:**")
        print(f"   1. cd {Path(args.output_dir).absolute()}")
        print(f"   2. bash run_sanity_check.sh")
        print(f"\n📊 **Expected outcome:**")
        print(f"   - If potential formulation is CORRECT: Excellent results")
        print(f"   - If potential formulation has issues: Poor results even with perfect geometry")
        print(f"\n🔍 **What this tests:**")
        print(f"   - Pure effectiveness of V_feat = -C(x) * (G(x) · ∇X(x))")
        print(f"   - Whether the MLP can use potential features effectively")
        print(f"   - Eliminates geometry learning as a confounding factor")
        
        print(f"\n✅ Setup completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 