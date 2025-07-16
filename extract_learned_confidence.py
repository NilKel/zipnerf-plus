#!/usr/bin/env python3
"""
Extract Learned Confidence Grid from Working Potential Model

This script loads a trained potential model and extracts its learned confidence grid,
then compares it with our generated confidence grid to identify discrepancies.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

import accelerate
import gin
from internal import configs
from internal import models
from internal import checkpoints
from internal import utils


def load_potential_model(checkpoint_path):
    """Load a trained potential model and extract its confidence field."""
    print(f"🔧 Loading potential model from: {checkpoint_path}")
    
    # Extract experiment path from checkpoint path
    checkpoint_path = Path(checkpoint_path)
    exp_path = checkpoint_path.parent.parent
    checkpoint_dir = checkpoint_path.parent
    
    print(f"Experiment path: {exp_path}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Look for config.gin
    config_gin_path = exp_path / "config.gin"
    if not config_gin_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_gin_path}")
    
    # Clear and parse config
    gin.clear_config()
    gin.parse_config_file(str(config_gin_path))
    config = configs.Config()
    
    print(f"Model config:")
    print(f"  use_potential: {config.use_potential}")
    print(f"  confidence_grid_resolution: {config.confidence_grid_resolution}")
    print(f"  data_dir: {config.data_dir}")
    
    # Setup accelerator and load model
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    
    model = models.Model(config=config)
    model.eval()
    model = accelerator.prepare(model)
    
    # Load checkpoint
    step = checkpoints.restore_checkpoint(checkpoint_dir, accelerator)
    print(f"✅ Loaded model from step {step}")
    
    # Extract confidence field
    unwrapped_model = accelerator.unwrap_model(model)
    if hasattr(unwrapped_model, 'confidence_field'):
        confidence_field = unwrapped_model.confidence_field
        print(f"Confidence field resolution: {confidence_field.resolution}")
        
        # Get the learned confidence grid (logits)
        learned_logits = confidence_field.c_grid.data.clone().cpu()
        print(f"Learned logits shape: {learned_logits.shape}")
        print(f"Learned logits range: [{learned_logits.min():.3f}, {learned_logits.max():.3f}]")
        
        # Convert to confidence probabilities
        learned_conf = torch.sigmoid(learned_logits)
        print(f"Learned confidence range: [{learned_conf.min():.6f}, {learned_conf.max():.6f}]")
        print(f"Mean confidence: {learned_conf.mean():.6f}")
        print(f"High confidence voxels (>0.5): {(learned_conf > 0.5).sum().item()}/{learned_conf.numel()}")
        
        return learned_logits, learned_conf, config, step
    else:
        raise ValueError("Model does not have a confidence field!")


def compare_confidence_grids(learned_logits, generated_grid_path):
    """Compare learned confidence grid with our generated grid."""
    print(f"\n📊 Comparing confidence grids...")
    
    # Load generated grid
    if not Path(generated_grid_path).exists():
        print(f"❌ Generated grid not found: {generated_grid_path}")
        return
    
    generated_logits = torch.load(generated_grid_path, map_location='cpu')
    print(f"Generated logits shape: {generated_logits.shape}")
    print(f"Generated logits range: [{generated_logits.min():.3f}, {generated_logits.max():.3f}]")
    
    # Convert both to confidence
    learned_conf = torch.sigmoid(learned_logits)
    generated_conf = torch.sigmoid(generated_logits)
    
    # Resize if needed for comparison
    if learned_logits.shape != generated_logits.shape:
        print(f"⚠️  Shape mismatch: learned {learned_logits.shape} vs generated {generated_logits.shape}")
        
        # Interpolate to match sizes for comparison
        if learned_logits.numel() < generated_logits.numel():
            # Upsample learned to match generated
            learned_conf_resized = F.interpolate(
                learned_conf.unsqueeze(0).unsqueeze(0), 
                size=generated_conf.shape, 
                mode='trilinear', 
                align_corners=True
            ).squeeze(0).squeeze(0)
            compare_conf_1, compare_conf_2 = learned_conf_resized, generated_conf
            label_1, label_2 = "learned (upsampled)", "generated"
        else:
            # Downsample generated to match learned
            generated_conf_resized = F.interpolate(
                generated_conf.unsqueeze(0).unsqueeze(0), 
                size=learned_conf.shape, 
                mode='trilinear', 
                align_corners=True
            ).squeeze(0).squeeze(0)
            compare_conf_1, compare_conf_2 = learned_conf, generated_conf_resized
            label_1, label_2 = "learned", "generated (downsampled)"
    else:
        compare_conf_1, compare_conf_2 = learned_conf, generated_conf
        label_1, label_2 = "learned", "generated"
    
    # Compute comparison metrics
    print(f"\n📈 Comparison metrics:")
    print(f"  {label_1} mean confidence: {compare_conf_1.mean():.6f}")
    print(f"  {label_2} mean confidence: {compare_conf_2.mean():.6f}")
    
    print(f"  {label_1} high conf voxels: {(compare_conf_1 > 0.5).sum().item()}")
    print(f"  {label_2} high conf voxels: {(compare_conf_2 > 0.5).sum().item()}")
    
    # Correlation
    corr = torch.corrcoef(torch.stack([compare_conf_1.flatten(), compare_conf_2.flatten()]))[0, 1]
    print(f"  Correlation: {corr:.6f}")
    
    # MSE
    mse = F.mse_loss(compare_conf_1, compare_conf_2)
    print(f"  MSE: {mse:.6f}")
    
    # Show some sample values
    center_idx = tuple(s // 2 for s in compare_conf_1.shape)
    print(f"\n📍 Sample values at center {center_idx}:")
    print(f"  {label_1}: {compare_conf_1[center_idx].item():.6f}")
    print(f"  {label_2}: {compare_conf_2[center_idx].item():.6f}")
    
    return compare_conf_1, compare_conf_2, corr, mse


def create_comparison_visualization(learned_conf, generated_conf, output_path):
    """Create visualization comparing the two confidence grids."""
    print(f"\n🎨 Creating comparison visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Get center slices
    center = learned_conf.shape[0] // 2
    
    # Learned confidence slices
    axes[0, 0].imshow(learned_conf[center, :, :].numpy(), cmap='viridis', origin='lower')
    axes[0, 0].set_title('Learned Confidence - XY slice')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    axes[0, 1].imshow(learned_conf[:, center, :].numpy(), cmap='viridis', origin='lower')
    axes[0, 1].set_title('Learned Confidence - XZ slice')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    
    axes[0, 2].imshow(learned_conf[:, :, center].numpy(), cmap='viridis', origin='lower')
    axes[0, 2].set_title('Learned Confidence - YZ slice')
    axes[0, 2].set_xlabel('Y')
    axes[0, 2].set_ylabel('Z')
    
    # Generated confidence slices
    axes[1, 0].imshow(generated_conf[center, :, :].numpy(), cmap='viridis', origin='lower')
    axes[1, 0].set_title('Generated Confidence - XY slice')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    
    axes[1, 1].imshow(generated_conf[:, center, :].numpy(), cmap='viridis', origin='lower')
    axes[1, 1].set_title('Generated Confidence - XZ slice')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Z')
    
    axes[1, 2].imshow(generated_conf[:, :, center].numpy(), cmap='viridis', origin='lower')
    axes[1, 2].set_title('Generated Confidence - YZ slice')
    axes[1, 2].set_xlabel('Y')
    axes[1, 2].set_ylabel('Z')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved comparison visualization to: {output_path}")
    plt.close()


def analyze_density_sampling_correctness():
    """Analyze if our density sampling approach was correct."""
    print(f"\n🔍 Analyzing density sampling correctness...")
    
    # Check our density sampling script approach
    print(f"Our approach in sample_density_to_confidence.py:")
    print(f"1. Load baseline model with use_potential=False, use_triplane=False")
    print(f"2. Sample density at regular 3D grid points in world space")
    print(f"3. Convert density to confidence via: p = density / (scale * max_density)")
    print(f"4. Convert to logits: logit = log(p / (1-p))")
    
    print(f"\nPotential issues to check:")
    print(f"1. ❓ Coordinate space mismatch - baseline vs potential model")
    print(f"2. ❓ Scene contraction differences")
    print(f"3. ❓ Density normalization scale (currently 0.5)")
    print(f"4. ❓ Grid resolution/bounds mismatch")
    print(f"5. ❓ Different model architectures (baseline vs potential)")


def main():
    parser = argparse.ArgumentParser(description="Extract and compare learned confidence grid")
    parser.add_argument("--checkpoint_path", type=str, 
                       default="/home/nilkel/Projects/zipnerf-pytorch/exp/lego_potential_25000_0710_1932_defaultconf_nogate/checkpoints/025000",
                       help="Path to working potential model checkpoint")
    parser.add_argument("--generated_grid", type=str,
                       default="confidence_grids_lego/confidence_grid_128.pt",
                       help="Path to our generated confidence grid")
    parser.add_argument("--output_dir", type=str, default="./confidence_comparison",
                       help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    print("🔬 Analyzing Confidence Grid Issues")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Extract learned confidence grid
        learned_logits, learned_conf, config, step = load_potential_model(args.checkpoint_path)
        
        # Save learned grid for future use
        learned_grid_path = output_dir / "learned_confidence_grid.pt"
        torch.save(learned_logits, learned_grid_path)
        print(f"💾 Saved learned confidence grid to: {learned_grid_path}")
        
        # Save metadata
        metadata = {
            'resolution': learned_logits.shape,
            'checkpoint_step': step,
            'checkpoint_path': str(args.checkpoint_path),
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
                'std': learned_conf.std().item()
            }
        }
        
        metadata_path = output_dir / "learned_confidence_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Saved metadata to: {metadata_path}")
        
        # Compare with generated grid
        if Path(args.generated_grid).exists():
            learned_compare, generated_compare, corr, mse = compare_confidence_grids(learned_logits, args.generated_grid)
            
            # Create visualization
            viz_path = output_dir / "confidence_comparison.png"
            create_comparison_visualization(learned_compare, generated_compare, viz_path)
            
            # Summary
            print(f"\n🏁 Summary:")
            print(f"  Correlation: {corr:.6f} ({'GOOD' if corr > 0.7 else 'POOR' if corr < 0.3 else 'MODERATE'})")
            print(f"  MSE: {mse:.6f}")
            
            if corr < 0.5:
                print(f"\n❌ Poor correlation suggests issues with our density sampling approach!")
                print(f"   Possible causes:")
                print(f"   1. Wrong coordinate system or bounds")
                print(f"   2. Incorrect density normalization")
                print(f"   3. Model architecture differences")
                print(f"   4. Scene contraction discrepancies")
            else:
                print(f"\n✅ Reasonable correlation - density sampling approach seems correct")
        else:
            print(f"❌ Generated grid not found: {args.generated_grid}")
        
        # Analyze sampling correctness
        analyze_density_sampling_correctness()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 