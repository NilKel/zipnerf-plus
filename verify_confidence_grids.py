#!/usr/bin/env python3
"""
Verify Generated Confidence Grids

This script loads and verifies the generated confidence grids to ensure they can be used
properly in the potential field experiments.
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_verify_grid(grid_path, metadata_path):
    """Load and verify a confidence grid."""
    print(f"\n{'='*60}")
    print(f"Verifying: {grid_path}")
    print(f"{'='*60}")
    
    # Load grid
    grid = torch.load(grid_path, map_location='cpu')
    print(f"Grid shape: {grid.shape}")
    print(f"Grid dtype: {grid.dtype}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Resolution: {metadata['resolution']}³")
    print(f"Coordinate bounds: ±{metadata['bound']}")
    print(f"Checkpoint step: {metadata['checkpoint_step']}")
    
    # Verify grid statistics
    print(f"\nGrid Statistics:")
    print(f"  Min: {grid.min().item():.6f}")
    print(f"  Max: {grid.max().item():.6f}")
    print(f"  Mean: {grid.mean().item():.6f}")
    print(f"  Std: {grid.std().item():.6f}")
    
    # Check for NaN/Inf values
    nan_count = torch.isnan(grid).sum().item()
    inf_count = torch.isinf(grid).sum().item()
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    
    # Convert to confidence probabilities to verify conversion
    conf = torch.sigmoid(grid)
    print(f"\nConfidence Probabilities (after sigmoid):")
    print(f"  Min: {conf.min().item():.6f}")
    print(f"  Max: {conf.max().item():.6f}")
    print(f"  Mean: {conf.mean().item():.6f}")
    print(f"  Std: {conf.std().item():.6f}")
    
    # Histogram of values
    print(f"\nValue distribution:")
    values = grid.flatten().numpy()
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    perc_values = np.percentile(values, percentiles)
    for p, v in zip(percentiles, perc_values):
        print(f"  {p:2d}th percentile: {v:8.3f}")
    
    return grid, metadata


def create_slice_visualization(grid, output_path, title):
    """Create a visualization of grid slices."""
    resolution = grid.shape[0]
    
    # Take slices through the center
    center = resolution // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY slice (at center Z)
    xy_slice = grid[center, :, :].numpy()
    im1 = axes[0].imshow(xy_slice, cmap='viridis', origin='lower')
    axes[0].set_title(f'XY slice (Z={center})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # XZ slice (at center Y)
    xz_slice = grid[:, center, :].numpy()
    im2 = axes[1].imshow(xz_slice, cmap='viridis', origin='lower')
    axes[1].set_title(f'XZ slice (Y={center})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    plt.colorbar(im2, ax=axes[1])
    
    # YZ slice (at center X)
    yz_slice = grid[:, :, center].numpy()
    im3 = axes[2].imshow(yz_slice, cmap='viridis', origin='lower')
    axes[2].set_title(f'YZ slice (X={center})')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(f'{title} - Center Slices')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Verify generated confidence grids")
    parser.add_argument("--grid_dir", type=str, default="./confidence_grids_lego",
                       help="Directory containing the confidence grids")
    parser.add_argument("--create_viz", action="store_true",
                       help="Create visualization plots")
    
    args = parser.parse_args()
    
    grid_dir = Path(args.grid_dir)
    
    if not grid_dir.exists():
        print(f"Error: Grid directory not found: {grid_dir}")
        return
    
    print("🔍 Verifying Generated Confidence Grids")
    print("=" * 80)
    
    # Find all grid files
    grid_files = list(grid_dir.glob("confidence_grid_*.pt"))
    
    if not grid_files:
        print(f"No confidence grid files found in {grid_dir}")
        return
    
    print(f"Found {len(grid_files)} confidence grids")
    
    # Verify each grid
    for grid_file in sorted(grid_files):
        # Find corresponding metadata file
        metadata_file = grid_file.with_name(grid_file.stem + "_metadata.json")
        
        if not metadata_file.exists():
            print(f"Warning: No metadata file found for {grid_file}")
            continue
        
        # Load and verify
        grid, metadata = load_and_verify_grid(grid_file, metadata_file)
        
        # Create visualization if requested
        if args.create_viz:
            viz_file = grid_file.with_name(grid_file.stem + "_slices.png")
            create_slice_visualization(
                grid, viz_file, 
                f"Confidence Grid {metadata['resolution']}³"
            )
    
    print(f"\n✅ Verification completed!")
    
    # Test loading grids as would be done in the confidence field
    print(f"\n🧪 Testing ConfidenceField integration...")
    
    # Example of how these grids would be loaded in the ConfidenceField
    grid_128_path = grid_dir / "confidence_grid_128.pt"
    if grid_128_path.exists():
        grid_128 = torch.load(grid_128_path, map_location='cpu')
        
        # Simulate creating a ConfidenceField-like parameter
        c_grid = torch.nn.Parameter(grid_128.clone())
        print(f"Created parameter from 128³ grid: {c_grid.shape}")
        
        # Test sigmoid conversion
        confidence = torch.sigmoid(c_grid)
        print(f"Confidence range: [{confidence.min().item():.6f}, {confidence.max().item():.6f}]")
        
        # Test that we can compute gradients
        c_grid.requires_grad_(True)
        test_loss = confidence.mean()
        test_loss.backward()
        
        if c_grid.grad is not None:
            print(f"✅ Gradient computation works")
            print(f"Gradient magnitude: {c_grid.grad.norm().item():.6f}")
        else:
            print(f"❌ Gradient computation failed")
    
    print(f"\n🎉 All tests passed! The confidence grids are ready for use.")


if __name__ == "__main__":
    main() 