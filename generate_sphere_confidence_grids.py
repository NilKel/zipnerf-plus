#!/usr/bin/env python3
"""
Generate Sphere-Based Confidence Grids for Vector Potential Training

This script generates confidence grids initialized with the sphere equation,
converting the SDF to logits for use with ConfidenceField.

Usage:
    python generate_sphere_confidence_grids.py --output_dir sphere_confidence_grids
"""

import numpy as np
import torch
import argparse
from pathlib import Path
import json


def sphere_sdf(coords, sphere_center, sphere_radius):
    """
    Compute signed distance function for a sphere.
    
    Args:
        coords: (D, H, W, 3) tensor of coordinates
        sphere_center: (3,) center of sphere
        sphere_radius: radius of sphere
    
    Returns:
        sdf: (D, H, W) signed distance values (negative inside, positive outside)
    """
    # Distance from each point to sphere center
    distances = torch.norm(coords - sphere_center, dim=-1)
    
    # SDF: negative inside sphere, positive outside
    sdf = distances - sphere_radius
    
    return sdf


def sdf_to_logits(sdf, steepness=10.0, offset=0.0):
    """
    Convert SDF to logits for sigmoid activation.
    
    Args:
        sdf: (D, H, W) signed distance values
        steepness: Controls how sharp the transition is
        offset: Bias term (positive = more inside)
    
    Returns:
        logits: (D, H, W) logits where sigmoid(logits) ≈ occupancy
    """
    # Convert SDF to logits: negative SDF (inside) -> positive logits (high confidence)
    logits = -steepness * sdf + offset
    
    return logits


def generate_sphere_confidence_grid(resolution, sphere_radius=1.0, sphere_center=None, 
                                  bbox_size=2.0, steepness=10.0, offset=0.0):
    """
    Generate confidence grid initialized with sphere equation.
    
    Args:
        resolution: Grid resolution (int or tuple)
        sphere_radius: Radius of sphere
        sphere_center: Center of sphere (default: origin)
        bbox_size: Half-size of bounding box
        steepness: Controls sharpness of sigmoid transition
        offset: Bias term for logits
    
    Returns:
        logits: (D, H, W) tensor of logits
        metadata: dict with generation parameters
    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution, resolution)
    
    if sphere_center is None:
        sphere_center = torch.tensor([0.0, 0.0, 0.0])
    else:
        sphere_center = torch.tensor(sphere_center, dtype=torch.float32)
    
    D, H, W = resolution
    
    print(f"🔲 Generating sphere confidence grid {D}x{H}x{W}")
    print(f"   Sphere radius: {sphere_radius}")
    print(f"   Sphere center: {sphere_center}")
    print(f"   Bbox size: ±{bbox_size}")
    print(f"   Steepness: {steepness}")
    print(f"   Offset: {offset}")
    
    # Create coordinate grids
    x = torch.linspace(-bbox_size, bbox_size, W)
    y = torch.linspace(-bbox_size, bbox_size, H)
    z = torch.linspace(-bbox_size, bbox_size, D)
    
    # Create meshgrid matching grid layout (Z, Y, X indexing)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([X, Y, Z], dim=-1)  # (D, H, W, 3)
    
    # Compute SDF
    sdf = sphere_sdf(coords, sphere_center, sphere_radius)
    
    # Convert to logits
    logits = sdf_to_logits(sdf, steepness, offset)
    
    # Convert to confidence for statistics
    confidence = torch.sigmoid(logits)
    
    print(f"   SDF range: [{sdf.min().item():.3f}, {sdf.max().item():.3f}]")
    print(f"   Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"   Confidence range: [{confidence.min().item():.6f}, {confidence.max().item():.6f}]")
    print(f"   High confidence voxels (>0.5): {(confidence > 0.5).sum().item()}/{confidence.numel()}")
    print(f"   Mean confidence: {confidence.mean().item():.6f}")
    
    metadata = {
        'sphere_radius': float(sphere_radius),
        'sphere_center': sphere_center.tolist(),
        'bbox_size': float(bbox_size),
        'resolution': list(resolution),
        'steepness': float(steepness),
        'offset': float(offset),
        'coordinate_system': 'right_handed_z_up',
        'grid_range': [-bbox_size, bbox_size],
        'sdf_range': [float(sdf.min()), float(sdf.max())],
        'logits_range': [float(logits.min()), float(logits.max())],
        'confidence_range': [float(confidence.min()), float(confidence.max())],
        'high_confidence_voxels': int((confidence > 0.5).sum().item()),
        'total_voxels': int(confidence.numel()),
        'mean_confidence': float(confidence.mean().item())
    }
    
    return logits, metadata


def save_confidence_grid(logits, metadata, resolution, output_dir):
    """Save confidence grid and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save logits (format expected by ConfidenceField)
    grid_path = output_dir / f"sphere_confidence_grid_{resolution[0]}.pt"
    torch.save(logits, grid_path)
    
    # Save metadata
    metadata_path = output_dir / f"sphere_confidence_metadata_{resolution[0]}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"   💾 Saved: {grid_path}")
    print(f"   📄 Metadata: {metadata_path}")
    
    return grid_path, metadata_path


def visualize_confidence_grid(logits, output_dir, resolution):
    """Create visualization of the confidence grid."""
    try:
        import matplotlib.pyplot as plt
        
        confidence = torch.sigmoid(logits)
        D, H, W = confidence.shape
        
        # Create slices through the center
        center_z, center_y, center_x = D//2, H//2, W//2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY slice (through center Z)
        axes[0].imshow(confidence[center_z, :, :].numpy(), cmap='viridis')
        axes[0].set_title(f'XY slice (z={center_z})')
        axes[0].axis('off')
        
        # XZ slice (through center Y)  
        axes[1].imshow(confidence[:, center_y, :].numpy(), cmap='viridis')
        axes[1].set_title(f'XZ slice (y={center_y})')
        axes[1].axis('off')
        
        # YZ slice (through center X)
        axes[2].imshow(confidence[:, :, center_x].numpy(), cmap='viridis')
        axes[2].set_title(f'YZ slice (x={center_x})')
        axes[2].axis('off')
        
        plt.tight_layout()
        viz_path = Path(output_dir) / f"sphere_confidence_viz_{resolution[0]}.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   🖼️  Visualization: {viz_path}")
        
    except ImportError:
        print("   ⚠️  Matplotlib not available, skipping visualization")


def main():
    parser = argparse.ArgumentParser(description="Generate sphere-based confidence grids")
    parser.add_argument("--output_dir", default="sphere_confidence_grids",
                       help="Output directory for grids")
    parser.add_argument("--sphere_radius", type=float, default=1.0,
                       help="Radius of the sphere")
    parser.add_argument("--sphere_center", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                       help="Center of sphere (x, y, z)")
    parser.add_argument("--bbox_size", type=float, default=2.0,
                       help="Half-size of bounding box")
    parser.add_argument("--steepness", type=float, default=10.0,
                       help="Steepness of sigmoid transition")
    parser.add_argument("--offset", type=float, default=0.0,
                       help="Bias term for logits")
    parser.add_argument("--resolutions", type=int, nargs='+', default=[64, 128, 256],
                       help="Grid resolutions to generate")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations")
    
    args = parser.parse_args()
    
    print(f"🌟 Generating sphere-based confidence grids")
    print(f"   Sphere radius: {args.sphere_radius}")
    print(f"   Sphere center: {args.sphere_center}")
    print(f"   Bbox size: ±{args.bbox_size}")
    print(f"   Steepness: {args.steepness}")
    print(f"   Offset: {args.offset}")
    print(f"   Resolutions: {args.resolutions}")
    
    sphere_center = args.sphere_center
    
    for resolution in args.resolutions:
        print(f"\n📐 Generating {resolution}³ confidence grid...")
        
        # Generate grid
        logits, metadata = generate_sphere_confidence_grid(
            resolution=resolution,
            sphere_radius=args.sphere_radius,
            sphere_center=sphere_center,
            bbox_size=args.bbox_size,
            steepness=args.steepness,
            offset=args.offset
        )
        
        # Save grid
        grid_path, metadata_path = save_confidence_grid(
            logits, metadata, (resolution, resolution, resolution), args.output_dir
        )
        
        # Optional visualization
        if args.visualize:
            visualize_confidence_grid(
                logits, args.output_dir, (resolution, resolution, resolution)
            )
    
    print(f"\n🎉 Confidence grid generation complete!")
    print(f"📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main() 