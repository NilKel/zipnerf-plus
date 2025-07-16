#!/usr/bin/env python3
"""
Generate Analytical Grids for Sphere Vector Potential Testing

This script generates perfect analytical occupancy and gradient grids for a sphere,
following the refined approach that bypasses logits and conv3d operations.

Key Benefits:
1. Direct occupancy values (0/1) instead of logits requiring sigmoid
2. Perfect analytical normals instead of finite difference approximations
3. Cleaner testing of V_feat formulation without numerical errors

Usage:
    python generate_analytical_grids.py --sphere_radius 1.0 --output_dir analytical_grids
"""

import numpy as np
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import json


def generate_analytical_occupancy_grid(resolution, sphere_radius, sphere_center=None, bbox_size=2.0):
    """
    Generate analytical binary occupancy grid for a sphere.
    
    Args:
        resolution: Grid resolution (int or tuple of 3 ints)
        sphere_radius: Radius of the sphere
        sphere_center: Center of sphere in world coordinates (default: origin)
        bbox_size: Size of the bounding box (grid spans [-bbox_size, bbox_size])
    
    Returns:
        occupancy_grid: torch.Tensor of shape (D, H, W) with binary values {0, 1}
    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution, resolution)
    
    if sphere_center is None:
        sphere_center = np.array([0.0, 0.0, 0.0])
    
    D, H, W = resolution
    
    print(f"🔲 Generating analytical occupancy grid {D}x{H}x{W}")
    print(f"   Sphere radius: {sphere_radius}")
    print(f"   Sphere center: {sphere_center}")
    print(f"   Bbox size: ±{bbox_size}")
    
    # Create coordinate grids
    # Grid coordinates range from [-bbox_size, bbox_size] in each dimension
    x = torch.linspace(-bbox_size, bbox_size, W)
    y = torch.linspace(-bbox_size, bbox_size, H) 
    z = torch.linspace(-bbox_size, bbox_size, D)
    
    # Create meshgrid - note the indexing matches the grid layout
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')  # Shape: (D, H, W)
    
    # Stack coordinates
    coords = torch.stack([X, Y, Z], dim=-1)  # Shape: (D, H, W, 3)
    
    # Convert sphere center to tensor
    sphere_center_tensor = torch.tensor(sphere_center, dtype=torch.float32)
    
    # Compute signed distance function (SDF)
    # SDF = ||p - center|| - radius
    distances = torch.norm(coords - sphere_center_tensor, dim=-1)  # Shape: (D, H, W)
    sdf = distances - sphere_radius
    
    # Binary occupancy: 1 inside sphere (sdf <= 0), 0 outside (sdf > 0)
    occupancy = (sdf <= 0).float()
    
    print(f"   Occupancy statistics:")
    print(f"     Total voxels: {occupancy.numel()}")
    print(f"     Inside sphere: {(occupancy == 1).sum().item()}")
    print(f"     Outside sphere: {(occupancy == 0).sum().item()}")
    print(f"     Fill ratio: {occupancy.mean().item():.4f}")
    
    return occupancy


def generate_analytical_gradient_grid(resolution, sphere_radius, sphere_center=None, bbox_size=2.0):
    """
    Generate analytical gradient grid (surface normals) for a sphere.
    
    Args:
        resolution: Grid resolution (int or tuple of 3 ints)
        sphere_radius: Radius of the sphere
        sphere_center: Center of sphere in world coordinates (default: origin)
        bbox_size: Size of the bounding box (grid spans [-bbox_size, bbox_size])
    
    Returns:
        gradient_grid: torch.Tensor of shape (3, D, H, W) with normalized gradients
    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution, resolution)
    
    if sphere_center is None:
        sphere_center = np.array([0.0, 0.0, 0.0])
    
    D, H, W = resolution
    
    print(f"📐 Generating analytical gradient grid 3x{D}x{H}x{W}")
    print(f"   Sphere radius: {sphere_radius}")
    print(f"   Sphere center: {sphere_center}")
    
    # Create coordinate grids
    x = torch.linspace(-bbox_size, bbox_size, W)
    y = torch.linspace(-bbox_size, bbox_size, H)
    z = torch.linspace(-bbox_size, bbox_size, D)
    
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')  # Shape: (D, H, W)
    coords = torch.stack([X, Y, Z], dim=-1)  # Shape: (D, H, W, 3)
    
    # Convert sphere center to tensor
    sphere_center_tensor = torch.tensor(sphere_center, dtype=torch.float32)
    
    # Compute vectors from sphere center to each point
    vectors = coords - sphere_center_tensor  # Shape: (D, H, W, 3)
    
    # Compute distances
    distances = torch.norm(vectors, dim=-1, keepdim=True)  # Shape: (D, H, W, 1)
    
    # Compute unit normal vectors (gradient of SDF)
    # For a sphere, ∇SDF = (p - center) / ||p - center||
    # Handle the case where distance is 0 (at sphere center)
    epsilon = 1e-8
    normals = vectors / (distances + epsilon)  # Shape: (D, H, W, 3)
    
    # At the exact center, set normal to zero vector
    center_mask = (distances.squeeze(-1) < epsilon)
    normals[center_mask] = 0.0
    
    # Rearrange to (3, D, H, W) format expected by the model
    gradient_grid = normals.permute(3, 0, 1, 2)  # Shape: (3, D, H, W)
    
    print(f"   Gradient statistics:")
    print(f"     Gradient magnitude min: {torch.norm(gradient_grid, dim=0).min().item():.6f}")
    print(f"     Gradient magnitude max: {torch.norm(gradient_grid, dim=0).max().item():.6f}")
    print(f"     Gradient magnitude mean: {torch.norm(gradient_grid, dim=0).mean().item():.6f}")
    
    return gradient_grid


def save_analytical_grids(occupancy_grid, gradient_grid, sphere_radius, sphere_center, bbox_size, output_dir):
    """
    Save analytical grids and metadata.
    
    Args:
        occupancy_grid: Binary occupancy grid (D, H, W)
        gradient_grid: Gradient grid (3, D, H, W)
        sphere_radius: Radius of the sphere
        sphere_center: Center of sphere
        bbox_size: Bounding box size
        output_dir: Directory to save files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    resolution = occupancy_grid.shape
    
    # Save grids
    occupancy_path = output_dir / f"analytical_occupancy_grid_{resolution[0]}.pt"
    gradient_path = output_dir / f"analytical_gradient_grid_{resolution[0]}.pt"
    
    torch.save({
        'grid': occupancy_grid,
        'resolution': resolution,
        'sphere_radius': sphere_radius,
        'sphere_center': sphere_center,
        'bbox_size': bbox_size,
        'grid_type': 'binary_occupancy'
    }, occupancy_path)
    
    torch.save({
        'grid': gradient_grid,
        'resolution': resolution,
        'sphere_radius': sphere_radius,
        'sphere_center': sphere_center,
        'bbox_size': bbox_size,
        'grid_type': 'analytical_gradient'
    }, gradient_path)
    
    # Save metadata
    metadata = {
        'dataset_type': 'analytical_sphere_grids',
        'sphere_radius': float(sphere_radius),
        'sphere_center': sphere_center.tolist(),
        'bbox_size': float(bbox_size),
        'resolution': list(resolution),
        'coordinate_system': 'right_handed_z_up',
        'grid_range': [-bbox_size, bbox_size],
        'occupancy_grid_file': occupancy_path.name,
        'gradient_grid_file': gradient_path.name,
        'occupancy_statistics': {
            'total_voxels': int(occupancy_grid.numel()),
            'inside_voxels': int((occupancy_grid == 1).sum().item()),
            'outside_voxels': int((occupancy_grid == 0).sum().item()),
            'fill_ratio': float(occupancy_grid.mean().item())
        },
        'gradient_statistics': {
            'magnitude_min': float(torch.norm(gradient_grid, dim=0).min().item()),
            'magnitude_max': float(torch.norm(gradient_grid, dim=0).max().item()),
            'magnitude_mean': float(torch.norm(gradient_grid, dim=0).mean().item())
        }
    }
    
    metadata_path = output_dir / "analytical_grids_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n💾 Saved analytical grids:")
    print(f"   Occupancy: {occupancy_path}")
    print(f"   Gradient: {gradient_path}")
    print(f"   Metadata: {metadata_path}")


def test_grid_properties(occupancy_grid, gradient_grid):
    """
    Test analytical properties of the generated grids.
    
    Args:
        occupancy_grid: Binary occupancy grid (D, H, W)
        gradient_grid: Gradient grid (3, D, H, W)
    """
    print(f"\n🧪 Testing grid properties...")
    
    # Test 1: Occupancy is truly binary
    unique_values = torch.unique(occupancy_grid)
    print(f"   Occupancy unique values: {unique_values.tolist()}")
    is_binary = len(unique_values) <= 2 and all(v in [0.0, 1.0] for v in unique_values)
    print(f"   ✅ Binary occupancy: {is_binary}")
    
    # Test 2: Gradient magnitudes
    gradient_magnitudes = torch.norm(gradient_grid, dim=0)
    print(f"   Gradient magnitude range: [{gradient_magnitudes.min():.6f}, {gradient_magnitudes.max():.6f}]")
    
    # Test 3: Gradients are normalized (except at center)
    non_zero_mask = gradient_magnitudes > 1e-6
    normalized_gradients = gradient_magnitudes[non_zero_mask]
    print(f"   Non-zero gradients close to unit magnitude: {torch.allclose(normalized_gradients, torch.ones_like(normalized_gradients), atol=1e-5)}")
    
    # Test 4: Gradient points outward from sphere center
    # Sample a few points on the sphere surface and check
    D, H, W = occupancy_grid.shape
    center_d, center_h, center_w = D//2, H//2, W//2
    
    # Check gradient at a point away from center
    test_point = (center_d, center_h, center_w + 10)  # Point to the right of center
    if test_point[2] < W:
        gradient_at_point = gradient_grid[:, test_point[0], test_point[1], test_point[2]]
        print(f"   Gradient at test point {test_point}: {gradient_at_point.tolist()}")
        print(f"   Points in positive X direction: {gradient_at_point[0] > 0}")
    
    print(f"   ✅ Grid properties validated")


def main():
    parser = argparse.ArgumentParser(description="Generate analytical sphere grids")
    parser.add_argument("--sphere_radius", type=float, default=1.0,
                       help="Radius of the sphere")
    parser.add_argument("--sphere_center", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                       help="Center of sphere (x, y, z)")
    parser.add_argument("--bbox_size", type=float, default=2.0,
                       help="Half-size of bounding box (grid spans [-bbox_size, bbox_size])")
    parser.add_argument("--resolution", type=int, default=128,
                       help="Grid resolution (cubic)")
    parser.add_argument("--output_dir", default="analytical_grids",
                       help="Output directory for grids")
    parser.add_argument("--test", action="store_true",
                       help="Run property tests on generated grids")
    
    args = parser.parse_args()
    
    sphere_center = np.array(args.sphere_center)
    
    print(f"🌟 Generating analytical sphere grids")
    print(f"   Resolution: {args.resolution}³")
    print(f"   Sphere radius: {args.sphere_radius}")
    print(f"   Sphere center: {sphere_center}")
    print(f"   Bbox size: ±{args.bbox_size}")
    
    # Generate grids
    occupancy_grid = generate_analytical_occupancy_grid(
        resolution=args.resolution,
        sphere_radius=args.sphere_radius,
        sphere_center=sphere_center,
        bbox_size=args.bbox_size
    )
    
    gradient_grid = generate_analytical_gradient_grid(
        resolution=args.resolution,
        sphere_radius=args.sphere_radius,
        sphere_center=sphere_center,
        bbox_size=args.bbox_size
    )
    
    # Save grids
    save_analytical_grids(
        occupancy_grid=occupancy_grid,
        gradient_grid=gradient_grid,
        sphere_radius=args.sphere_radius,
        sphere_center=sphere_center,
        bbox_size=args.bbox_size,
        output_dir=args.output_dir
    )
    
    # Optional testing
    if args.test:
        test_grid_properties(occupancy_grid, gradient_grid)
    
    print(f"\n🎉 Analytical grid generation complete!")


if __name__ == "__main__":
    main() 