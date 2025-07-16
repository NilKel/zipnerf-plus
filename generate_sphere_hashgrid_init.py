#!/usr/bin/env python3
"""
Generate pre-initialized hashgrid embeddings for sphere geometry.
This approach uses sparse sampling instead of full 3D coordinate grids.
"""

import numpy as np
import torch
import os
from pathlib import Path

def hash_encode(coords, log2_hashmap_size, level, base_resolution, per_level_scale):
    """
    Simplified hash encoding function to mimic GridEncoder's hash mapping.
    This maps 3D coordinates to hash table indices for a given level.
    """
    # Calculate resolution for this level
    resolution = int(base_resolution * (per_level_scale ** level))
    
    # Scale coordinates to grid resolution
    coords_scaled = coords * resolution / 2.0 + resolution / 2.0
    coords_int = coords_scaled.long().clamp(0, resolution - 1)
    
    # Simple hash function (simplified version of what GridEncoder uses)
    hashmap_size = 2 ** log2_hashmap_size
    
    # Hash the integer coordinates
    hash_coords = coords_int[..., 0] * 73856093 + coords_int[..., 1] * 19349663 + coords_int[..., 2] * 83492791
    hash_indices = hash_coords % hashmap_size
    
    return hash_indices

def generate_sphere_hashgrid_init(sphere_radius=1.0, sphere_center=None, num_samples=100000):
    """
    Generate pre-initialized hashgrid embeddings for sphere geometry.
    
    Args:
        sphere_radius: Radius of the sphere
        sphere_center: Center of the sphere [x, y, z]
        num_samples: Number of sample points to use for initialization
    """
    if sphere_center is None:
        sphere_center = [0.0, 0.0, 0.0]
    
    sphere_center = torch.tensor(sphere_center, dtype=torch.float32)
    
    print(f"🌟 Generating sphere hashgrid initialization")
    print(f"   Sphere radius: {sphere_radius}")
    print(f"   Sphere center: {sphere_center.tolist()}")
    print(f"   Sample points: {num_samples:,}")
    
    # Grid parameters (matching our config)
    base_resolution = 16
    per_level_scale = 2.0  # This will be overridden by model config
    log2_hashmap_size = 19
    num_levels = 10  # This will be calculated by model
    level_dim = 1
    
    # Sample points around the sphere
    # Mix of surface points and volume points
    surface_ratio = 0.7
    num_surface = int(num_samples * surface_ratio)
    num_volume = num_samples - num_surface
    
    print(f"📍 Sampling {num_surface:,} surface points and {num_volume:,} volume points")
    
    # Surface points (on sphere surface)
    theta = torch.rand(num_surface) * 2 * np.pi  # azimuth
    phi = torch.acos(1 - 2 * torch.rand(num_surface))  # polar (uniform on sphere)
    
    surface_points = torch.stack([
        sphere_radius * torch.sin(phi) * torch.cos(theta),
        sphere_radius * torch.sin(phi) * torch.sin(theta),
        sphere_radius * torch.cos(phi)
    ], dim=-1) + sphere_center
    
    # Volume points (around sphere)
    volume_radius = torch.rand(num_volume) * sphere_radius * 1.5  # Extend beyond surface
    theta_vol = torch.rand(num_volume) * 2 * np.pi
    phi_vol = torch.acos(1 - 2 * torch.rand(num_volume))
    
    volume_points = torch.stack([
        volume_radius * torch.sin(phi_vol) * torch.cos(theta_vol),
        volume_radius * torch.sin(phi_vol) * torch.sin(theta_vol),
        volume_radius * torch.cos(phi_vol)
    ], dim=-1) + sphere_center
    
    # Combine all sample points
    all_points = torch.cat([surface_points, volume_points], dim=0)
    
    # Normalize points to [-1, 1] (matching NeRF coordinate system)
    all_points = all_points / 3.0  # Assuming scene bounds of [-3, 3]
    all_points = torch.clamp(all_points, -1, 1)
    
    # Compute distance from sphere center for each point
    distances = torch.norm(all_points - sphere_center / 3.0, dim=-1)
    
    # Create sphere-aware initialization values based on distance
    # Points near surface get higher values, points far away get lower values
    surface_distance = torch.abs(distances - sphere_radius / 3.0)
    init_values = torch.exp(-surface_distance * 5.0)  # Exponential falloff
    
    print(f"📊 Initialization value range: [{init_values.min():.4f}, {init_values.max():.4f}]")
    
    # For demonstration, we'll create a mapping of hash indices to initialization values
    # In practice, this would be integrated into the GridEncoder initialization
    hashgrid_init_data = {
        'sphere_radius': sphere_radius,
        'sphere_center': sphere_center.tolist(),
        'sample_points': all_points,
        'init_values': init_values,
        'grid_params': {
            'base_resolution': base_resolution,
            'per_level_scale': per_level_scale,
            'log2_hashmap_size': log2_hashmap_size,
            'num_levels': num_levels,
            'level_dim': level_dim,
        }
    }
    
    # Save initialization data
    output_dir = Path('sphere_hashgrid_init')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'sphere_hashgrid_init.pt'
    
    torch.save(hashgrid_init_data, output_file)
    
    print(f"✅ Saved sphere hashgrid initialization to: {output_file}")
    print(f"   Sample points shape: {all_points.shape}")
    print(f"   Values shape: {init_values.shape}")
    
    return hashgrid_init_data

if __name__ == "__main__":
    generate_sphere_hashgrid_init() 