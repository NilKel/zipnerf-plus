#!/usr/bin/env python3
"""
Generate sphere confidence occupancy probabilities directly.
This is cleaner since we're only using them for sampling guidance, not initialization.
"""

import numpy as np
import torch
import os
from pathlib import Path

def generate_sphere_confidence_probabilities(sphere_radius=1.0, sphere_center=None, resolutions=[64, 128, 256]):
    """
    Generate confidence occupancy probabilities directly for sphere geometry.
    Values are between 0 and 1, representing occupancy probability.
    
    Args:
        sphere_radius: Radius of the sphere
        sphere_center: Center of the sphere [x, y, z] 
        resolutions: List of grid resolutions to generate
    """
    if sphere_center is None:
        sphere_center = [0.0, 0.0, 0.0]
    
    print(f"🌍 Generating sphere confidence occupancy probabilities")
    print(f"   Sphere radius: {sphere_radius}")
    print(f"   Sphere center: {sphere_center}")
    print(f"   Resolutions: {resolutions}")
    
    output_dir = Path('sphere_confidence_probabilities')
    output_dir.mkdir(exist_ok=True)
    
    for resolution in resolutions:
        print(f"\n📐 Generating {resolution}³ confidence probability grid...")
        
        # Create coordinate grid
        coords = torch.linspace(-2, 2, resolution)  # Scene bounds [-2, 2]
        Z, Y, X = torch.meshgrid(coords, coords, coords, indexing='ij')
        grid_coords = torch.stack([X, Y, Z], dim=-1)  # (res, res, res, 3)
        
        # Compute distance from sphere center
        sphere_center_tensor = torch.tensor(sphere_center, dtype=torch.float32)
        distances = torch.norm(grid_coords - sphere_center_tensor, dim=-1)
        
        # Compute signed distance field (SDF)
        sdf = distances - sphere_radius
        
        # Convert SDF directly to occupancy probabilities
        # Using sigmoid-like function: high probability near surface, low far away
        sharpness = 5.0  # Controls transition sharpness
        occupancy_probs = torch.sigmoid(-sdf * sharpness)
        
        # Ensure probabilities are in [0, 1] (sigmoid guarantees this, but explicit for clarity)
        occupancy_probs = torch.clamp(occupancy_probs, 0.0, 1.0)
        
        # Statistics
        mean_prob = occupancy_probs.mean()
        high_conf_count = (occupancy_probs > 0.5).sum()
        total_voxels = occupancy_probs.numel()
        
        print(f"   📊 Probability statistics:")
        print(f"      Range: [{occupancy_probs.min():.6f}, {occupancy_probs.max():.6f}]")
        print(f"      Mean: {mean_prob:.6f}")
        print(f"      High confidence voxels (>0.5): {high_conf_count}/{total_voxels}")
        print(f"      High confidence ratio: {high_conf_count/total_voxels:.4f}")
        
        # Save probability grid
        output_file = output_dir / f'sphere_confidence_probs_{resolution}.pt'
        
        # Save with metadata
        save_data = {
            'occupancy_probabilities': occupancy_probs,
            'sphere_radius': sphere_radius,
            'sphere_center': sphere_center,
            'resolution': resolution,
            'scene_bounds': [-2.0, 2.0],
            'sharpness': sharpness,
            'statistics': {
                'mean_probability': mean_prob.item(),
                'min_probability': occupancy_probs.min().item(),
                'max_probability': occupancy_probs.max().item(),
                'high_confidence_count': high_conf_count.item(),
                'total_voxels': total_voxels,
                'high_confidence_ratio': (high_conf_count/total_voxels).item(),
            }
        }
        
        torch.save(save_data, output_file)
        print(f"   ✅ Saved to: {output_file}")
    
    print(f"\n🎉 Generated confidence probability grids for all resolutions!")
    return output_dir

if __name__ == "__main__":
    generate_sphere_confidence_probabilities() 