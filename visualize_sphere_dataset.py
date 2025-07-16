#!/usr/bin/env python3
"""
Visualize Sphere Dataset and Analytical Grids

This script loads and visualizes the generated sphere dataset and analytical grids
to verify they are correct for the vector potential testing.

Usage:
    python visualize_sphere_dataset.py --dataset_dir ../data/nerf_synthetic/sphere_analytical
"""

import numpy as np
import torch
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def visualize_dataset_images(dataset_dir, n_samples=4):
    """Visualize sample images from the sphere dataset."""
    dataset_dir = Path(dataset_dir)
    
    print(f"📊 Visualizing sphere dataset from {dataset_dir}")
    
    # Load metadata
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"   Dataset type: {metadata['dataset_type']}")
        print(f"   Sphere radius: {metadata['sphere_radius']}")
        print(f"   Camera distance: {metadata['camera_distance']}")
        print(f"   Image size: {metadata['image_size']}x{metadata['image_size']}")
    
    # Visualize sample images from each split
    splits = ['train', 'val', 'test']
    
    fig, axes = plt.subplots(len(splits), n_samples, figsize=(n_samples*3, len(splits)*3))
    if len(splits) == 1:
        axes = axes.reshape(1, -1)
    
    for split_idx, split_name in enumerate(splits):
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
            continue
            
        # Get list of images
        image_files = sorted(list(split_dir.glob("r_*.png")))
        
        # Sample images evenly
        if len(image_files) > 0:
            indices = np.linspace(0, len(image_files)-1, min(n_samples, len(image_files)), dtype=int)
            
            for i, idx in enumerate(indices):
                if i >= n_samples:
                    break
                    
                # Load and display image
                image_path = image_files[idx]
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                ax = axes[split_idx, i] if len(splits) > 1 else axes[i]
                ax.imshow(image)
                ax.set_title(f"{split_name}: {image_path.name}")
                ax.axis('off')
        
        # Fill remaining subplot slots
        for i in range(len(indices) if len(image_files) > 0 else 0, n_samples):
            ax = axes[split_idx, i] if len(splits) > 1 else axes[i]
            ax.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(dataset_dir / "dataset_visualization.png", dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved visualization: {dataset_dir / 'dataset_visualization.png'}")
    plt.show()


def visualize_analytical_grids(grids_dir, slice_indices=None):
    """Visualize analytical occupancy and gradient grids."""
    grids_dir = Path(grids_dir)
    
    print(f"📐 Visualizing analytical grids from {grids_dir}")
    
    # Load metadata
    metadata_path = grids_dir / "analytical_grids_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"   Resolution: {metadata['resolution']}")
        print(f"   Sphere radius: {metadata['sphere_radius']}")
        print(f"   Fill ratio: {metadata['occupancy_statistics']['fill_ratio']:.4f}")
    
    # Load grids
    resolution = metadata['resolution'][0] if metadata_path.exists() else 64
    
    occupancy_file = grids_dir / f"analytical_occupancy_grid_{resolution}.pt"
    gradient_file = grids_dir / f"analytical_gradient_grid_{resolution}.pt"
    
    if not occupancy_file.exists() or not gradient_file.exists():
        print(f"   ❌ Grid files not found")
        return
    
    occupancy_data = torch.load(occupancy_file, weights_only=False)
    gradient_data = torch.load(gradient_file, weights_only=False)
    
    occupancy_grid = occupancy_data['grid']
    gradient_grid = gradient_data['grid']
    
    print(f"   Occupancy grid shape: {occupancy_grid.shape}")
    print(f"   Gradient grid shape: {gradient_grid.shape}")
    
    # Default slice indices (middle slices)
    if slice_indices is None:
        D, H, W = occupancy_grid.shape
        slice_indices = [D//2, H//2, W//2]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Occupancy slices
    axes[0, 0].imshow(occupancy_grid[slice_indices[0], :, :].numpy(), cmap='binary')
    axes[0, 0].set_title(f'Occupancy XY (z={slice_indices[0]})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(occupancy_grid[:, slice_indices[1], :].numpy(), cmap='binary')
    axes[0, 1].set_title(f'Occupancy XZ (y={slice_indices[1]})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(occupancy_grid[:, :, slice_indices[2]].numpy(), cmap='binary')
    axes[0, 2].set_title(f'Occupancy YZ (x={slice_indices[2]})')
    axes[0, 2].axis('off')
    
    # Gradient magnitude slices
    grad_magnitude = torch.norm(gradient_grid, dim=0)
    
    axes[1, 0].imshow(grad_magnitude[slice_indices[0], :, :].numpy(), cmap='viridis')
    axes[1, 0].set_title(f'Gradient Magnitude XY (z={slice_indices[0]})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(grad_magnitude[:, slice_indices[1], :].numpy(), cmap='viridis')
    axes[1, 1].set_title(f'Gradient Magnitude XZ (y={slice_indices[1]})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(grad_magnitude[:, :, slice_indices[2]].numpy(), cmap='viridis')
    axes[1, 2].set_title(f'Gradient Magnitude YZ (x={slice_indices[2]})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(grids_dir / "grids_visualization.png", dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved visualization: {grids_dir / 'grids_visualization.png'}")
    plt.show()


def verify_coordinate_consistency(dataset_dir, grids_dir):
    """Verify that dataset and grids use consistent coordinate systems."""
    print(f"🔍 Verifying coordinate consistency...")
    
    # Load dataset metadata
    dataset_metadata_path = Path(dataset_dir) / "metadata.json"
    if dataset_metadata_path.exists():
        with open(dataset_metadata_path, 'r') as f:
            dataset_metadata = json.load(f)
    else:
        print(f"   ❌ Dataset metadata not found")
        return
    
    # Load grids metadata
    grids_metadata_path = Path(grids_dir) / "analytical_grids_metadata.json"
    if grids_metadata_path.exists():
        with open(grids_metadata_path, 'r') as f:
            grids_metadata = json.load(f)
    else:
        print(f"   ❌ Grids metadata not found")
        return
    
    # Check sphere parameters match
    dataset_radius = dataset_metadata['sphere_radius']
    grids_radius = grids_metadata['sphere_radius']
    
    dataset_coords = dataset_metadata['coordinate_system']
    grids_coords = grids_metadata['coordinate_system']
    
    print(f"   Dataset sphere radius: {dataset_radius}")
    print(f"   Grids sphere radius: {grids_radius}")
    print(f"   Dataset coordinate system: {dataset_coords}")
    print(f"   Grids coordinate system: {grids_coords}")
    
    if abs(dataset_radius - grids_radius) < 1e-6:
        print(f"   ✅ Sphere radii match")
    else:
        print(f"   ❌ Sphere radii mismatch!")
    
    if dataset_coords == grids_coords:
        print(f"   ✅ Coordinate systems match")
    else:
        print(f"   ❌ Coordinate systems mismatch!")


def main():
    parser = argparse.ArgumentParser(description="Visualize sphere dataset and analytical grids")
    parser.add_argument("--dataset_dir", default="../data/nerf_synthetic/sphere_analytical",
                       help="Directory containing the sphere dataset")
    parser.add_argument("--grids_dir", default="sphere_analytical_grids",
                       help="Directory containing the analytical grids")
    parser.add_argument("--n_samples", type=int, default=4,
                       help="Number of sample images to show per split")
    
    args = parser.parse_args()
    
    print(f"🌟 Sphere Dataset & Grids Visualization")
    print(f"=" * 50)
    
    # Check if paths exist
    dataset_dir = Path(args.dataset_dir)
    grids_dir = Path(args.grids_dir)
    
    if dataset_dir.exists():
        visualize_dataset_images(dataset_dir, args.n_samples)
    else:
        print(f"❌ Dataset directory not found: {dataset_dir}")
    
    if grids_dir.exists():
        visualize_analytical_grids(grids_dir)
    else:
        print(f"❌ Grids directory not found: {grids_dir}")
    
    if dataset_dir.exists() and grids_dir.exists():
        verify_coordinate_consistency(dataset_dir, grids_dir)
    
    print(f"\n🎉 Visualization complete!")


if __name__ == "__main__":
    main() 