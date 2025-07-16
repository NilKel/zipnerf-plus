#!/usr/bin/env python3
"""
Generate Analytical Sphere Dataset for Vector Potential Testing (No Shading)

This script creates a synthetic dataset of a sphere with uniform color (no lighting/shading)
following the NeRF synthetic format. This is specifically for the analytical experiment
where we want to test the mathematical formulation without lighting complexity.

Usage:
    python generate_analytical_sphere_dataset.py --output_dir ../data/nerf_synthetic/sphere_analytical_simple
"""

import numpy as np
import torch
import json
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_camera_poses_spherical(n_poses, radius=4.0, camera_distance=4.0, up_variance=0.3):
    """
    Generate camera poses on a sphere looking at the center.
    
    Args:
        n_poses: Number of camera poses to generate
        radius: Radius of the sphere being observed (for reference)
        camera_distance: Distance of cameras from origin
        up_variance: Variance in the up direction to avoid all cameras on equator
    
    Returns:
        poses: Array of shape (n_poses, 4, 4) containing camera-to-world matrices
        fov: Field of view in radians
    """
    poses = []
    
    # Generate evenly spaced points on sphere with some randomization
    for i in range(n_poses):
        # Generate spherical coordinates
        # Azimuth angle (0 to 2π)
        azimuth = (2 * np.pi * i) / n_poses + np.random.normal(0, 0.1)
        
        # Elevation angle (with variance around equator)
        elevation = np.random.normal(0, up_variance)
        elevation = np.clip(elevation, -np.pi/3, np.pi/3)  # Avoid extreme angles
        
        # Convert to Cartesian coordinates
        x = camera_distance * np.cos(elevation) * np.cos(azimuth)
        y = camera_distance * np.cos(elevation) * np.sin(azimuth)
        z = camera_distance * np.sin(elevation)
        
        camera_pos = np.array([x, y, z])
        
        # Camera looks at origin
        forward = -camera_pos / np.linalg.norm(camera_pos)  # Look towards origin
        
        # Compute right and up vectors
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Construct camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = camera_pos
        
        poses.append(c2w)
    
    poses = np.array(poses)
    
    # Field of view - set so sphere roughly fills the image
    fov = 2 * np.arctan(radius * 1.2 / camera_distance)  # 20% margin around sphere
    
    return poses, fov


def render_sphere_image_no_shading(c2w, intrinsics, image_size, sphere_center, sphere_radius, 
                                  sphere_color=None, bg_color=0.0):
    """
    Render an image of a sphere with uniform color (no lighting/shading).
    
    Args:
        c2w: Camera-to-world matrix (4x4)
        intrinsics: Camera intrinsics matrix (3x3)
        image_size: (height, width) of output image
        sphere_center: 3D position of sphere center
        sphere_radius: Radius of the sphere
        sphere_color: RGB color of sphere [r, g, b] in [0,1] (default: light gray)
        bg_color: Background color (0.0 = black, 1.0 = white)
    
    Returns:
        image: Rendered image as numpy array (H, W, 3)
        depth: Depth map as numpy array (H, W)
    """
    if sphere_color is None:
        sphere_color = np.array([0.8, 0.8, 0.8])  # Light gray
    
    height, width = image_size
    image = np.full((height, width, 3), bg_color, dtype=np.float32)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    
    # Get camera position and orientation
    camera_pos = c2w[:3, 3]
    w2c = np.linalg.inv(c2w)
    
    # Generate ray directions for each pixel
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Create meshgrid of pixel coordinates
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    
    # Convert to normalized device coordinates
    dirs = np.stack([
        (i - cx) / fx,
        -(j - cy) / fy,  # Negative because image y is flipped
        np.ones_like(i)   # Positive Z for forward direction
    ], axis=-1)
    
    # Transform ray directions to world space
    dirs_world = dirs @ c2w[:3, :3].T
    dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    
    # Ray-sphere intersection
    # Ray: p(t) = camera_pos + t * dir
    # Sphere: ||p - sphere_center||^2 = radius^2
    
    oc = camera_pos - sphere_center  # Vector from sphere center to camera
    
    # Vectorized ray-sphere intersection
    dirs_flat = dirs_world.reshape(-1, 3)
    
    a = np.sum(dirs_flat ** 2, axis=1)
    b = 2.0 * np.sum(oc * dirs_flat, axis=1)
    c = np.sum(oc ** 2) - sphere_radius ** 2
    
    discriminant = b ** 2 - 4 * a * c
    
    # Find intersections
    valid = discriminant >= 0
    
    if np.any(valid):
        sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        
        # Use nearest positive intersection
        t = np.where(t1 > 0, t1, t2)
        t = np.where(t > 0, t, np.inf)
        
        # Simple uniform color (no lighting)
        t_reshaped = t.reshape(height, width)
        valid_reshaped = (t_reshaped < np.inf) & valid.reshape(height, width)
        
        # Set uniform sphere color for all hit pixels
        image[valid_reshaped] = sphere_color
        depth[valid_reshaped] = t_reshaped[valid_reshaped]
    
    return image, depth


def save_transforms_json(poses, fov, image_paths, output_path):
    """Save camera poses in NeRF synthetic format."""
    
    data = {
        "camera_angle_x": float(fov),
        "frames": []
    }
    
    for i, (pose, image_path) in enumerate(zip(poses, image_paths)):
        frame = {
            "file_path": image_path,
            "rotation": 0.0,
            "transform_matrix": pose.tolist()
        }
        data["frames"].append(frame)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def generate_analytical_sphere_dataset(output_dir, 
                                      n_train=100, n_val=100, n_test=200,
                                      image_size=800, 
                                      sphere_radius=1.0, 
                                      camera_distance=4.0,
                                      sphere_color=None):
    """
    Generate complete sphere dataset in NeRF synthetic format with no shading.
    
    Args:
        output_dir: Directory to save the dataset
        n_train, n_val, n_test: Number of images for each split
        image_size: Size of rendered images (assumed square)
        sphere_radius: Radius of the sphere
        camera_distance: Distance of cameras from origin
        sphere_color: RGB color of sphere (default: light gray)
    """
    if sphere_color is None:
        sphere_color = np.array([0.8, 0.8, 0.8])  # Light gray
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(exist_ok=True)
    
    print(f"🌍 Generating analytical sphere dataset (no shading) in {output_dir}")
    print(f"📐 Sphere radius: {sphere_radius}")
    print(f"📷 Camera distance: {camera_distance}")
    print(f"🖼️  Image size: {image_size}x{image_size}")
    print(f"🎨 Sphere color: {sphere_color}")
    
    # Camera intrinsics
    fov_x = 2 * np.arctan(sphere_radius * 1.2 / camera_distance)
    focal = image_size / (2 * np.tan(fov_x / 2))
    intrinsics = np.array([
        [focal, 0, image_size/2],
        [0, focal, image_size/2],
        [0, 0, 1]
    ])
    
    sphere_center = np.array([0.0, 0.0, 0.0])
    
    # Generate datasets
    splits = [
        ('train', n_train),
        ('val', n_val), 
        ('test', n_test)
    ]
    
    for split_name, n_images in splits:
        print(f"\n📸 Generating {split_name} split ({n_images} images)...")
        
        # Generate camera poses
        poses, fov = generate_camera_poses_spherical(
            n_images, 
            radius=sphere_radius,
            camera_distance=camera_distance,
            up_variance=0.3 if split_name == 'train' else 0.5  # More variety in val/test
        )
        
        # Render images
        image_paths = []
        for i in tqdm(range(n_images), desc=f"Rendering {split_name}"):
            # Render image with no shading
            image, depth = render_sphere_image_no_shading(
                poses[i], intrinsics, (image_size, image_size),
                sphere_center, sphere_radius, sphere_color, bg_color=0.0
            )
            
            # Convert to uint8 and save
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            image_path = output_dir / split_name / f"r_{i}.png"
            cv2.imwrite(str(image_path), cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))
            
            image_paths.append(f"{split_name}/r_{i}")
        
        # Save transforms
        transforms_path = output_dir / f"transforms_{split_name}.json"
        save_transforms_json(poses, fov, image_paths, transforms_path)
        
        print(f"✅ Saved {split_name}: {n_images} images, FOV={np.degrees(fov):.1f}°")
    
    # Save dataset metadata
    metadata = {
        "dataset_type": "analytical_sphere_no_shading",
        "sphere_radius": sphere_radius,
        "sphere_center": sphere_center.tolist(),
        "sphere_color": sphere_color.tolist(),
        "camera_distance": camera_distance,
        "image_size": image_size,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "fov_degrees": float(np.degrees(fov)),
        "coordinate_system": "right_handed_z_up",
        "shading": "none",
        "description": "Sphere with uniform color for analytical testing"
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n🎉 Dataset generation complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total images: {n_train + n_val + n_test}")
    print(f"🎨 Sphere rendered with uniform color (no shading)")


def visualize_sample_images(output_dir, n_samples=4):
    """Visualize a few sample images from the generated dataset."""
    output_dir = Path(output_dir)
    
    # Load some sample images
    train_dir = output_dir / "train"
    sample_files = list(train_dir.glob("*.png"))[:n_samples]
    
    if not sample_files:
        print("No sample images found")
        return
    
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 4))
    if n_samples == 1:
        axes = [axes]
    
    for i, img_path in enumerate(sample_files):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "sample_images.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📸 Sample images saved to {output_dir / 'sample_images.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate analytical sphere dataset (no shading)")
    parser.add_argument("--output_dir", default="../data/nerf_synthetic/sphere_analytical_simple",
                       help="Output directory for the dataset")
    parser.add_argument("--sphere_radius", type=float, default=1.0,
                       help="Radius of the sphere")
    parser.add_argument("--camera_distance", type=float, default=4.0,
                       help="Distance of cameras from origin")
    parser.add_argument("--image_size", type=int, default=800,
                       help="Size of rendered images (square)")
    parser.add_argument("--n_train", type=int, default=50,
                       help="Number of training images")
    parser.add_argument("--n_val", type=int, default=25,
                       help="Number of validation images") 
    parser.add_argument("--n_test", type=int, default=50,
                       help="Number of test images")
    parser.add_argument("--sphere_color", type=float, nargs=3, default=[0.8, 0.8, 0.8],
                       help="RGB color of sphere (r g b) in [0,1]")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate sample visualizations")
    
    args = parser.parse_args()
    
    sphere_color = np.array(args.sphere_color)
    
    generate_analytical_sphere_dataset(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        image_size=args.image_size,
        sphere_radius=args.sphere_radius,
        camera_distance=args.camera_distance,
        sphere_color=sphere_color
    )
    
    if args.visualize:
        visualize_sample_images(args.output_dir)


if __name__ == "__main__":
    main() 