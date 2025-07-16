#!/usr/bin/env python3
"""
Generate Analytical Sphere Dataset for Vector Potential Testing

This script creates a synthetic dataset of a sphere following the NeRF synthetic format.
It generates camera poses and renders images for training, validation, and testing.

Usage:
    python generate_sphere_dataset.py --output_dir ../data/nerf_synthetic/sphere_analytical
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


def render_sphere_image(c2w, intrinsics, image_size, sphere_center, sphere_radius, bg_color=0.0):
    """
    Render an image of a sphere from the given camera pose.
    
    Args:
        c2w: Camera-to-world matrix (4x4)
        intrinsics: Camera intrinsics matrix (3x3)
        image_size: (height, width) of output image
        sphere_center: 3D position of sphere center
        sphere_radius: Radius of the sphere
        bg_color: Background color (0.0 = black, 1.0 = white)
    
    Returns:
        image: Rendered image as numpy array (H, W, 3)
        depth: Depth map as numpy array (H, W)
    """
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
        
        # Compute intersection points and normals
        hit_points = camera_pos[None, :] + t[:, None] * dirs_flat
        normals = (hit_points - sphere_center) / sphere_radius
        
        # Simple lighting model (Lambertian shading)
        light_dir = np.array([0.5, 0.5, 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Compute shading
        cos_theta = np.maximum(np.sum(normals * light_dir, axis=1), 0.0)
        ambient = 0.3
        diffuse = 0.7
        intensity = ambient + diffuse * cos_theta
        
        # Sphere color (light gray)
        sphere_color = np.array([0.8, 0.8, 0.8])
        final_colors = intensity[:, None] * sphere_color
        
        # Update image and depth
        t_reshaped = t.reshape(height, width)
        final_colors_reshaped = final_colors.reshape(height, width, 3)
        
        valid_reshaped = (t_reshaped < np.inf) & valid.reshape(height, width)
        
        image[valid_reshaped] = final_colors_reshaped[valid_reshaped]
        depth[valid_reshaped] = t_reshaped[valid_reshaped]
    
    return image, depth


def save_transforms_json(poses, fov, image_paths, output_path):
    """Save camera poses and metadata in NeRF synthetic format."""
    frames = []
    
    for i, (pose, img_path) in enumerate(zip(poses, image_paths)):
        # Remove file extension and add relative path
        file_path = str(Path(img_path).with_suffix(''))
        if not file_path.startswith('./'):
            file_path = './' + file_path
        
        frame = {
            "file_path": file_path,
            "rotation": 0.0,  # No rolling motion
            "transform_matrix": pose.tolist()
        }
        frames.append(frame)
    
    transforms = {
        "camera_angle_x": float(fov),
        "frames": frames
    }
    
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=4)


def generate_sphere_dataset(output_dir, 
                          n_train=100, n_val=100, n_test=200,
                          image_size=800, 
                          sphere_radius=1.0, 
                          camera_distance=4.0):
    """
    Generate complete sphere dataset in NeRF synthetic format.
    
    Args:
        output_dir: Directory to save the dataset
        n_train, n_val, n_test: Number of images for each split
        image_size: Size of rendered images (assumed square)
        sphere_radius: Radius of the sphere
        camera_distance: Distance of cameras from origin
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(exist_ok=True)
    
    print(f"🌍 Generating sphere dataset in {output_dir}")
    print(f"📐 Sphere radius: {sphere_radius}")
    print(f"📷 Camera distance: {camera_distance}")
    print(f"🖼️  Image size: {image_size}x{image_size}")
    
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
            # Render image
            image, depth = render_sphere_image(
                poses[i], intrinsics, (image_size, image_size),
                sphere_center, sphere_radius, bg_color=0.0
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
        "dataset_type": "analytical_sphere",
        "sphere_radius": sphere_radius,
        "camera_distance": camera_distance,
        "image_size": image_size,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "fov_degrees": float(np.degrees(fov)),
        "coordinate_system": "right_handed_z_up"
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n🎉 Dataset generation complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total images: {n_train + n_val + n_test}")


def main():
    parser = argparse.ArgumentParser(description="Generate analytical sphere dataset")
    parser.add_argument("--output_dir", default="../data/nerf_synthetic/sphere_analytical",
                       help="Output directory for the dataset")
    parser.add_argument("--sphere_radius", type=float, default=1.0,
                       help="Radius of the sphere")
    parser.add_argument("--camera_distance", type=float, default=4.0,
                       help="Distance of cameras from origin")
    parser.add_argument("--image_size", type=int, default=800,
                       help="Size of rendered images (square)")
    parser.add_argument("--n_train", type=int, default=100,
                       help="Number of training images")
    parser.add_argument("--n_val", type=int, default=100,
                       help="Number of validation images") 
    parser.add_argument("--n_test", type=int, default=200,
                       help="Number of test images")
    
    args = parser.parse_args()
    
    generate_sphere_dataset(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        image_size=args.image_size,
        sphere_radius=args.sphere_radius,
        camera_distance=args.camera_distance
    )


if __name__ == "__main__":
    main() 