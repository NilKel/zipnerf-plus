#!/usr/bin/env python3
"""
Debug Sphere Rendering Issues

This script helps debug why the sphere images are coming out black.
"""

import numpy as np
import json
import cv2
from pathlib import Path

def debug_camera_pose():
    """Check camera poses from the generated dataset."""
    with open('../data/nerf_synthetic/sphere_analytical/transforms_train.json', 'r') as f:
        transforms = json.load(f)
    
    print("🔍 Debugging Camera Poses")
    print(f"Camera FOV: {transforms['camera_angle_x']} radians = {np.degrees(transforms['camera_angle_x']):.1f} degrees")
    
    # Check first camera pose
    pose = np.array(transforms['frames'][0]['transform_matrix'])
    camera_pos = pose[:3, 3]
    
    print(f"First camera position: {camera_pos}")
    print(f"Distance from origin: {np.linalg.norm(camera_pos):.3f}")
    
    # Check if camera is looking at origin
    forward = pose[:3, 2]
    to_origin = -camera_pos / np.linalg.norm(camera_pos)
    
    print(f"Camera forward direction: {forward}")
    print(f"Direction to origin: {to_origin}")
    print(f"Dot product (should be ~1): {np.dot(forward, to_origin):.3f}")

def debug_ray_sphere_intersection():
    """Test ray-sphere intersection with simple case."""
    print("\n🧮 Testing Ray-Sphere Intersection")
    
    # Simple test case: camera at (0,0,4) looking at sphere at origin with radius 1
    camera_pos = np.array([0.0, 0.0, 4.0])
    sphere_center = np.array([0.0, 0.0, 0.0])
    sphere_radius = 1.0
    
    # Ray from camera towards sphere center
    ray_dir = (sphere_center - camera_pos) / np.linalg.norm(sphere_center - camera_pos)
    
    print(f"Camera position: {camera_pos}")
    print(f"Ray direction: {ray_dir}")
    print(f"Sphere center: {sphere_center}")
    print(f"Sphere radius: {sphere_radius}")
    
    # Ray-sphere intersection calculation
    oc = camera_pos - sphere_center
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere_radius ** 2
    
    discriminant = b * b - 4 * a * c
    
    print(f"Ray parameters: a={a:.3f}, b={b:.3f}, c={c:.3f}")
    print(f"Discriminant: {discriminant:.3f}")
    
    if discriminant >= 0:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        print(f"Intersection points: t1={t1:.3f}, t2={t2:.3f}")
        
        # Check hit points
        hit1 = camera_pos + t1 * ray_dir
        hit2 = camera_pos + t2 * ray_dir
        print(f"Hit point 1: {hit1}")
        print(f"Hit point 2: {hit2}")
        print(f"Distance to sphere center: {np.linalg.norm(hit1 - sphere_center):.3f}")
    else:
        print("❌ No intersection found!")

def test_simple_render():
    """Test rendering with a simple setup."""
    print("\n🎨 Testing Simple Render")
    
    # Simple camera setup
    image_size = 100
    sphere_radius = 1.0
    camera_distance = 4.0
    
    # Camera at (0, 0, camera_distance) looking at origin
    c2w = np.eye(4)
    c2w[:3, 3] = [0, 0, camera_distance]  # Camera position
    # Camera looks in -Z direction (towards origin)
    c2w[:3, 2] = [0, 0, -1]  # Forward direction
    c2w[:3, 0] = [1, 0, 0]   # Right direction  
    c2w[:3, 1] = [0, 1, 0]   # Up direction
    
    # Simple intrinsics
    focal = image_size / 2  # 90 degree FOV
    intrinsics = np.array([
        [focal, 0, image_size/2],
        [0, focal, image_size/2],
        [0, 0, 1]
    ])
    
    print(f"Camera position: {c2w[:3, 3]}")
    print(f"Camera forward: {c2w[:3, 2]}")
    print(f"Focal length: {focal}")
    
    # Simple ray test - center pixel should hit sphere
    center_i, center_j = image_size//2, image_size//2
    
    # Convert to normalized device coordinates
    dir_x = (center_i - image_size/2) / focal
    dir_y = -(center_j - image_size/2) / focal  # Negative for image y flip
    dir_z = -1.0
    
    ray_dir_camera = np.array([dir_x, dir_y, dir_z])
    ray_dir_world = ray_dir_camera @ c2w[:3, :3].T
    ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)
    
    print(f"Center pixel ray direction: {ray_dir_world}")
    
    # Ray-sphere intersection
    camera_pos = c2w[:3, 3]
    sphere_center = np.array([0.0, 0.0, 0.0])
    
    oc = camera_pos - sphere_center
    a = np.dot(ray_dir_world, ray_dir_world)
    b = 2.0 * np.dot(oc, ray_dir_world)
    c = np.dot(oc, oc) - sphere_radius ** 2
    
    discriminant = b * b - 4 * a * c
    print(f"Center pixel intersection: discriminant={discriminant:.6f}")
    
    if discriminant >= 0:
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        hit_point = camera_pos + t * ray_dir_world
        print(f"✅ Hit point: {hit_point}")
        print(f"Distance from sphere center: {np.linalg.norm(hit_point):.6f}")
    else:
        print("❌ No intersection for center pixel!")

def main():
    debug_camera_pose()
    debug_ray_sphere_intersection()
    test_simple_render()

if __name__ == "__main__":
    main() 