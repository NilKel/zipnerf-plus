#!/usr/bin/env python3
"""
Example: How to Load and Use Generated Confidence Grids

This script demonstrates how to load the pre-generated confidence grids
and use them to initialize a ConfidenceField for potential field experiments.

This implements the "sanity check" experiment described in your approach:
using a "ground truth" confidence grid from a well-trained baseline model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from internal.field import ConfidenceField


class PretrainedConfidenceField(ConfidenceField):
    """
    ConfidenceField initialized with pre-generated confidence grid from a trained model.
    
    This is for the "sanity check" experiment where we use a "ground truth" 
    confidence grid to test if the potential field formulation works.
    """
    
    def __init__(self, pretrained_grid_path, device='cuda', freeze=True):
        """
        Initialize with pre-generated confidence grid.
        
        Args:
            pretrained_grid_path: Path to the .pt file containing confidence logits
            device: Device to place the grid on
            freeze: If True, freeze the confidence grid (don't update during training)
        """
        # Load the pretrained grid
        print(f"Loading pretrained confidence grid from: {pretrained_grid_path}")
        pretrained_logits = torch.load(pretrained_grid_path, map_location='cpu')
        
        # Get resolution from the loaded grid
        resolution = pretrained_logits.shape[0]
        assert len(pretrained_logits.shape) == 3, f"Expected 3D grid, got shape {pretrained_logits.shape}"
        assert pretrained_logits.shape[0] == pretrained_logits.shape[1] == pretrained_logits.shape[2], \
            f"Expected cubic grid, got shape {pretrained_logits.shape}"
        
        print(f"Loaded {resolution}³ confidence grid")
        print(f"Logits range: [{pretrained_logits.min():.3f}, {pretrained_logits.max():.3f}]")
        
        # Initialize parent class with the same resolution
        super().__init__(
            resolution=(resolution, resolution, resolution),
            device=device
        )
        
        # Replace the randomly initialized grid with the pretrained one
        with torch.no_grad():
            self.c_grid.data.copy_(pretrained_logits.to(device))
        
        # Optionally freeze the grid for sanity check experiments
        if freeze:
            self.c_grid.requires_grad_(False)
            print("✅ Confidence grid frozen (will not be updated during training)")
        else:
            print("⚠️  Confidence grid is trainable (will be updated during training)")
        
        # Compute initial gradient
        self.compute_gradient()
        
        # Print statistics after loading
        conf = self.get_confidence()
        print(f"Confidence probabilities range: [{conf.min():.6f}, {conf.max():.6f}]")
        print(f"Mean confidence: {conf.mean():.6f}")


def example_usage():
    """Example of how to use the pretrained confidence field."""
    
    # Path to the generated confidence grid
    grid_path = "confidence_grids_lego/confidence_grid_128.pt"
    
    if not Path(grid_path).exists():
        print(f"❌ Confidence grid not found: {grid_path}")
        print("Please run sample_density_to_confidence.py first!")
        return
    
    print("🧪 Example: Using Pretrained Confidence Field")
    print("=" * 60)
    
    # 1. Load the pretrained confidence field (frozen for sanity check)
    confidence_field = PretrainedConfidenceField(
        pretrained_grid_path=grid_path,
        device='cpu',  # Use CPU for this example
        freeze=True    # Freeze for sanity check experiment
    )
    
    # 2. Test querying confidence and gradient at sample points
    print(f"\n📍 Testing confidence field queries...")
    
    # Sample some test points in [-1, 1] coordinate space
    test_points = torch.tensor([
        [0.0, 0.0, 0.0],    # Center of scene
        [0.5, 0.5, 0.5],    # Offset point
        [-0.3, 0.2, -0.1],  # Another test point
        [0.0, 0.0, 0.8],    # Near boundary
    ], dtype=torch.float32)
    
    # Query confidence and gradients
    sampled_conf, sampled_grad = confidence_field.query(test_points)
    
    print(f"Test points: {test_points.shape}")
    print(f"Sampled confidence: {sampled_conf.squeeze().tolist()}")
    print(f"Gradient magnitudes: {sampled_grad.norm(dim=1).tolist()}")
    
    # 3. Example of how this would be used in training
    print(f"\n🏃 Example training integration...")
    
    # Simulate getting features from potential encoder (random for example)
    batch_size, num_levels, level_dim = 1000, 10, 8
    potential_features = torch.randn(batch_size, num_levels, level_dim, 3)
    
    # Simulate sample coordinates
    sample_coords = torch.rand(batch_size, 3) * 2.0 - 1.0  # [-1, 1] range
    
    # Query confidence field (this is what happens in models.py)
    sampled_conf, sampled_grad = confidence_field.query(sample_coords)
    
    # Reshape gradient to match feature dimensions
    sampled_grad_expanded = sampled_grad.unsqueeze(1).expand(-1, num_levels, -1)  # [batch, levels, 3]
    sampled_conf_expanded = sampled_conf.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1]
    
    # Compute volume integral features: V_feat = -C(x) * (G(x) · ∇X(x))
    dot_product = torch.sum(potential_features * sampled_grad_expanded.unsqueeze(-2), dim=-1)  # [batch, levels, level_dim]
    volume_features = -sampled_conf_expanded.squeeze(-1) * dot_product  # [batch, levels, level_dim]
    
    print(f"Potential features shape: {potential_features.shape}")
    print(f"Volume features shape: {volume_features.shape}")
    print(f"Volume features range: [{volume_features.min():.6f}, {volume_features.max():.6f}]")
    
    # 4. Show gradient computation works
    print(f"\n🔀 Testing gradient computation...")
    if confidence_field.c_grid.requires_grad:
        loss = volume_features.mean()
        loss.backward()
        print(f"✅ Gradients computed successfully")
    else:
        print(f"⚠️  Grid is frozen - no gradients computed (as expected for sanity check)")
    
    print(f"\n✅ Example completed successfully!")
    print(f"\nNext steps for sanity check experiment:")
    print(f"1. Use PretrainedConfidenceField(freeze=True) in your model")
    print(f"2. Train with the frozen 'ground truth' confidence grid")
    print(f"3. Compare results to baseline to verify the formulation works")


def compare_resolutions():
    """Compare different resolution confidence grids."""
    
    print(f"\n📏 Comparing different resolution grids...")
    
    grids = {
        128: "confidence_grids_lego/confidence_grid_128.pt",
        256: "confidence_grids_lego/confidence_grid_256.pt"
    }
    
    for resolution, path in grids.items():
        if Path(path).exists():
            grid = torch.load(path, map_location='cpu')
            conf = torch.sigmoid(grid)
            
            print(f"\n{resolution}³ grid:")
            print(f"  File size: {Path(path).stat().st_size / 1024**2:.1f} MB")
            print(f"  Memory usage: {grid.element_size() * grid.numel() / 1024**2:.1f} MB")
            print(f"  Confidence mean: {conf.mean():.6f}")
            print(f"  High confidence voxels (>0.5): {(conf > 0.5).sum().item()}/{conf.numel()}")
        else:
            print(f"{resolution}³ grid not found: {path}")


if __name__ == "__main__":
    print("🎯 Pretrained Confidence Field Example")
    print("=" * 80)
    
    example_usage()
    compare_resolutions()
    
    print(f"\n💡 Usage in your potential field experiments:")
    print(f"   # Replace ConfidenceField initialization with:")
    print(f"   confidence_field = PretrainedConfidenceField(")
    print(f"       pretrained_grid_path='confidence_grids_lego/confidence_grid_128.pt',")
    print(f"       freeze=True  # For sanity check experiment")
    print(f"   )") 