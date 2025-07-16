#!/usr/bin/env python3
"""
Debug Density Sampling Issues

This script investigates why our density sampling produces poor confidence grids
compared to learned confidence grids. It examines coordinate systems, bounds,
and sampling approaches.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import accelerate
import gin
from internal import configs
from internal import models
from internal import checkpoints
from internal import coord


def load_baseline_model(checkpoint_path):
    """Load baseline model for density sampling."""
    print(f"🔧 Loading baseline model from: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    exp_path = checkpoint_path.parent.parent
    checkpoint_dir = checkpoint_path.parent
    
    # Load and modify config
    config_gin_path = exp_path / "config.gin"
    gin.clear_config()
    gin.parse_config_file(str(config_gin_path))
    config = configs.Config()
    
    # Force disable potential and triplane for baseline sampling
    config.use_potential = False
    config.use_triplane = False
    
    print(f"Baseline model config (modified):")
    print(f"  use_potential: {config.use_potential}")
    print(f"  use_triplane: {config.use_triplane}")
    
    # Load model
    accelerator = accelerate.Accelerator()
    model = models.Model(config=config)
    model.eval()
    model = accelerator.prepare(model)
    
    step = checkpoints.restore_checkpoint(checkpoint_dir, accelerator)
    print(f"✅ Loaded baseline model from step {step}")
    
    return model, config, accelerator


def load_potential_model(checkpoint_path):
    """Load potential model to understand coordinate system."""
    print(f"🔧 Loading potential model from: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    exp_path = checkpoint_path.parent.parent
    checkpoint_dir = checkpoint_path.parent
    
    config_gin_path = exp_path / "config.gin"
    gin.clear_config()
    gin.parse_config_file(str(config_gin_path))
    config = configs.Config()
    
    accelerator = accelerate.Accelerator()
    model = models.Model(config=config)
    model.eval()
    model = accelerator.prepare(model)
    
    step = checkpoints.restore_checkpoint(checkpoint_dir, accelerator)
    return model, config, accelerator


def test_coordinate_systems():
    """Test different coordinate systems and bounds."""
    print(f"\n🧭 Testing coordinate systems...")
    
    # Test different coordinate ranges
    bounds_to_test = [
        (-1.0, 1.0, "[-1, 1] (confidence grid space)"),
        (-1.1, 1.1, "[-1.1, 1.1] (our current approach)"),
        (-2.0, 2.0, "[-2, 2] (contracted space)"),
        (-0.5, 0.5, "[-0.5, 0.5] (smaller space)"),
    ]
    
    for min_bound, max_bound, description in bounds_to_test:
        # Create small test grid
        res = 16
        lin = torch.linspace(min_bound, max_bound, res)
        X, Y, Z = torch.meshgrid(lin, lin, lin, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        
        print(f"  {description}: {coords.shape[0]} points")
        print(f"    Range: [{coords.min():.3f}, {coords.max():.3f}]")
        print(f"    Center point: {coords[coords.shape[0]//2]}")


def sample_density_with_different_approaches(baseline_model, config, accelerator):
    """Test different density sampling approaches."""
    print(f"\n🎯 Testing different density sampling approaches...")
    
    unwrapped_model = accelerator.unwrap_model(baseline_model)
    device = next(unwrapped_model.parameters()).device
    
    # Create test coordinates
    res = 32
    approaches = [
        (-1.1, 1.1, "Original approach ([-1.1, 1.1])"),
        (-1.0, 1.0, "Confidence grid space ([-1, 1])"),
        (-2.0, 2.0, "Contracted space ([-2, 2])"),
    ]
    
    results = {}
    
    for min_bound, max_bound, name in approaches:
        print(f"\n  Testing {name}...")
        
        # Create coordinate grid
        lin = torch.linspace(min_bound, max_bound, res)
        X, Y, Z = torch.meshgrid(lin, lin, lin, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).to(device)
        
        # Sample density
        batch_size = 1024
        densities = []
        
        with torch.no_grad():
            for i in range(0, coords.shape[0], batch_size):
                batch_coords = coords[i:i + batch_size]
                batch_means = batch_coords[:, None, :]  # [batch, 1, 3]
                batch_stds = torch.full((batch_coords.shape[0], 1), 0.01, device=device)
                
                # Sample with and without warping
                raw_density_no_warp, _, _ = unwrapped_model.nerf_mlp.predict_density(
                    batch_means, batch_stds, rand=False, no_warp=True, training_step=None
                )
                raw_density_warp, _, _ = unwrapped_model.nerf_mlp.predict_density(
                    batch_means, batch_stds, rand=False, no_warp=False, training_step=None
                )
                
                density_no_warp = F.softplus(raw_density_no_warp + unwrapped_model.nerf_mlp.density_bias)
                density_warp = F.softplus(raw_density_warp + unwrapped_model.nerf_mlp.density_bias)
                
                densities.append({
                    'no_warp': density_no_warp.squeeze().cpu(),
                    'warp': density_warp.squeeze().cpu()
                })
        
        # Combine results
        densities_no_warp = torch.cat([d['no_warp'] for d in densities])
        densities_warp = torch.cat([d['warp'] for d in densities])
        
        print(f"    No warp - Range: [{densities_no_warp.min():.6f}, {densities_no_warp.max():.6f}], Mean: {densities_no_warp.mean():.6f}")
        print(f"    With warp - Range: [{densities_warp.min():.6f}, {densities_warp.max():.6f}], Mean: {densities_warp.mean():.6f}")
        
        # Count high density points
        high_density_no_warp = (densities_no_warp > 1.0).sum().item()
        high_density_warp = (densities_warp > 1.0).sum().item()
        
        print(f"    High density points (>1.0): no_warp={high_density_no_warp}, warp={high_density_warp}")
        
        results[name] = {
            'coords': coords.cpu(),
            'densities_no_warp': densities_no_warp,
            'densities_warp': densities_warp,
            'bounds': (min_bound, max_bound)
        }
    
    return results


def compare_with_learned_confidence(sampling_results, learned_grid_path):
    """Compare sampling results with learned confidence grid."""
    print(f"\n📊 Comparing with learned confidence grid...")
    
    if not Path(learned_grid_path).exists():
        print(f"❌ Learned grid not found: {learned_grid_path}")
        return
    
    learned_logits = torch.load(learned_grid_path, map_location='cpu')
    learned_conf = torch.sigmoid(learned_logits)
    
    print(f"Learned confidence stats:")
    print(f"  Range: [{learned_conf.min():.6f}, {learned_conf.max():.6f}]")
    print(f"  Mean: {learned_conf.mean():.6f}")
    print(f"  High conf (>0.5): {(learned_conf > 0.5).sum().item()}/{learned_conf.numel()}")
    
    # Test which coordinate system matches best
    confidence_coords = create_confidence_grid_coordinates(learned_conf.shape, (-1.0, 1.0))
    
    print(f"\nTesting correlations with different sampling approaches:")
    
    for name, result in sampling_results.items():
        coords = result['coords']
        densities = result['densities_warp']  # Use warped version
        bounds = result['bounds']
        
        # Convert density to confidence using same method as our script
        sigma_max = densities.max().item()
        p = torch.clamp(densities / (0.5 * sigma_max), 0, 1 - 1e-8)
        conf_from_density = p  # Don't convert to logits, just use confidence
        
        # Resize sampling result to match learned grid for comparison
        if coords.shape[0] != learned_conf.numel():
            # Reshape and interpolate
            res = int(round(coords.shape[0] ** (1/3)))
            conf_reshaped = conf_from_density.reshape(res, res, res)
            
            conf_resized = F.interpolate(
                conf_reshaped.unsqueeze(0).unsqueeze(0),
                size=learned_conf.shape,
                mode='trilinear',
                align_corners=True
            ).squeeze(0).squeeze(0)
        else:
            conf_resized = conf_from_density.reshape(learned_conf.shape)
        
        # Compute correlation
        corr = torch.corrcoef(torch.stack([learned_conf.flatten(), conf_resized.flatten()]))[0, 1]
        
        print(f"  {name}: correlation = {corr:.6f}")


def create_confidence_grid_coordinates(shape, bounds):
    """Create coordinates for confidence grid sampling."""
    min_bound, max_bound = bounds
    lin_x = torch.linspace(min_bound, max_bound, shape[0])
    lin_y = torch.linspace(min_bound, max_bound, shape[1])
    lin_z = torch.linspace(min_bound, max_bound, shape[2])
    
    X, Y, Z = torch.meshgrid(lin_x, lin_y, lin_z, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    
    return coords


def analyze_coordinate_contraction():
    """Analyze how coordinate contraction affects sampling."""
    print(f"\n🔄 Analyzing coordinate contraction...")
    
    # Test points in different coordinate spaces
    test_points = torch.tensor([
        [0.0, 0.0, 0.0],     # Center
        [0.5, 0.5, 0.5],     # Moderate distance
        [1.0, 1.0, 1.0],     # Edge of unit cube
        [1.5, 1.5, 1.5],     # Outside unit cube
        [2.0, 2.0, 2.0],     # Far outside
    ])
    
    print(f"Testing coordinate contraction on sample points:")
    for i, point in enumerate(test_points):
        # Apply contraction as done in models
        contracted_point, _ = coord.track_linearize('contract', point.unsqueeze(0), torch.zeros_like(point.unsqueeze(0)))
        contracted_point = contracted_point.squeeze(0)
        
        # Then normalize to [-1, 1] as done in predict_density
        bound = 2
        normalized_point = contracted_point / bound
        
        print(f"  Point {i}: {point.numpy()} -> contracted: {contracted_point.numpy()} -> normalized: {normalized_point.numpy()}")


def main():
    print("🐛 Debug Density Sampling Issues")
    print("=" * 60)
    
    # Paths
    baseline_checkpoint = "/home/nilkel/Projects/zipnerf-pytorch/exp/lego_baseline_25000_0704_2320/checkpoints/025000"
    potential_checkpoint = "/home/nilkel/Projects/zipnerf-pytorch/exp/lego_potential_25000_0710_1932_defaultconf_nogate/checkpoints/025000"
    learned_grid_path = "confidence_comparison/learned_confidence_grid.pt"
    
    try:
        # Test coordinate systems
        test_coordinate_systems()
        
        # Analyze coordinate contraction
        analyze_coordinate_contraction()
        
        # Load baseline model and test sampling
        baseline_model, baseline_config, accelerator = load_baseline_model(baseline_checkpoint)
        sampling_results = sample_density_with_different_approaches(baseline_model, baseline_config, accelerator)
        
        # Compare with learned confidence
        compare_with_learned_confidence(sampling_results, learned_grid_path)
        
        print(f"\n🔍 Key Findings:")
        print(f"1. Our original approach uses bounds [-1.1, 1.1]")
        print(f"2. Confidence grid lives in [-1, 1] space")
        print(f"3. Scene contraction maps world space to [-2, 2], then normalizes to [-1, 1]")
        print(f"4. We need to match the coordinate system used by the confidence field")
        
        print(f"\n💡 Recommended fixes:")
        print(f"1. Use [-1, 1] bounds instead of [-1.1, 1.1]")
        print(f"2. Ensure proper coordinate contraction")
        print(f"3. Match the density normalization used during training")
        print(f"4. Consider using the same std_value as during training")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 