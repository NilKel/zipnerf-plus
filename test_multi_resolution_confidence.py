#!/usr/bin/env python3
"""
Test script for multi-resolution confidence field functionality.
Demonstrates how to use the new multi-resolution feature with different grid sizes.
"""

import torch
import torch.nn as nn
from internal.field import ConfidenceField
from internal import configs
import numpy as np

def test_multi_resolution_confidence():
    """Test multi-resolution confidence field functionality."""
    print("🧪 Testing Multi-Resolution Confidence Field")
    print("=" * 50)
    
    # Define multi-resolution grids: 16³, 32³, 64³
    resolutions = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    
    # Test both combination methods
    for combination_method in ["mlp", "sum"]:
        print(f"\n📋 Testing combination method: {combination_method}")
        print("-" * 30)
        
        # Create multi-resolution confidence field
        conf_field = ConfidenceField(
            resolution=(128, 128, 128),  # Legacy parameter (ignored in multi-res mode)
            resolutions=resolutions,
            combination_method=combination_method,
            mlp_hidden_dim=32,
            mlp_num_layers=2,
            device=device,
            init_val=-1.0,
            init_rand_mag=0.5
        )
        
        print(f"✅ Created multi-resolution confidence field")
        print(f"   Number of grids: {conf_field.num_grids}")
        print(f"   Resolutions: {conf_field.resolutions}")
        print(f"   Combination method: {conf_field.combination_method}")
        
        # Compute gradients for all grids
        print(f"🔄 Computing gradients for all grids...")
        conf_field.compute_gradient()
        print(f"✅ Gradients computed")
        
        # Test querying at random points
        num_test_points = 1000
        test_points = torch.rand(num_test_points, 3, device=device) * 2 - 1  # [-1, 1] range
        
        print(f"🎯 Querying {num_test_points} test points...")
        sampled_conf, sampled_grad = conf_field.query(test_points)
        
        print(f"✅ Query successful")
        print(f"   Confidence shape: {sampled_conf.shape}")
        print(f"   Gradient shape: {sampled_grad.shape}")
        print(f"   Confidence range: [{sampled_conf.min():.4f}, {sampled_conf.max():.4f}]")
        print(f"   Gradient magnitude range: [{torch.norm(sampled_grad, dim=1).min():.4f}, {torch.norm(sampled_grad, dim=1).max():.4f}]")
        
        # Test individual grid access
        print(f"🔍 Testing individual grid access...")
        for i in range(conf_field.num_grids):
            conf_i = conf_field.get_confidence(grid_index=i)
            print(f"   Grid {i} ({resolutions[i][0]}³): confidence range [{conf_i.min():.4f}, {conf_i.max():.4f}]")
        
        # Test getting all grids at once
        all_confs = conf_field.get_confidence()
        print(f"   All grids returned as list: {len(all_confs)} grids")
        
        del conf_field  # Clean up
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def test_backward_compatibility():
    """Test that single-resolution mode still works (backward compatibility)."""
    print("\n🔄 Testing Backward Compatibility (Single Resolution)")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create single-resolution confidence field (legacy mode)
    conf_field = ConfidenceField(
        resolution=(64, 64, 64),
        device=device,
        init_val=-1.0,
        init_rand_mag=0.5
    )
    
    print(f"✅ Created single-resolution confidence field")
    print(f"   Resolution: {conf_field.resolution}")
    print(f"   Multi-resolution mode: {conf_field.use_multi_resolution}")
    
    # Compute gradients
    conf_field.compute_gradient()
    
    # Test querying
    num_test_points = 100
    test_points = torch.rand(num_test_points, 3, device=device) * 2 - 1
    
    sampled_conf, sampled_grad = conf_field.query(test_points)
    
    print(f"✅ Single-resolution query successful")
    print(f"   Confidence shape: {sampled_conf.shape}")
    print(f"   Gradient shape: {sampled_grad.shape}")
    
    del conf_field
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def test_config_integration():
    """Test integration with the config system."""
    print("\n⚙️  Testing Config Integration")
    print("=" * 50)
    
    # Create a test config
    config = configs.Config()
    
    # Set multi-resolution parameters
    config.confidence_grid_resolutions = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
    config.confidence_combination_method = "mlp"
    config.confidence_mlp_hidden_dim = 64
    config.confidence_mlp_num_layers = 3
    
    print(f"✅ Config created with multi-resolution settings:")
    print(f"   Resolutions: {config.confidence_grid_resolutions}")
    print(f"   Combination method: {config.confidence_combination_method}")
    print(f"   MLP hidden dim: {config.confidence_mlp_hidden_dim}")
    print(f"   MLP num layers: {config.confidence_mlp_num_layers}")

def benchmark_performance():
    """Simple benchmark comparing single vs multi-resolution."""
    print("\n⏱️  Performance Benchmark")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_test_points = 5000
    test_points = torch.rand(num_test_points, 3, device=device) * 2 - 1
    
    # Single resolution benchmark
    conf_field_single = ConfidenceField(
        resolution=(64, 64, 64),
        device=device
    )
    conf_field_single.compute_gradient()
    
    # Warm up
    _ = conf_field_single.query(test_points[:100])
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    import time
    start_time = time.time()
    for _ in range(10):
        _ = conf_field_single.query(test_points)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    single_time = (time.time() - start_time) / 10
    
    # Multi-resolution benchmark
    conf_field_multi = ConfidenceField(
        resolutions=[(32, 32, 32), (64, 64, 64)],
        combination_method="sum",  # Faster than MLP
        device=device
    )
    conf_field_multi.compute_gradient()
    
    # Warm up
    _ = conf_field_multi.query(test_points[:100])
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(10):
        _ = conf_field_multi.query(test_points)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    multi_time = (time.time() - start_time) / 10
    
    print(f"📊 Performance Results ({num_test_points} points):")
    print(f"   Single resolution (64³): {single_time*1000:.2f} ms")
    print(f"   Multi-resolution (32³+64³): {multi_time*1000:.2f} ms")
    print(f"   Slowdown factor: {multi_time/single_time:.2f}x")
    
    del conf_field_single, conf_field_multi
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    print("🚀 Multi-Resolution Confidence Field Test Suite")
    print("=" * 60)
    
    try:
        test_multi_resolution_confidence()
        test_backward_compatibility()
        test_config_integration()
        
        if torch.cuda.is_available():
            benchmark_performance()
        else:
            print("\n⚠️  Skipping performance benchmark (CUDA not available)")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 