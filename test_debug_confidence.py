#!/usr/bin/env python3
"""
Test the new debug confidence grid functionality in ConfidenceField.

This script tests that the modified ConfidenceField can load pretrained grids
and that the configuration parameters work correctly.
"""

import torch
import torch.nn as nn
from pathlib import Path
from internal.field import ConfidenceField


def test_basic_confidence_field():
    """Test basic ConfidenceField functionality without pretrained grid."""
    print("🧪 Test 1: Basic ConfidenceField (no pretrained grid)")
    print("-" * 60)
    
    # Create basic confidence field
    conf_field = ConfidenceField(
        resolution=(64, 64, 64),
        device='cpu'
    )
    
    print(f"✅ Created ConfidenceField with resolution: {conf_field.resolution}")
    print(f"   Grid shape: {conf_field.c_grid.shape}")
    print(f"   Grid requires_grad: {conf_field.c_grid.requires_grad}")
    
    # Test gradient computation
    conf_field.compute_gradient()
    print(f"   Gradient computed: {conf_field.grad_c_grid is not None}")
    print(f"   Gradient shape: {conf_field.grad_c_grid.shape if conf_field.grad_c_grid is not None else 'None'}")
    
    # Test querying
    test_points = torch.tensor([[0.0, 0.0, 0.0], [0.5, -0.3, 0.1]], dtype=torch.float32)
    conf, grad = conf_field.query(test_points)
    print(f"   Query test: conf shape {conf.shape}, grad shape {grad.shape}")
    

def test_pretrained_confidence_field():
    """Test ConfidenceField with pretrained grid."""
    print("\n🧪 Test 2: ConfidenceField with pretrained grid")
    print("-" * 60)
    
    # Check if pretrained grid exists
    grid_path = "confidence_grids_lego/confidence_grid_128.pt"
    if not Path(grid_path).exists():
        print(f"❌ Pretrained grid not found: {grid_path}")
        print("   Please run sample_density_to_confidence.py first!")
        return
    
    # Test with frozen pretrained grid
    print("\n--- Testing with frozen pretrained grid ---")
    conf_field_frozen = ConfidenceField(
        resolution=(128, 128, 128),  # This will be updated to match pretrained
        device='cpu',
        pretrained_grid_path=grid_path,
        freeze_pretrained=True
    )
    
    print(f"   Final resolution: {conf_field_frozen.resolution}")
    print(f"   Grid requires_grad: {conf_field_frozen.c_grid.requires_grad}")
    
    # Test gradient computation
    conf_field_frozen.compute_gradient()
    print(f"   Gradient computed: {conf_field_frozen.grad_c_grid is not None}")
    
    # Test querying
    test_points = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]], dtype=torch.float32)
    conf, grad = conf_field_frozen.query(test_points)
    print(f"   Query results: conf range [{conf.min():.6f}, {conf.max():.6f}]")
    print(f"   Gradient magnitudes: {grad.norm(dim=1).tolist()}")
    
    # Test with trainable pretrained grid
    print("\n--- Testing with trainable pretrained grid ---")
    conf_field_trainable = ConfidenceField(
        resolution=(128, 128, 128),
        device='cpu',
        pretrained_grid_path=grid_path,
        freeze_pretrained=False
    )
    
    print(f"   Grid requires_grad: {conf_field_trainable.c_grid.requires_grad}")
    
    # Test that gradients can flow through trainable version
    test_loss = conf_field_trainable.get_confidence().mean()
    test_loss.backward()
    
    if conf_field_trainable.c_grid.grad is not None:
        print(f"   ✅ Gradients flow correctly (grad magnitude: {conf_field_trainable.c_grid.grad.norm():.6f})")
    else:
        print(f"   ❌ No gradients computed")


def test_config_integration():
    """Test integration with config system."""
    print("\n🧪 Test 3: Config integration")
    print("-" * 60)
    
    # Mock a config object
    class MockConfig:
        def __init__(self):
            self.use_potential = True
            self.confidence_grid_resolution = (64, 64, 64)
            self.dpcpp_backend = False
            self.debug_confidence_grid_path = None
            self.freeze_debug_confidence = True
    
    # Test with no debug grid
    print("--- Testing config with no debug grid ---")
    config = MockConfig()
    
    conf_field = ConfidenceField(
        resolution=config.confidence_grid_resolution,
        device='cpu',
        pretrained_grid_path=config.debug_confidence_grid_path,
        freeze_pretrained=config.freeze_debug_confidence
    )
    
    print(f"   Created field with resolution: {conf_field.resolution}")
    print(f"   Using pretrained grid: {conf_field.pretrained_grid_path is not None}")
    
    # Test with debug grid
    grid_path = "confidence_grids_lego/confidence_grid_128.pt"
    if Path(grid_path).exists():
        print("\n--- Testing config with debug grid ---")
        config.debug_confidence_grid_path = grid_path
        config.freeze_debug_confidence = True
        
        conf_field_debug = ConfidenceField(
            resolution=config.confidence_grid_resolution,
            device='cpu',
            pretrained_grid_path=config.debug_confidence_grid_path,
            freeze_pretrained=config.freeze_debug_confidence
        )
        
        print(f"   Loaded debug grid: {conf_field_debug.pretrained_grid_path}")
        print(f"   Grid frozen: {not conf_field_debug.c_grid.requires_grad}")
        print(f"   Final resolution: {conf_field_debug.resolution}")
    else:
        print(f"   Skipping debug grid test (file not found: {grid_path})")


def test_error_handling():
    """Test error handling for invalid paths."""
    print("\n🧪 Test 4: Error handling")
    print("-" * 60)
    
    # Test with invalid path
    try:
        conf_field = ConfidenceField(
            resolution=(64, 64, 64),
            device='cpu',
            pretrained_grid_path="nonexistent_grid.pt",
            freeze_pretrained=True
        )
        print("❌ Expected FileNotFoundError but didn't get one")
    except FileNotFoundError as e:
        print(f"✅ Correctly caught FileNotFoundError: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def main():
    print("🎯 Testing Debug Confidence Grid Functionality")
    print("=" * 80)
    
    test_basic_confidence_field()
    test_pretrained_confidence_field()
    test_config_integration()
    test_error_handling()
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    
    print(f"\n💡 Usage examples:")
    print(f"   # Basic usage (no debug grid)")
    print(f"   confidence_field = ConfidenceField(resolution=(128, 128, 128))")
    print(f"   ")
    print(f"   # With debug grid (frozen for sanity check)")
    print(f"   confidence_field = ConfidenceField(")
    print(f"       resolution=(128, 128, 128),")
    print(f"       pretrained_grid_path='confidence_grids_lego/confidence_grid_128.pt',")
    print(f"       freeze_pretrained=True")
    print(f"   )")
    print(f"   ")
    print(f"   # In config file (add to gin bindings):")
    print(f"   --gin_bindings=\"Config.debug_confidence_grid_path = 'confidence_grids_lego/confidence_grid_128.pt'\"")
    print(f"   --gin_bindings=\"Config.freeze_debug_confidence = True\"")


if __name__ == "__main__":
    main() 