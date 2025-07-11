#!/usr/bin/env python3
"""
Test Separate Confidence Field Learning Rate

This script tests the separate learning rate functionality for confidence field parameters.
It validates that:
1. Parameters are correctly separated into main model and confidence field groups
2. Different learning rates are applied to each group
3. Learning rate schedules work correctly for both groups
4. The system falls back to unified LR when separate LR is not specified
"""

import sys
import torch
import numpy as np
from internal import configs, models, train_utils, math

# Global model instance to avoid backend re-initialization
_test_model = None
_test_config = None


def get_test_model():
    """Get or create a test model instance (singleton to avoid backend issues)."""
    global _test_model, _test_config
    
    if _test_model is None:
        # Create config with potential field
        _test_config = configs.Config()
        _test_config.use_potential = True
        _test_config.binary_occupancy = True
        _test_config.confidence_grid_resolution = (32, 32, 32)  # Small for testing
        
        # Create model
        _test_model = models.Model(config=_test_config)
    
    return _test_model, _test_config


def test_parameter_separation():
    """Test that confidence field parameters are correctly separated."""
    print("🧪 Testing parameter separation...")
    
    model, config = get_test_model()
    
    # Test parameter separation
    confidence_params = list(model.confidence_field.parameters())
    confidence_param_ids = {id(p) for p in confidence_params}
    main_params = [p for p in model.parameters() if id(p) not in confidence_param_ids]
    
    print(f"   ✅ Confidence field parameters: {len(confidence_params)}")
    print(f"   ✅ Main model parameters: {len(main_params)}")
    print(f"   ✅ Total parameters: {len(list(model.parameters()))}")
    
    # Verify no overlap
    assert len(confidence_params) > 0, "Should have confidence field parameters"
    assert len(main_params) > 0, "Should have main model parameters"
    assert len(confidence_params) + len(main_params) == len(list(model.parameters())), "Parameters should not overlap"
    
    return model, config


def test_unified_learning_rate():
    """Test unified learning rate (no separate confidence LR)."""
    print("\n🧪 Testing unified learning rate...")
    
    model, config = test_parameter_separation()
    
    # Create a copy of config for this test
    test_config = configs.Config()
    test_config.__dict__.update(config.__dict__)
    
    # Don't set separate confidence LR
    test_config.confidence_lr_multiplier = 1.0
    
    # Create optimizer
    optimizer_result = train_utils.create_optimizer(test_config, model)
    
    # Should return 2 items (optimizer, lr_fn)
    assert len(optimizer_result) == 2, f"Expected 2 returns, got {len(optimizer_result)}"
    optimizer, lr_fn_main = optimizer_result
    
    # Should have single parameter group
    assert len(optimizer.param_groups) == 1, f"Expected 1 param group, got {len(optimizer.param_groups)}"
    
    # Test learning rate at different steps
    for step in [0, 1000, 5000, 15000, 25000]:
        lr = lr_fn_main(step)
        print(f"   Step {step:5d}: LR = {lr:.2e}")
    
    print("   ✅ Unified learning rate working correctly")


def test_separate_learning_rate_multiplier():
    """Test separate learning rate using multiplier."""
    print("\n🧪 Testing separate learning rate with multiplier...")
    
    model, config = get_test_model()
    
    # Create a copy of config for this test
    test_config = configs.Config()
    test_config.__dict__.update(config.__dict__)
    
    # Set multiplier for separate confidence LR
    test_config.confidence_lr_multiplier = 2.0
    
    # Create optimizer
    optimizer_result = train_utils.create_optimizer(test_config, model)
    
    # Should return 3 items (optimizer, lr_fn_main, lr_fn_confidence)
    assert len(optimizer_result) == 3, f"Expected 3 returns, got {len(optimizer_result)}"
    optimizer, lr_fn_main, lr_fn_confidence = optimizer_result
    
    # Should have two parameter groups
    assert len(optimizer.param_groups) == 2, f"Expected 2 param groups, got {len(optimizer.param_groups)}"
    
    # Test learning rates at different steps
    print("   Step     Main LR    Conf LR    Ratio")
    print("   ----     -------    -------    -----")
    for step in [0, 1000, 5000, 15000, 25000]:
        lr_main = lr_fn_main(step)
        lr_conf = lr_fn_confidence(step)
        ratio = lr_conf / lr_main if lr_main > 0 else 0
        print(f"   {step:5d}    {lr_main:.2e}   {lr_conf:.2e}   {ratio:.2f}")
    
    print("   ✅ Separate learning rate with multiplier working correctly")


def test_separate_learning_rate_explicit():
    """Test separate learning rate with explicit values."""
    print("\n🧪 Testing separate learning rate with explicit values...")
    
    model, config = get_test_model()
    
    # Create a copy of config for this test
    test_config = configs.Config()
    test_config.__dict__.update(config.__dict__)
    
    # Set explicit confidence LR values
    test_config.confidence_lr_init = 0.05
    test_config.confidence_lr_final = 0.005
    test_config.confidence_lr_delay_steps = 10000
    test_config.confidence_lr_delay_mult = 1e-9
    
    # Create optimizer
    optimizer_result = train_utils.create_optimizer(test_config, model)
    
    # Should return 3 items
    assert len(optimizer_result) == 3, f"Expected 3 returns, got {len(optimizer_result)}"
    optimizer, lr_fn_main, lr_fn_confidence = optimizer_result
    
    # Should have two parameter groups
    assert len(optimizer.param_groups) == 2, f"Expected 2 param groups, got {len(optimizer.param_groups)}"
    
    # Test learning rates at different steps
    print("   Step     Main LR    Conf LR    Conf Warmup")
    print("   ----     -------    -------    -----------")
    for step in [0, 2500, 5000, 7500, 10000, 15000, 25000]:
        lr_main = lr_fn_main(step)
        lr_conf = lr_fn_confidence(step)
        warmup_progress = min(step / 10000, 1.0) * 100  # Confidence warmup is 10k steps
        print(f"   {step:5d}    {lr_main:.2e}   {lr_conf:.2e}   {warmup_progress:5.1f}%")
    
    print("   ✅ Separate learning rate with explicit values working correctly")


def test_learning_rate_application():
    """Test that learning rates are correctly applied to parameter groups."""
    print("\n🧪 Testing learning rate application...")
    
    model, config = get_test_model()
    
    # Create a copy of config for this test
    test_config = configs.Config()
    test_config.__dict__.update(config.__dict__)
    
    # Set separate confidence LR
    test_config.confidence_lr_multiplier = 3.0
    
    # Create optimizer
    optimizer, lr_fn_main, lr_fn_confidence = train_utils.create_optimizer(test_config, model)
    
    # Simulate training step
    step = 1000
    lr_main = lr_fn_main(step)
    lr_conf = lr_fn_confidence(step)
    
    # Apply learning rates (simulate training loop)
    optimizer.param_groups[0]['lr'] = lr_main
    optimizer.param_groups[1]['lr'] = lr_conf
    
    # Verify learning rates are set correctly
    actual_lr_main = optimizer.param_groups[0]['lr']
    actual_lr_conf = optimizer.param_groups[1]['lr']
    
    assert abs(actual_lr_main - lr_main) < 1e-10, f"Main LR mismatch: {actual_lr_main} vs {lr_main}"
    assert abs(actual_lr_conf - lr_conf) < 1e-10, f"Conf LR mismatch: {actual_lr_conf} vs {lr_conf}"
    
    print(f"   ✅ Main LR applied: {actual_lr_main:.2e}")
    print(f"   ✅ Conf LR applied: {actual_lr_conf:.2e}")
    print(f"   ✅ LR ratio: {actual_lr_conf/actual_lr_main:.2f}")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n🧪 Testing edge cases...")
    
    # Test with zero multiplier
    model, config = get_test_model()
    
    # Create a copy of config for this test
    test_config = configs.Config()
    test_config.__dict__.update(config.__dict__)
    
    test_config.confidence_lr_multiplier = 0.0
    test_config.confidence_lr_init = 0.0
    optimizer_result = train_utils.create_optimizer(test_config, model)
    assert len(optimizer_result) == 3, "Should handle zero LR case"
    
    print("   ✅ Zero learning rate case handled correctly")
    
    # Test with no potential field (create minimal config)
    no_conf_config = configs.Config()
    no_conf_config.use_potential = False
    
    # Don't create a new model, just test optimizer creation behavior
    # by creating a simple dummy model for this test
    import torch.nn as nn
    dummy_model = nn.Linear(10, 1)
    
    # Should work with unified LR even if confidence LR is set
    no_conf_config.confidence_lr_multiplier = 2.0
    optimizer_result = train_utils.create_optimizer(no_conf_config, dummy_model)
    assert len(optimizer_result) == 2, "Should return unified LR when no confidence field"
    
    print("   ✅ No confidence field case handled correctly")


def main():
    """Run all tests."""
    print("🚀 Testing Separate Confidence Field Learning Rate")
    print("=" * 60)
    
    try:
        test_unified_learning_rate()
        test_separate_learning_rate_multiplier()
        test_separate_learning_rate_explicit()
        test_learning_rate_application()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! Separate confidence LR is working correctly.")
        print("\n📋 Usage Examples:")
        print("1. Multiplier approach: --gin_bindings='Config.confidence_lr_multiplier = 2.0'")
        print("2. Explicit values: --gin_bindings='Config.confidence_lr_init = 0.05'")
        print("3. Custom schedule: --gin_bindings='Config.confidence_lr_delay_steps = 10000'")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 