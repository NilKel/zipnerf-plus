#!/usr/bin/env python3
"""
Example configuration script for multi-resolution confidence field.
Shows how to set up and use the new multi-resolution feature.
"""

from internal import configs

def create_multi_resolution_config():
    """Create a configuration with multi-resolution confidence grids."""
    
    # Create base config
    config = configs.Config()
    
    # Example 1: Multi-resolution with MLP combination
    print("📋 Example 1: Multi-resolution with MLP combination")
    print("-" * 50)
    
    config.confidence_grid_resolutions = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
    config.confidence_combination_method = "mlp"
    config.confidence_mlp_hidden_dim = 32
    config.confidence_mlp_num_layers = 2
    
    print(f"✅ Multi-resolution grids: {config.confidence_grid_resolutions}")
    print(f"   Combination method: {config.confidence_combination_method}")
    print(f"   MLP hidden dim: {config.confidence_mlp_hidden_dim}")
    print(f"   MLP layers: {config.confidence_mlp_num_layers}")
    
    return config

if __name__ == "__main__":
    print("🚀 Multi-Resolution Confidence Field Configuration Guide")
    print("=" * 60)
    
    create_multi_resolution_config()
    print("\n🎉 Configuration guide complete!") 