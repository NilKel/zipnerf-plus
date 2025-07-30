#!/usr/bin/env python3
"""
Debug script to identify PSNR issues during training.
Run this during training to see what's happening with the data and metrics.
"""

import torch
import numpy as np
import gin
from internal import configs
from internal import datasets
from internal import models
from internal import train_utils
from internal.image import mse_to_psnr

def debug_training_step():
    """Debug a single training step to see what's happening with PSNR."""
    
    # Load config
    gin.parse_config_file('configs/miptest.gin')
    config = configs.Config()
    
    print("🔍 PSNR Debug Analysis")
    print("=" * 50)
    
    # Check config issues
    print("📋 Configuration Issues Found:")
    
    issues = []
    if config.data_loss_type == 'mse':
        issues.append("❌ Using MSE loss instead of Charbonnier (should be 'charb')")
    
    if config.batch_size != 2**16:
        issues.append(f"❌ Batch size {config.batch_size} != 65536 (paper default)")
    
    if config.factor != 4:
        issues.append(f"❌ Factor {config.factor} != 4 (should be 4 for 360° scenes)")
    
    if config.hash_decay_mults != 0.1:
        issues.append(f"❌ Hash decay {config.hash_decay_mults} != 0.1 (too high)")
    
    if not hasattr(config, 'NerfMLP') or getattr(config, 'NerfMLP.disable_density_normals', True):
        issues.append("❌ Density normals disabled (hurts quality)")
    
    if len(issues) == 0:
        print("✅ No obvious config issues found")
    else:
        for issue in issues:
            print(f"   {issue}")
    
    print()
    
    # Test PSNR calculation
    print("🧮 PSNR Calculation Test:")
    test_mses = [0.1, 0.01, 0.001, 0.0001]
    for mse in test_mses:
        psnr = mse_to_psnr(mse)
        print(f"   MSE {mse:.4f} -> PSNR {psnr:.2f} dB")
    
    print()
    
    # Expected PSNR ranges
    print("📊 Expected PSNR Ranges:")
    print("   🟢 Excellent: 30-40+ dB")
    print("   🟡 Good: 25-30 dB") 
    print("   🟠 Fair: 20-25 dB")
    print("   🔴 Poor: <20 dB")
    
    print()
    print("💡 Common Causes of Low PSNR:")
    print("   1. Wrong loss function (MSE vs Charbonnier)")
    print("   2. Too small batch size")
    print("   3. Wrong factor (too much downsampling)")
    print("   4. Missing model parameters (normals, grid settings)")
    print("   5. Learning rate too high/low")
    print("   6. Dataset loading issues")
    print("   7. Model architecture problems")
    
    print()
    print("🔧 Quick Fixes to Try:")
    print("   1. Change data_loss_type = 'charb'")
    print("   2. Set batch_size = 65536")
    print("   3. Set factor = 4")
    print("   4. Enable density normals: NerfMLP.disable_density_normals = False")
    print("   5. Use high_quality_360.gin config")
    
    return issues

def analyze_batch_data():
    """Analyze a training batch to see data ranges and issues."""
    print("\n📦 Batch Data Analysis:")
    print("=" * 30)
    
    try:
        # Load dataset
        gin.parse_config_file('configs/miptest.gin')
        config = configs.Config()
        
        dataset = datasets.load_dataset('train', config.data_dir, config)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        
        batch = next(iter(dataloader))
        
        print("Data ranges:")
        print(f"   RGB min/max: {batch['rgb'].min():.4f}/{batch['rgb'].max():.4f}")
        print(f"   RGB mean/std: {batch['rgb'].mean():.4f}/{batch['rgb'].std():.4f}")
        
        if 'lossmult' in batch:
            print(f"   Lossmult: {batch['lossmult'].min():.4f}/{batch['lossmult'].max():.4f}")
        
        print(f"   Batch shape: {batch['rgb'].shape}")
        
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")

if __name__ == "__main__":
    debug_training_step()
    analyze_batch_data() 