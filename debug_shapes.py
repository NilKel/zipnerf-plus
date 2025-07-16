#!/usr/bin/env python3

import torch
import gin
import sys
import os

# Add current directory to path
sys.path.append('.')

from internal import configs  
from internal.models import Model
from internal.datasets import load_dataset

def debug_tensor_shapes():
    """Debug tensor shapes in the Model forward pass."""
    
    # Load config 
    gin.parse_config_file('configs/sphere_f_encoder_test.gin')
    config = configs.Config()
    
    # Add missing attributes
    if not hasattr(config, 'world_size'):
        config.world_size = 1
    if not hasattr(config, 'global_rank'):
        config.global_rank = 0
    
    print("=== TENSOR SHAPE DEBUG ===")
    print(f"use_potential: {config.use_potential}")
    
    # Create model
    model = Model(config=config)
    model.cuda()
    
    print(f"Model created successfully")
    print(f"num_nerf_samples: {model.num_nerf_samples}")
    print(f"num_prop_samples: {model.num_prop_samples}")
    print(f"num_levels: {model.num_levels}")
    
    try:
        # Load dataset to get real batch
        dataset = load_dataset('train', config.data_dir, config)
        batch = next(iter(dataset))
        
        # Move to device and take a small subset for debugging
        batch_subset = {}
        batch_size = 2  # Very small for debugging
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_subset[k] = v[:batch_size].cuda()
            else:
                batch_subset[k] = v
        
        print(f"\nBatch shapes:")
        for k, v in batch_subset.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
        
        # Try model forward pass
        print(f"\nTesting model forward pass...")
        
        # Add debug hook to capture tensor shapes during forward pass
        def debug_hook(name):
            def hook(module, input, output):
                if isinstance(output, dict):
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor) and k in ['density', 'rgb']:
                            print(f"    {name} output {k}: {v.shape}")
                elif isinstance(output, torch.Tensor):
                    print(f"    {name} output: {output.shape}")
            return hook
        
        # Register hooks
        model.nerf_mlp.register_forward_hook(debug_hook("NerfMLP"))
        if hasattr(model, 'prop_mlp') and model.prop_mlp is not None:
            model.prop_mlp.register_forward_hook(debug_hook("PropMLP"))
        
        renderings, ray_history = model(
            rand=None,
            batch=batch_subset,
            train_frac=0.0,
            compute_extras=False,
            training_step=0
        )
        
        print(f"\n✅ Forward pass successful!")
        print(f"Rendering keys: {list(renderings.keys())}")
        for k, v in renderings.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Let's debug the specific line where it fails
        print(f"\n=== DEBUGGING ERROR LOCATION ===")
        
        # The error happens in compute_alpha_weights, let's debug that
        print(f"Likely issue: density tensor doesn't have expected sample dimension")

if __name__ == "__main__":
    debug_tensor_shapes() 