#!/usr/bin/env python3
"""
Training Script for Analytical Oracle Experiment

This script trains the analytical model that uses perfect oracle functions
to test the mathematical formulation V_feat = -G⋅∇O.

The key insight being tested:
- F(p) = 1 (constant function to integrate)
- G(p) = p/3 (vector potential with div(G) = F)  
- O(p) = 1 if on sphere shell, 0 elsewhere
- ∇O(p) = unit normal if on sphere, 0 elsewhere
- V_feat = -G⋅∇O should allow MLP to reconstruct the sphere

Expected outcome: Very fast convergence to high PSNR since all components are perfect.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import gin
from internal import configs, datasets, utils, train_utils
from analytical_models import AnalyticalModel
from analytical_oracles import test_analytical_oracles


def setup_config():
    """Set up configuration for analytical experiment."""
    config = configs.Config()
    
    # Backend setup
    from extensions import Backend
    Backend.set_backend('cuda')
    
    # Dataset configuration  
    config.exp_name = 'analytical_oracle_experiment'
    config.dataset_loader = 'blender'
    config.data_dir = '../data/nerf_synthetic/sphere_analytical_simple'
    config.near = 2.0
    config.far = 6.0
    config.factor = 4  # Downsample to 200x200 for faster training
    
    # Analytical experiment settings
    config.analytical_experiment = True
    config.sphere_radius = 1.0
    config.sphere_center = [0.0, 0.0, 0.0]
    config.sphere_epsilon = 0.05
    
    # Disable normal NeRF components
    config.use_potential = False
    config.use_triplane = False
    config.binary_occupancy = False
    config.analytical_gradient = False
    
    # Training settings
    config.max_steps = 3000
    config.lr_init = 0.02
    config.lr_final = 0.002
    config.lr_delay_steps = 500
    config.batch_size = 4096
    config.world_size = 1  # Single GPU
    config.global_rank = 0  # Main process
    
    # Dataset loading settings
    config.patch_size = 1
    config.batching = 'all_images'
    config.use_tiffs = False
    config.compute_disp_metrics = False
    config.compute_normal_metrics = False
    config.num_border_pixels_to_mask = 0
    config.apply_bayer_mask = False
    config.render_path = False
    config.compute_visibility = False
    
    # Loss settings
    config.data_loss_type = 'mse'
    config.data_loss_mult = 1.0
    config.confidence_reg_mult = 0.0
    config.hash_decay_mults = []
    
    # Logging
    config.print_every = 50
    config.train_render_every = 500
    config.checkpoint_every = 1000
    config.eval_only_once = False
    config.eval_quantize_metrics = True
    
    # Wandb
    config.use_wandb = False  # Disable for this experiment
    
    # Visualization
    config.vis_num_rays = 1024
    
    return config


def compute_metrics(rendering, batch):
    """Compute evaluation metrics."""
    metrics = {}
    
    # Get predicted and target images
    rgb_pred = rendering['rgb']
    rgb_target = batch['rgb']
    
    # PSNR
    mse = F.mse_loss(rgb_pred, rgb_target)
    psnr = -10 * torch.log10(mse)
    metrics['psnr'] = psnr.item()
    
    # SSIM (simple approximation)
    rgb_pred_flat = rgb_pred.view(-1, 3)
    rgb_target_flat = rgb_target.view(-1, 3)
    
    # Pearson correlation as SSIM proxy
    pred_mean = rgb_pred_flat.mean(dim=0)
    target_mean = rgb_target_flat.mean(dim=0)
    
    pred_centered = rgb_pred_flat - pred_mean
    target_centered = rgb_target_flat - target_mean
    
    correlation = (pred_centered * target_centered).sum(dim=0) / (
        torch.sqrt((pred_centered ** 2).sum(dim=0) * (target_centered ** 2).sum(dim=0)) + 1e-8)
    
    metrics['ssim_approx'] = correlation.mean().item()
    
    return metrics


def train_analytical_experiment():
    """Main training function for analytical experiment."""
    
    print("🔮 Starting Analytical Oracle Experiment")
    print("="*60)
    
    # Test analytical oracles first
    print("🧪 Testing analytical oracles...")
    test_analytical_oracles()
    print("✅ Oracle tests passed!\n")
    
    # Setup
    config = setup_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📋 Experiment Configuration:")
    print(f"   Device: {device}")
    print(f"   Data dir: {config.data_dir}")
    print(f"   Max steps: {config.max_steps}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.lr_init} → {config.lr_final}")
    print(f"   Sphere radius: {config.sphere_radius}")
    print(f"   Shell thickness: {config.sphere_epsilon}")
    print()
    
    # Check if dataset exists
    if not Path(config.data_dir).exists():
        print(f"❌ Dataset not found at {config.data_dir}")
        print("Please run: python generate_analytical_sphere_dataset.py")
        return
    
    # Load dataset
    print("📊 Loading dataset...")
    dataset = datasets.load_dataset('train', config.data_dir, config)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=True, num_workers=0, pin_memory=True
    )
    
    # Create analytical model
    print("🔮 Creating analytical model...")
    model = AnalyticalModel(config=config)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    oracle_params = 0  # Oracle components have frozen dummy parameters
    for module in [model.nerf_mlp.encoder, model.nerf_mlp.confidence_field]:
        if hasattr(module, 'dummy_param'):
            oracle_params += module.dummy_param.numel()
    trainable_params = total_params - oracle_params
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Oracle parameters (frozen): {oracle_params:,}")
    print()
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=(config.lr_final / config.lr_init) ** (1 / config.max_steps)
    )
    
    # Training loop
    print("🚀 Starting training...")
    print("="*60)
    
    model.train()
    step = 0
    start_time = time.time()
    
    for epoch in range(1000):  # Large number, will break when max_steps reached
        for batch in dataloader:
            if step >= config.max_steps:
                break
                
            # Move batch to device (handle None values)
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            
            # Forward pass
            rand = torch.Generator(device=device).manual_seed(step)
            renderings, ray_history = model(
                rand=rand,
                batch=batch,
                train_frac=step / config.max_steps,
                compute_extras=(step % config.train_render_every == 0)
            )
            
            # Compute loss
            rendering = renderings[-1]  # Use final rendering
            rgb_pred = rendering['rgb']
            rgb_target = batch['rgb']
            
            loss = F.mse_loss(rgb_pred, rgb_target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Logging
            if step % config.print_every == 0:
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                
                metrics = compute_metrics(rendering, batch)
                
                print(f"Step {step:5d} | "
                      f"Loss: {loss.item():.6f} | "
                      f"PSNR: {metrics['psnr']:.2f} | "
                      f"SSIM: {metrics['ssim_approx']:.3f} | "
                      f"LR: {lr:.6f} | "
                      f"Time: {elapsed:.1f}s")
                
                # Check for expected analytical behavior
                if step > 500:  # After some initial training
                    if metrics['psnr'] > 30:
                        print(f"🎉 Excellent convergence detected! PSNR = {metrics['psnr']:.2f}")
                    elif metrics['psnr'] < 20:
                        print(f"⚠️  Lower than expected PSNR: {metrics['psnr']:.2f}")
                        print("    This might indicate an issue with the analytical formulation")
            
            # Render visualization
            if step % config.train_render_every == 0 and step > 0:
                print(f"📸 Rendering visualization at step {step}...")
                
                # Save a sample rendering
                rgb_vis = rgb_pred[0].detach().cpu().numpy()  # First image in batch
                rgb_target_vis = rgb_target[0].detach().cpu().numpy()
                
                # Ensure output directory exists
                vis_dir = Path('analytical_experiment_vis')
                vis_dir.mkdir(exist_ok=True)
                
                # Save images
                utils.save_img_u8(rgb_vis, vis_dir / f'step_{step:05d}_pred.png')
                utils.save_img_u8(rgb_target_vis, vis_dir / f'step_{step:05d}_target.png')
                
                print(f"   Saved to {vis_dir}/")
            
            step += 1
            
        if step >= config.max_steps:
            break
    
    # Final evaluation
    print("\n" + "="*60)
    print("🏁 Training Complete!")
    
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.1f}s")
    print(f"Steps per second: {step / total_time:.2f}")
    
    # Final metrics
    with torch.no_grad():
        final_metrics = compute_metrics(rendering, batch)
        print(f"\n📊 Final Metrics:")
        print(f"   PSNR: {final_metrics['psnr']:.2f} dB")
        print(f"   SSIM (approx): {final_metrics['ssim_approx']:.3f}")
        
        # Analytical validation
        expected_v_feat = model.nerf_mlp.oracles.expected_v_feat_on_sphere()
        print(f"\n🔍 Analytical Validation:")
        print(f"   Expected V_feat on sphere: {expected_v_feat:.4f}")
        
        # Check if convergence meets expectations
        if final_metrics['psnr'] > 35:
            print(f"✅ Excellent! PSNR > 35 indicates the formulation works perfectly")
        elif final_metrics['psnr'] > 25:
            print(f"✅ Good! PSNR > 25 indicates the formulation works well")
        else:
            print(f"⚠️  PSNR < 25 suggests potential issues with the formulation")
    
    print("\n🔮 Analytical Oracle Experiment Complete!")
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Train analytical oracle experiment")
    parser.add_argument("--config", help="Path to gin config file (optional)")
    parser.add_argument("--data_dir", help="Override data directory")
    parser.add_argument("--max_steps", type=int, help="Override max training steps")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        gin.parse_config_file(args.config)
    
    # Run experiment
    try:
        metrics = train_analytical_experiment()
        
        # Print summary
        print(f"\n🎯 Experiment Summary:")
        print(f"   Final PSNR: {metrics['psnr']:.2f} dB")
        print(f"   Final SSIM: {metrics['ssim_approx']:.3f}")
        
        if metrics['psnr'] > 30:
            print(f"✅ SUCCESS: High PSNR confirms V_feat = -G⋅∇O works!")
        else:
            print(f"⚠️  WARNING: Lower PSNR suggests potential formulation issues")
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main() 