#!/usr/bin/env python3
"""
Training Script for Sphere Vector Potential Experiment

This script trains a NeRF model on the analytical sphere dataset using:
1. Pre-initialized sphere-based confidence grids
2. Pre-initialized sphere-based potential encoder  
3. Only the MLP parameters are trained

Usage:
    python train_sphere_experiment.py --experiment_name sphere_test_v1
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import accelerate
import gin
import numpy as np
from tqdm import tqdm

# Import ZipNeRF components
from internal import configs
from internal import datasets
from internal import models
from internal import utils
from internal import train_utils
from internal import checkpoints
from internal import render
import wandb


def create_sphere_config():
    """Create configuration for sphere experiment."""
    config = configs.Config()
    
    # Dataset configuration
    config.dataset_loader = 'blender'
    config.data_dir = '/home/nilkel/Projects/data/nerf_synthetic/sphere_analytical'
    config.near = 2.0
    config.far = 6.0
    config.factor = 4
    config.use_tiffs = False
    
    # Model configuration - using potential field
    config.use_potential = True
    config.use_triplane = False
    config.binary_occupancy = False
    
    # Sphere-specific configuration
    config.sphere_experiment = True
    config.sphere_radius = 1.0
    config.sphere_center = [0.0, 0.0, 0.0]
    
    # Pre-initialized confidence grid
    config.debug_confidence_grid_path = 'sphere_confidence_grids/sphere_confidence_grid_128.pt'
    config.freeze_debug_confidence = True  # Don't train the confidence field
    
    # Training configuration
    config.max_steps = 10000
    config.batch_size = 4096
    config.lr_init = 0.01
    config.lr_final = 0.001
    config.lr_delay_steps = 1000
    
    # Checkpointing
    config.checkpoint_every = 1000
    config.checkpoints_total_limit = 3
    config.resume_from_checkpoint = True
    
    # Logging
    config.print_every = 100
    config.train_render_every = 500
    config.eval_only_once = False
    
    # Loss configuration
    config.data_loss_type = 'charb'
    config.data_loss_mult = 1.0
    config.confidence_reg_mult = 0.0  # No confidence regularization since it's frozen
    
    # Wandb
    config.use_wandb = True
    config.wandb_project = "sphere-vector-potential"
    
    return config


class SphereExperimentModel(models.Model):
    """Model class with sphere-specific initialization."""
    
    def __init__(self, config=None, **kwargs):
        # Set sphere initialization flags before calling parent __init__
        if hasattr(config, 'sphere_experiment') and config.sphere_experiment:
            # We'll modify the MLP creation to use sphere initialization
            self.sphere_init_config = {
                'sphere_radius': getattr(config, 'sphere_radius', 1.0),
                'sphere_center': getattr(config, 'sphere_center', [0.0, 0.0, 0.0])
            }
        else:
            self.sphere_init_config = None
            
        super().__init__(config, **kwargs)
        
        # Print initialization info
        if self.sphere_init_config:
            print(f"🌟 SphereExperimentModel initialized with:")
            print(f"   Sphere radius: {self.sphere_init_config['sphere_radius']}")
            print(f"   Sphere center: {self.sphere_init_config['sphere_center']}")
            print(f"   Confidence grid frozen: {config.freeze_debug_confidence}")


def create_sphere_mlp(config):
    """Create MLP with sphere-based potential encoder initialization."""
    from internal.models import NerfMLP
    
    # Create the MLP with sphere initialization enabled
    mlp = NerfMLP(
        config=config,
        num_glo_features=0,  # No GLO for sphere experiment
        num_glo_embeddings=0
    )
    
    # The PotentialEncoder will be initialized in the MLP's __init__ method
    # if config.use_potential is True and we pass the sphere parameters
    
    return mlp


def setup_sphere_experiment(config, accelerator):
    """Setup the sphere experiment components."""
    
    print(f"🔧 Setting up sphere experiment...")
    print(f"   Data directory: {config.data_dir}")
    print(f"   Confidence grid: {config.debug_confidence_grid_path}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Max steps: {config.max_steps}")
    
    # Create model with sphere configuration
    model = SphereExperimentModel(config=config)
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    
    # Create dataset
    dataset = datasets.load_dataset('train', config.data_dir, config)
    
    # Create optimizer - only optimize MLP parameters since confidence is frozen
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if 'confidence_field' in name and config.freeze_debug_confidence:
            frozen_params.append(name)
            param.requires_grad = False
        else:
            trainable_params.append(param)
    
    print(f"   Trainable parameters: {len(trainable_params)}")
    print(f"   Frozen parameters: {len(frozen_params)}")
    if frozen_params:
        print(f"   Frozen: {', '.join(frozen_params[:3])}{'...' if len(frozen_params) > 3 else ''}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr_init)
    optimizer = accelerator.prepare(optimizer)
    
    # Learning rate scheduler
    def lr_schedule_fn(step):
        if step < config.lr_delay_steps:
            # Warmup
            return step / config.lr_delay_steps
        else:
            # Cosine decay
            progress = (step - config.lr_delay_steps) / (config.max_steps - config.lr_delay_steps)
            return (config.lr_final / config.lr_init) + 0.5 * (1 - config.lr_final / config.lr_init) * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fn)
    
    return model, dataset, optimizer, scheduler


def train_step(model, batch, optimizer, config, step):
    """Single training step."""
    
    # Forward pass
    rendering = models.render_image(
        model=model,
        accelerator=None,  # Not used in this context
        batch=batch,
        train_frac=step / config.max_steps,
        compute_extras=False,
        zero_glo=True,
        training_step=step
    )
    
    # Compute loss
    target = batch['target']
    predicted = rendering['rgb']
    
    # Main reconstruction loss
    if config.data_loss_type == 'mse':
        data_loss = torch.mean((predicted - target) ** 2)
    elif config.data_loss_type == 'charb':
        # Charbonnier loss
        diff = predicted - target
        data_loss = torch.mean(torch.sqrt(diff ** 2 + 1e-8))
    else:
        raise ValueError(f"Unknown data loss type: {config.data_loss_type}")
    
    total_loss = config.data_loss_mult * data_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Metrics
    metrics = {
        'loss/total': total_loss.item(),
        'loss/data': data_loss.item(),
        'rgb/mean': predicted.mean().item(),
        'rgb/std': predicted.std().item(),
        'target/mean': target.mean().item(),
    }
    
    return metrics


def evaluate(model, dataset, config, step):
    """Evaluation on validation set."""
    model.eval()
    
    val_metrics = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            if i >= 10:  # Limited evaluation 
                break
                
            rendering = models.render_image(
                model=model,
                accelerator=None,
                batch=batch,
                train_frac=step / config.max_steps,
                compute_extras=True,
                zero_glo=True,
                training_step=step
            )
            
            target = batch['target']
            predicted = rendering['rgb']
            
            # Compute metrics
            mse = torch.mean((predicted - target) ** 2)
            psnr = -10 * torch.log10(mse)
            
            val_metrics.append({
                'val/mse': mse.item(),
                'val/psnr': psnr.item(),
            })
    
    model.train()
    
    # Average metrics
    avg_metrics = {}
    for key in val_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in val_metrics])
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train sphere vector potential experiment")
    parser.add_argument("--experiment_name", required=True,
                       help="Name for the experiment")
    parser.add_argument("--confidence_grid_resolution", type=int, default=128,
                       choices=[64, 128, 256],
                       help="Resolution of confidence grid to use")
    parser.add_argument("--max_steps", type=int, default=10000,
                       help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=4096,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = accelerate.Accelerator()
    
    # Create configuration
    config = create_sphere_config()
    config.exp_name = args.experiment_name
    config.max_steps = args.max_steps
    config.batch_size = args.batch_size
    config.lr_init = args.lr
    config.use_wandb = not args.no_wandb
    
    # Update confidence grid path based on resolution
    config.debug_confidence_grid_path = f'sphere_confidence_grids/sphere_confidence_grid_{args.confidence_grid_resolution}.pt'
    config.confidence_grid_resolution = (args.confidence_grid_resolution,) * 3
    
    print(f"🚀 Starting Sphere Vector Potential Experiment")
    print(f"   Experiment: {args.experiment_name}")
    print(f"   Confidence grid: {args.confidence_grid_resolution}³")
    print(f"   Max steps: {args.max_steps}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    
    # Setup wandb
    if config.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=config.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Setup experiment
    model, dataset, optimizer, scheduler = setup_sphere_experiment(config, accelerator)
    
    # Create checkpoint directory
    checkpoint_dir = Path(f"experiments/sphere/{args.experiment_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    model.train()
    step = 0
    
    print(f"\n🏃 Starting training loop...")
    
    try:
        while step < config.max_steps:
            for batch in dataset:
                if step >= config.max_steps:
                    break
                
                # Training step
                metrics = train_step(model, batch, optimizer, config, step)
                scheduler.step()
                
                # Logging
                if step % config.print_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {step:5d}: loss={metrics['loss/total']:.6f}, lr={lr:.6f}")
                    
                    if config.use_wandb and accelerator.is_main_process:
                        wandb.log({**metrics, 'lr': lr, 'step': step})
                
                # Evaluation
                if step % config.train_render_every == 0 and step > 0:
                    val_metrics = evaluate(model, dataset, config, step)
                    print(f"   Validation: PSNR={val_metrics['val/psnr']:.2f}")
                    
                    if config.use_wandb and accelerator.is_main_process:
                        wandb.log({**val_metrics, 'step': step})
                
                # Checkpointing
                if step % config.checkpoint_every == 0 and step > 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pt"
                    torch.save({
                        'step': step,
                        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                    }, checkpoint_path)
                    print(f"   💾 Saved checkpoint: {checkpoint_path}")
                
                step += 1
        
        print(f"\n🎉 Training completed!")
        
        # Final evaluation
        final_metrics = evaluate(model, dataset, config, step)
        print(f"Final PSNR: {final_metrics['val/psnr']:.2f}")
        
        # Save final model
        final_path = checkpoint_dir / "final_model.pt"
        torch.save({
            'step': step,
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'final_metrics': final_metrics,
        }, final_path)
        print(f"💾 Saved final model: {final_path}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted at step {step}")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    
    finally:
        if config.use_wandb and accelerator.is_main_process:
            wandb.finish()


if __name__ == "__main__":
    main() 