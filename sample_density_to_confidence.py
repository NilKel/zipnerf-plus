#!/usr/bin/env python3
"""
Sample Density from Trained ZipNeRF Model to Generate Confidence Grids

This script loads a trained ZipNeRF model and samples its density predictions
on regular 3D grids, then converts these densities to confidence logits suitable
for initializing confidence fields in potential-based experiments.

Usage:
    python sample_density_to_confidence.py --checkpoint_path /path/to/checkpoint/025000 --output_dir ./confidence_grids

The script generates:
- confidence_grid_128.pt: 128x128x128 confidence logits
- confidence_grid_256.pt: 256x256x256 confidence logits
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import accelerate
import gin
from tqdm import tqdm

# Import ZipNeRF modules
from internal import configs
from internal import models
from internal import checkpoints
from internal import utils


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path, logger):
    """
    Load a trained ZipNeRF model from checkpoint.
    
    Args:
        checkpoint_path: Path to the specific checkpoint directory (e.g., exp/lego_baseline/checkpoints/025000)
        logger: Logger instance
        
    Returns:
        model: Loaded model
        accelerator: Accelerator instance
        step: Training step number
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Extract experiment path from checkpoint path
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.name.isdigit():
        # Path points to specific checkpoint (e.g., 025000)
        exp_path = checkpoint_path.parent.parent
        checkpoint_dir = checkpoint_path.parent
        step = int(checkpoint_path.name)
    else:
        # Path points to checkpoints directory
        exp_path = checkpoint_path.parent
        checkpoint_dir = checkpoint_path
        step = None
    
    logger.info(f"Experiment path: {exp_path}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Look for config.gin in the experiment directory
    config_gin_path = exp_path / "config.gin"
    if not config_gin_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_gin_path}")
    
    # Set up gin configuration
    # First clear any existing configuration
    gin.clear_config()
    
    # Parse the saved config
    gin.parse_config_file(str(config_gin_path))
    
    # Load the configuration
    config = configs.Config()
    
    # Override paths for loading
    config.exp_path = str(exp_path)
    config.checkpoint_dir = str(checkpoint_dir)
    
    # Disable potential and triplane for density sampling to get "clean" baseline density
    config.use_potential = False
    config.use_triplane = False
    
    logger.info(f"Config loaded - Dataset: {getattr(config, 'dataset_loader', 'unknown')}")
    logger.info(f"Config loaded - Data dir: {getattr(config, 'data_dir', 'unknown')}")
    
    # Setup accelerator
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating model...")
    model = models.Model(config=config)
    model.eval()
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    
    # Load checkpoint
    logger.info("Loading checkpoint...")
    loaded_step = checkpoints.restore_checkpoint(checkpoint_dir, accelerator, logger)
    
    if step is not None and loaded_step != step:
        logger.warning(f"Requested step {step} but loaded step {loaded_step}")
    
    logger.info(f"Model loaded successfully from step {loaded_step}")
    
    return model, accelerator, loaded_step


def create_3d_grid(resolution, bound=2.0):
    """
    Create a regular 3D grid of coordinates in world space.
    
    Args:
        resolution: Grid resolution (e.g., 128 for 128³)
        bound: Coordinate bounds [-bound, bound] in each dimension
        
    Returns:
        coords: Tensor of shape [resolution³, 3] with 3D coordinates
    """
    # Create 1D coordinate arrays
    lin = torch.linspace(-bound, bound, resolution)
    
    # Create 3D meshgrid
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing='ij')
    
    # Flatten and stack to get [N, 3] coordinate array
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    
    return coords


def sample_density_batch(model, coords, batch_size=65536, std_value=0.01):
    """
    Sample density from model at given coordinates in batches.
    
    Args:
        model: Trained ZipNeRF model
        coords: Tensor of coordinates [N, 3]
        batch_size: Batch size for processing
        std_value: Standard deviation for Gaussian samples
        
    Returns:
        densities: Tensor of density values [N]
    """
    model.eval()
    densities = []
    
    # Get the unwrapped model for direct access
    unwrapped_model = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        for i in tqdm(range(0, coords.shape[0], batch_size), desc="Sampling density"):
            batch_coords = coords[i:i + batch_size].to(next(unwrapped_model.parameters()).device)
            
            # Add sample dimension and create stds
            batch_means = batch_coords[:, None, :]  # [batch, 1, 3]
            batch_stds = torch.full((batch_coords.shape[0], 1), std_value, device=batch_coords.device)  # [batch, 1]
            
            # Sample raw density (use no_warp=False to let model handle coordinate transformation)
            raw_density, _, _ = unwrapped_model.nerf_mlp.predict_density(
                batch_means, batch_stds, rand=False, no_warp=False, training_step=None
            )
            
            # Apply softplus activation to get final density
            density = F.softplus(raw_density + unwrapped_model.nerf_mlp.density_bias)
            
            densities.append(density.squeeze().cpu())
    
    return torch.cat(densities, dim=0)


def density_to_confidence_logits(densities, density_scale=0.5, eps=1e-8):
    """
    Convert density values to confidence logits.
    
    Args:
        densities: Tensor of density values
        density_scale: Scale factor for density normalization
        eps: Small epsilon for numerical stability
        
    Returns:
        logits: Confidence logits
    """
    # Find maximum density for normalization
    sigma_max = densities.max().item()
    print(f"Max density: {sigma_max:.6f}")
    print(f"Mean density: {densities.mean().item():.6f}")
    print(f"Min density: {densities.min().item():.6f}")
    
    # Normalize density to pseudo-probability
    # Higher density -> higher confidence
    p = torch.clamp(densities / (density_scale * sigma_max), 0, 1 - eps)
    
    # Convert to logits
    logits = torch.log(p / (1 - p + eps))
    
    # Handle edge cases
    logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -10.0))
    
    return logits


def main():
    parser = argparse.ArgumentParser(description="Sample density from trained ZipNeRF model to generate confidence grids")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to checkpoint directory or specific checkpoint (e.g., exp/lego_baseline/checkpoints/025000)")
    parser.add_argument("--output_dir", type=str, default="./confidence_grids",
                       help="Output directory for confidence grids")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[128, 256],
                       help="Grid resolutions to generate (default: 128 256)")
    parser.add_argument("--bound", type=float, default=1.1,
                       help="Coordinate bounds [-bound, bound] for sampling (default: 1.1)")
    parser.add_argument("--batch_size", type=int, default=65536,
                       help="Batch size for density sampling (default: 65536)")
    parser.add_argument("--density_scale", type=float, default=0.5,
                       help="Scale factor for density normalization (default: 0.5)")
    parser.add_argument("--std_value", type=float, default=0.01,
                       help="Standard deviation for Gaussian samples (default: 0.01)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load model
    try:
        model, accelerator, step = load_model_from_checkpoint(args.checkpoint_path, logger)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Generate confidence grids for each resolution
    for resolution in args.resolutions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating {resolution}³ confidence grid")
        logger.info(f"{'='*60}")
        
        # Create 3D coordinate grid
        logger.info(f"Creating {resolution}³ coordinate grid with bounds ±{args.bound}")
        coords = create_3d_grid(resolution, bound=args.bound)
        logger.info(f"Grid shape: {coords.shape}")
        
        # Sample densities
        logger.info("Sampling densities from model...")
        densities = sample_density_batch(
            model, coords, 
            batch_size=args.batch_size, 
            std_value=args.std_value
        )
        logger.info(f"Density shape: {densities.shape}")
        
        # Convert to confidence logits
        logger.info("Converting densities to confidence logits...")
        logits = density_to_confidence_logits(densities, density_scale=args.density_scale)
        
        # Reshape to 3D grid
        logits_3d = logits.reshape(resolution, resolution, resolution)
        
        # Save confidence grid
        output_file = output_dir / f"confidence_grid_{resolution}.pt"
        torch.save(logits_3d, output_file)
        logger.info(f"Saved confidence grid to: {output_file}")
        
        # Print statistics
        logger.info(f"Logits statistics:")
        logger.info(f"  Shape: {logits_3d.shape}")
        logger.info(f"  Min: {logits_3d.min().item():.6f}")
        logger.info(f"  Max: {logits_3d.max().item():.6f}")
        logger.info(f"  Mean: {logits_3d.mean().item():.6f}")
        logger.info(f"  Std: {logits_3d.std().item():.6f}")
        
        # Also save some metadata
        metadata = {
            'resolution': resolution,
            'bound': args.bound,
            'density_scale': args.density_scale,
            'std_value': args.std_value,
            'checkpoint_step': step,
            'checkpoint_path': str(args.checkpoint_path),
            'logits_stats': {
                'min': logits_3d.min().item(),
                'max': logits_3d.max().item(), 
                'mean': logits_3d.mean().item(),
                'std': logits_3d.std().item()
            },
            'density_stats': {
                'min': densities.min().item(),
                'max': densities.max().item(),
                'mean': densities.mean().item(),
                'std': densities.std().item()
            }
        }
        metadata_file = output_dir / f"confidence_grid_{resolution}_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to: {metadata_file}")
    
    logger.info(f"\n✅ Confidence grid generation completed!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Generated grids: {args.resolutions}")


if __name__ == "__main__":
    main() 