#!/usr/bin/env python3
"""
Corrected Density Sampling for Confidence Grids

This script fixes the issues identified in the original density sampling approach:
1. Uses proper coordinate bounds [-1, 1] to match confidence grid space
2. Corrects density normalization to match learned confidence distribution
3. Tests different approaches and compares with learned confidence grid
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
    """Load a trained ZipNeRF model from checkpoint with corrected settings."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.name.isdigit():
        exp_path = checkpoint_path.parent.parent
        checkpoint_dir = checkpoint_path.parent
        step = int(checkpoint_path.name)
    else:
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
    gin.clear_config()
    gin.parse_config_file(str(config_gin_path))
    config = configs.Config()
    
    # Override paths for loading
    config.exp_path = str(exp_path)
    config.checkpoint_dir = str(checkpoint_dir)
    
    # CRITICAL: Force disable potential and triplane for "clean" baseline density
    config.use_potential = False
    config.use_triplane = False
    
    logger.info(f"Config loaded (corrected for baseline sampling):")
    logger.info(f"  use_potential: {config.use_potential}")
    logger.info(f"  use_triplane: {config.use_triplane}")
    logger.info(f"  Dataset: {getattr(config, 'dataset_loader', 'unknown')}")
    logger.info(f"  Data dir: {getattr(config, 'data_dir', 'unknown')}")
    
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


def create_3d_grid_corrected(resolution, bound=1.0):
    """
    Create a regular 3D grid of coordinates matching confidence grid space.
    
    Args:
        resolution: Grid resolution (e.g., 128 for 128³)
        bound: Coordinate bounds [-bound, bound] in each dimension (use 1.0 for [-1, 1])
        
    Returns:
        coords: Tensor of shape [resolution³, 3] with 3D coordinates
    """
    # Create 1D coordinate arrays - CRITICAL: use [-1, 1] to match confidence grid
    lin = torch.linspace(-bound, bound, resolution)
    
    # Create 3D meshgrid
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing='ij')
    
    # Flatten and stack to get [N, 3] coordinate array
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    
    return coords


def sample_density_corrected(model, coords, batch_size=65536, std_value=0.01):
    """
    Sample density from model at given coordinates with corrected approach.
    
    Args:
        model: Trained ZipNeRF model
        coords: Tensor of coordinates [N, 3] in [-1, 1] space
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
        for i in tqdm(range(0, coords.shape[0], batch_size), desc="Sampling density (corrected)"):
            batch_coords = coords[i:i + batch_size].to(next(unwrapped_model.parameters()).device)
            
            # Add sample dimension and create stds
            batch_means = batch_coords[:, None, :]  # [batch, 1, 3]
            batch_stds = torch.full((batch_coords.shape[0], 1), std_value, device=batch_coords.device)
            
            # CRITICAL: Use no_warp=False to allow proper coordinate transformation
            # This ensures the model processes coordinates correctly
            raw_density, _, _ = unwrapped_model.nerf_mlp.predict_density(
                batch_means, batch_stds, rand=False, no_warp=False, training_step=None
            )
            
            # Apply softplus activation to get final density
            density = F.softplus(raw_density + unwrapped_model.nerf_mlp.density_bias)
            
            densities.append(density.squeeze().cpu())
    
    return torch.cat(densities, dim=0)


def density_to_confidence_logits_corrected(densities, learned_conf_stats=None, method='adaptive'):
    """
    Convert density values to confidence logits with corrected normalization.
    
    Args:
        densities: Tensor of density values
        learned_conf_stats: Statistics from learned confidence grid for matching
        method: Normalization method ('adaptive', 'learned_match', 'percentile')
        
    Returns:
        logits: Confidence logits
    """
    print(f"Converting densities to confidence using method: {method}")
    print(f"Density stats: min={densities.min():.6f}, max={densities.max():.6f}, mean={densities.mean():.6f}")
    
    if method == 'adaptive':
        # Adaptive approach: use density statistics to determine threshold
        # High density areas should have high confidence
        density_90th = torch.quantile(densities, 0.9)
        density_50th = torch.quantile(densities, 0.5)
        
        # Create confidence: 0.5 at median density, 0.9 at 90th percentile
        # Scale factor to map density range to confidence range
        scale = 0.4 / (density_90th - density_50th + 1e-8)  # 0.4 = 0.9 - 0.5
        
        confidence = torch.sigmoid((densities - density_50th) * scale)
        
    elif method == 'learned_match' and learned_conf_stats is not None:
        # Match the learned confidence distribution
        target_mean = learned_conf_stats['mean']
        target_high_conf_ratio = learned_conf_stats['high_conf_ratio']
        
        # Find density threshold that gives similar high confidence ratio
        sorted_densities, _ = torch.sort(densities, descending=True)
        n_high_conf = int(len(densities) * target_high_conf_ratio)
        density_threshold = sorted_densities[n_high_conf] if n_high_conf < len(densities) else sorted_densities[-1]
        
        # Scale to match target mean confidence
        scale = 5.0  # Adjust this to match the learned distribution
        confidence = torch.sigmoid((densities - density_threshold) * scale)
        
        # Adjust to match target mean
        current_mean = confidence.mean()
        if current_mean > 0:
            confidence = confidence * (target_mean / current_mean)
            confidence = torch.clamp(confidence, 0, 1)
    
    elif method == 'percentile':
        # Use percentile-based normalization
        density_95th = torch.quantile(densities, 0.95)
        density_5th = torch.quantile(densities, 0.05)
        
        # Normalize to [0, 1] based on percentiles
        confidence = (densities - density_5th) / (density_95th - density_5th + 1e-8)
        confidence = torch.clamp(confidence, 0, 1)
    
    else:
        # Default: simple max normalization (original approach)
        sigma_max = densities.max().item()
        confidence = torch.clamp(densities / (0.5 * sigma_max), 0, 1 - 1e-8)
    
    print(f"Confidence stats: min={confidence.min():.6f}, max={confidence.max():.6f}, mean={confidence.mean():.6f}")
    print(f"High confidence (>0.5): {(confidence > 0.5).sum().item()}/{confidence.numel()}")
    
    # Convert to logits
    eps = 1e-8
    logits = torch.log(confidence / (1 - confidence + eps))
    
    # Handle edge cases
    logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -10.0))
    
    return logits


def compare_with_learned_grid(generated_logits, learned_grid_path):
    """Compare generated confidence grid with learned grid."""
    if not Path(learned_grid_path).exists():
        print(f"❌ Learned grid not found: {learned_grid_path}")
        return None
    
    learned_logits = torch.load(learned_grid_path, map_location='cpu')
    
    # Convert both to confidence for comparison
    generated_conf = torch.sigmoid(generated_logits)
    learned_conf = torch.sigmoid(learned_logits)
    
    # Compute correlation
    corr = torch.corrcoef(torch.stack([learned_conf.flatten(), generated_conf.flatten()]))[0, 1]
    mse = F.mse_loss(generated_conf, learned_conf)
    
    print(f"\n📊 Comparison with learned confidence grid:")
    print(f"  Learned - mean: {learned_conf.mean():.6f}, high conf: {(learned_conf > 0.5).sum().item()}")
    print(f"  Generated - mean: {generated_conf.mean():.6f}, high conf: {(generated_conf > 0.5).sum().item()}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  MSE: {mse:.6f}")
    
    return {
        'correlation': corr.item(),
        'mse': mse.item(),
        'learned_stats': {
            'mean': learned_conf.mean().item(),
            'high_conf_ratio': (learned_conf > 0.5).sum().item() / learned_conf.numel()
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Generate corrected confidence grids from trained ZipNeRF model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to checkpoint directory or specific checkpoint")
    parser.add_argument("--output_dir", type=str, default="./confidence_grids_corrected",
                       help="Output directory for confidence grids")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[128],
                       help="Grid resolutions to generate (default: 128)")
    parser.add_argument("--bound", type=float, default=1.0,
                       help="Coordinate bounds [-bound, bound] for sampling (default: 1.0 for [-1,1])")
    parser.add_argument("--batch_size", type=int, default=65536,
                       help="Batch size for density sampling (default: 65536)")
    parser.add_argument("--std_value", type=float, default=0.01,
                       help="Standard deviation for Gaussian samples (default: 0.01)")
    parser.add_argument("--method", type=str, choices=['adaptive', 'learned_match', 'percentile', 'original'],
                       default='learned_match', help="Density to confidence conversion method")
    parser.add_argument("--learned_grid_path", type=str, default="confidence_comparison/learned_confidence_grid.pt",
                       help="Path to learned confidence grid for comparison")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load learned confidence stats if available
    learned_stats = None
    if Path(args.learned_grid_path).exists():
        learned_logits = torch.load(args.learned_grid_path, map_location='cpu')
        learned_conf = torch.sigmoid(learned_logits)
        learned_stats = {
            'mean': learned_conf.mean().item(),
            'high_conf_ratio': (learned_conf > 0.5).sum().item() / learned_conf.numel()
        }
        logger.info(f"Loaded learned confidence stats: mean={learned_stats['mean']:.6f}, high_conf_ratio={learned_stats['high_conf_ratio']:.6f}")
    
    # Load model
    try:
        model, accelerator, step = load_model_from_checkpoint(args.checkpoint_path, logger)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Generate confidence grids for each resolution
    for resolution in args.resolutions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating CORRECTED {resolution}³ confidence grid")
        logger.info(f"{'='*60}")
        
        # Create 3D coordinate grid with corrected bounds
        logger.info(f"Creating {resolution}³ coordinate grid with bounds ±{args.bound}")
        coords = create_3d_grid_corrected(resolution, bound=args.bound)
        logger.info(f"Grid shape: {coords.shape}")
        logger.info(f"Coordinate range: [{coords.min():.3f}, {coords.max():.3f}]")
        
        # Sample densities
        logger.info("Sampling densities from model...")
        densities = sample_density_corrected(
            model, coords, 
            batch_size=args.batch_size, 
            std_value=args.std_value
        )
        logger.info(f"Density shape: {densities.shape}")
        
        # Convert to confidence logits with corrected method
        logger.info("Converting densities to confidence logits (corrected approach)...")
        logits = density_to_confidence_logits_corrected(
            densities, 
            learned_conf_stats=learned_stats, 
            method=args.method
        )
        
        # Reshape to 3D grid
        logits_3d = logits.reshape(resolution, resolution, resolution)
        
        # Save corrected confidence grid
        output_file = output_dir / f"confidence_grid_{resolution}_corrected.pt"
        torch.save(logits_3d, output_file)
        logger.info(f"Saved corrected confidence grid to: {output_file}")
        
        # Compare with learned grid
        comparison = compare_with_learned_grid(logits_3d, args.learned_grid_path)
        
        # Print statistics
        logger.info(f"Corrected logits statistics:")
        logger.info(f"  Shape: {logits_3d.shape}")
        logger.info(f"  Min: {logits_3d.min().item():.6f}")
        logger.info(f"  Max: {logits_3d.max().item():.6f}")
        logger.info(f"  Mean: {logits_3d.mean().item():.6f}")
        logger.info(f"  Std: {logits_3d.std().item():.6f}")
        
        if comparison:
            logger.info(f"  Correlation with learned: {comparison['correlation']:.6f}")
            
            if comparison['correlation'] > 0.5:
                logger.info(f"✅ Good correlation! Corrected approach is working.")
            else:
                logger.info(f"❌ Still poor correlation. May need further adjustments.")
        
        # Save metadata
        metadata = {
            'resolution': resolution,
            'bound': args.bound,
            'method': args.method,
            'std_value': args.std_value,
            'checkpoint_step': step,
            'checkpoint_path': str(args.checkpoint_path),
            'correction_applied': True,
            'coordinate_bounds': f'[-{args.bound}, {args.bound}]',
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
            },
            'comparison': comparison
        }
        
        metadata_file = output_dir / f"confidence_grid_{resolution}_corrected_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to: {metadata_file}")
    
    logger.info(f"\n✅ Corrected confidence grid generation completed!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Generated grids: {args.resolutions}")


if __name__ == "__main__":
    main() 