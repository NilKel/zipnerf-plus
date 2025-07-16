#!/bin/bash

# Sanity Check Experiment: ZipNeRF with Learned Confidence Grid
# This is the ultimate test of the potential field formulation

echo "🧪 Starting Sanity Check Experiment"
echo "Using learned confidence grid as debug grid"
echo "If potential field formulation is correct, this should yield excellent results!"

# Activate environment
conda activate zipnerf2

# Run training with debug confidence grid
python train.py \
    --gin_configs=sanity_check_config.gin \
    --gin_bindings="Config.data_dir = '/home/nilkel/Projects/data/nerf_synthetic/lego'" \
    --gin_bindings="Config.exp_path = 'exp/lego_sanity_check_learned_grid_$(date +%m%d_%H%M)'"

echo "✅ Sanity check experiment completed!"
echo "Compare results with baseline to validate potential field formulation"