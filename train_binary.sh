#!/bin/bash

# Training script for binary occupancy potential field experiments
# Usage: ./train_binary.sh <scene> [comment]

if [ $# -eq 0 ]; then
    echo "Usage: $0 <scene> [comment]"
    echo "Example: $0 lego binary_sanity_check"
    echo "Example: $0 chair binary_vs_smooth"
    exit 1
fi

SCENE="$1"
COMMENT="${2:-binary}"

# Build the command with binary occupancy config
CMD="accelerate launch train.py --gin_configs=configs/potential_binary.gin"
CMD="$CMD --gin_bindings=\"Config.data_dir = '/home/nilkel/Projects/data/nerf_synthetic/$SCENE'\""
CMD="$CMD --gin_bindings=\"Config.debug_confidence_grid_path = 'debug_grids/debug_confidence_grid_128.pt'\""
CMD="$CMD --gin_bindings=\"Config.freeze_debug_confidence = False\""

if [ -n "$COMMENT" ]; then
    CMD="$CMD --gin_bindings=\"Config.comment = '$COMMENT'\""
fi

echo "🔲 Training $SCENE with BINARY OCCUPANCY potential field..."
if [ -n "$COMMENT" ]; then
    echo "   Comment: $COMMENT"
fi
echo "   Config: potential_binary.gin"
echo "   Features: binary_occupancy=True, use_potential=True, use_triplane=True"
echo "   Formulation: binary_occ * (blended_features ⋅ ∇occupancy)"
echo ""
echo "Command: $CMD"
echo

eval $CMD 