#!/bin/bash

# Simple training script for potential+triplane models
# Usage: ./train_pt.sh <scene> [comment] [binary_occupancy]

if [ $# -eq 0 ]; then
    echo "Usage: $0 <scene> [comment] [binary_occupancy]"
    echo "Example: $0 lego my_experiment"
    echo "Example: $0 lego binary_test True"
    exit 1
fi

SCENE="$1"
COMMENT="${2:-}"
BINARY_OCC="${3:-False}"

# Build the command
CMD="accelerate launch train.py --gin_configs=configs/potential_triplane.gin"
CMD="$CMD --gin_bindings=\"Config.data_dir = '/home/nilkel/Projects/data/nerf_synthetic/$SCENE'\""
CMD="$CMD --gin_bindings=\"Config.debug_confidence_grid_path = 'debug_grids/debug_confidence_grid_128.pt'\""
CMD="$CMD --gin_bindings=\"Config.freeze_debug_confidence = False\""
CMD="$CMD --gin_bindings=\"Config.binary_occupancy = $BINARY_OCC\""

if [ -n "$COMMENT" ]; then
    CMD="$CMD --gin_bindings=\"Config.comment = '$COMMENT'\""
fi

echo "Training $SCENE with potential+triplane..."
if [ -n "$COMMENT" ]; then
    echo "Comment: $COMMENT"
fi
echo "Binary Occupancy: $BINARY_OCC"
echo "Command: $CMD"
echo

eval $CMD 