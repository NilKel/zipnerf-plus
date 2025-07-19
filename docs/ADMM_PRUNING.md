# ADMM Pruning for Confidence Field Sparsity

This implementation provides ADMM (Alternating Direction Method of Multipliers) pruning for the confidence field in potential encoder setups, based on the HollowNeRF paper. The ADMM pruner eliminates floaters and improves rendering quality by enforcing sparsity constraints on the confidence grid.

## What is ADMM Pruning?

ADMM pruning solves the constrained optimization problem:

```
minimize L(θ) subject to ||sigmoid(confidence_grid)||_1 ≤ C
```

Where:
- `L(θ)` is your rendering loss
- `C` is the sparsity constraint (fraction of voxels that remain active)
- The L1 norm counts the number of "active" voxels after sigmoid activation

The algorithm transforms this constrained problem into an unconstrained one using the augmented Lagrangian:

```
L_augmented = L(θ) + γ * g(x) + (ρ/2) * g(x)²
```

Where:
- `γ` is the dual variable (Lagrange multiplier)
- `g(x) = ||sigmoid(confidence_grid)||_1 - C` is the constraint violation
- `ρ` is the quadratic penalty coefficient

## Benefits

1. **Eliminates Floaters**: Removes artifacts in empty space by forcing low-confidence regions to zero
2. **Improves Quality**: Concentrates learning on important scene regions
3. **Memory Efficiency**: Sparse confidence grids use less memory
4. **Better Convergence**: Prevents the model from learning spurious geometry

## Configuration

Add these parameters to your `.gin` config file:

```gin
# Enable ADMM Pruning
Config.use_admm_pruner = True
Config.admm_sparsity_constraint = 0.04  # Target 4% sparsity
Config.admm_penalty_rho = 1e-4  # Stability parameter
Config.admm_dual_lr = 1e-5  # Dual variable learning rate
Config.admm_start_step = 1000  # When to start pruning
Config.admm_log_every = 100  # Logging frequency

# Also ensure potential encoder is enabled
Config.use_potential = True
```

## Hyperparameter Tuning

### Sparsity Constraint (`admm_sparsity_constraint`)
- **Default**: 0.04 (4% of voxels remain active)
- **Range**: 0.01 - 0.1
- **Lower values**: More aggressive pruning, may remove important details
- **Higher values**: Less pruning, may not eliminate all floaters

### Penalty Coefficient (`admm_penalty_rho`)
- **Default**: 1e-4
- **Range**: 1e-5 - 1e-3
- **Lower values**: Softer constraint enforcement, slower convergence
- **Higher values**: Stronger constraint enforcement, may cause instability

### Dual Learning Rate (`admm_dual_lr`)
- **Default**: 1e-5
- **Range**: 1e-6 - 1e-4
- **Should be smaller than main learning rate**
- **Too high**: Oscillations in dual variable
- **Too low**: Slow constraint enforcement

### Start Step (`admm_start_step`)
- **Default**: 1000
- **Allow initial training before starting pruning**
- **Too early**: May prune important geometry before it's learned
- **Too late**: Floaters may become entrenched

## Monitoring

The implementation provides comprehensive logging:

### Console Output
```
5000/25000: loss=0.02341,psnr=28.45,lr=5.00e-03 | data=0.02200,admm=0.00141,1250 r/s,spars=0.042(t=0.040),γ=2.34e-03
```

- `spars`: Current sparsity fraction
- `t`: Target sparsity
- `γ`: Current dual variable value

### TensorBoard/WandB Metrics
- `admm/current_sparsity_fraction`: Current sparsity as fraction
- `admm/target_sparsity_fraction`: Target sparsity
- `admm/sparsity_error`: Difference from target
- `admm/dual_variable`: Current γ value
- `admm/confidence_mean`: Mean confidence value
- `admm/high_confidence_voxels_XX`: Fraction above threshold

## Expected Behavior

1. **Initial Phase** (steps 0-1000): Normal training without ADMM
2. **Pruning Phase** (steps 1000+): 
   - Sparsity gradually decreases toward target
   - Dual variable γ increases from 0
   - Floaters disappear from renderings
3. **Convergence**: Sparsity stabilizes around target value

## Troubleshooting

### Sparsity not reaching target
- Increase `admm_penalty_rho`
- Increase `admm_dual_lr`
- Check if `admm_start_step` is too late

### Training instability after enabling ADMM
- Decrease `admm_penalty_rho`
- Decrease `admm_dual_lr`
- Increase `admm_start_step`

### Dual variable stuck at zero
- Increase `admm_dual_lr`
- Check if sparsity is already below target
- Ensure `use_potential = True`

### Over-pruning (missing geometry)
- Increase `admm_sparsity_constraint`
- Decrease `admm_penalty_rho`
- Start pruning later (`admm_start_step`)

## Example Configurations

### Conservative Pruning (Safe)
```gin
Config.admm_sparsity_constraint = 0.08  # 8% active
Config.admm_penalty_rho = 5e-5
Config.admm_dual_lr = 5e-6
Config.admm_start_step = 2000
```

### Aggressive Pruning (Maximum sparsity)
```gin
Config.admm_sparsity_constraint = 0.02  # 2% active
Config.admm_penalty_rho = 2e-4
Config.admm_dual_lr = 2e-5
Config.admm_start_step = 1000
```

### Debugging (Frequent logging)
```gin
Config.admm_log_every = 50
Config.print_every = 50
```

## Implementation Details

The ADMM algorithm alternates between two steps each iteration:

1. **Primal Step**: Update model parameters (including confidence grid) using standard gradient descent on the augmented Lagrangian
2. **Dual Step**: Update the dual variable γ using gradient ascent:
   ```python
   γ ← max(0, γ + lr_dual * (current_sparsity - target_sparsity))
   ```

The dual variable acts as a control signal:
- If sparsity is too high → γ increases → more pruning pressure
- If sparsity is too low → γ decreases → less pruning pressure

This automatic adjustment ensures the sparsity constraint is satisfied over time. 