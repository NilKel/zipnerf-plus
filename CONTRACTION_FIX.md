# Spatial Contraction Fix for Confidence Field Gradients

## Problem Identified

You correctly diagnosed a critical issue with the confidence field in your potential encoder setup. The problem was **not** a coordinate system mismatch between the hashgrid and confidence field (both receive the same contracted coordinates), but rather an incorrect assumption in the gradient calculation.

## Root Cause

### What Spatial Contraction Does
Zip-NeRF uses spatial contraction to handle unbounded scenes:
- **Interior (|x| ≤ 1)**: Points mapped nearly linearly  
- **Exterior (|x| > 1)**: Points compressed into shell between radius 1 and 2

This creates **non-uniform spacing** in the contracted coordinate system:
- Near origin: High effective resolution (small world regions per grid cell)
- Far from origin: Low effective resolution (large world regions per grid cell)

### The Gradient Calculation Bug
Your confidence field's finite difference gradient calculation assumed **uniform grid spacing**:

```python
# WRONG: Assumes uniform spacing after contraction
scale_z = (D - 1) / 2.0  # Uniform scaling
scale_y = (H - 1) / 2.0  
scale_x = (W - 1) / 2.0
```

But after spatial contraction, this assumption breaks down completely, leading to:
- **Incorrect gradient magnitudes** in background regions
- **Poor reconstruction quality** far from the scene center
- **Visible distortions** in rendered images

## Solution Implemented

### Contraction-Aware Gradient Scaling
Added a new method `_apply_contraction_aware_scaling()` that:

1. **Computes local scaling factors** based on distance from origin
2. **Accounts for contraction's non-uniform mapping**
3. **Applies position-dependent correction** to gradients

```python
# NEW: Position-dependent scaling
scaling_factor = torch.where(
    coord_norm <= 1.0,
    torch.ones_like(coord_norm),  # Linear region: no extra scaling
    1.0 / (coord_norm ** 2 + 1e-8)  # Contracted region: inverse scaling
)
```

### Configuration Control
- **`Config.contraction_aware_gradients = True`** (default: enabled)
- Can disable for debugging: `Config.contraction_aware_gradients = False`

## Files Modified

1. **`internal/configs.py`**: Added `contraction_aware_gradients` parameter
2. **`internal/field.py`**: 
   - Added `_apply_contraction_aware_scaling()` method
   - Updated `compute_gradient()` to use conditional scaling
   - Updated `__init__()` to accept new parameter
3. **`internal/models.py`**: Pass new parameter to ConfidenceField
4. **`configs/contraction_aware_potential.gin`**: Example configuration

## Expected Results

After this fix, you should see:
- ✅ **Uniform rendering quality** across entire scene
- ✅ **Better background reconstruction** 
- ✅ **Elimination of spatial distortions**
- ✅ **More accurate confidence gradients** in contracted regions

## Usage

### Enable (Default)
```gin
Config.contraction_aware_gradients = True
```

### Disable for Comparison
```gin
Config.contraction_aware_gradients = False
```

### Test Configuration
Use the provided `configs/contraction_aware_potential.gin` as a starting point.

## Technical Details

The fix approximates the Jacobian of the spatial contraction function. For the mip-NeRF 360 contraction `f(x) = x` if `|x|≤1`, else `(2-1/|x|) * x/|x|`:

- **Linear region**: `df/dx ≈ 1` (no correction needed)
- **Contracted region**: `df/dx ≈ 1/|x|²` (significant correction required)

This addresses the fundamental issue that finite differences computed in contracted space need to account for the non-uniform mapping back to world space.

## Validation

To verify the fix:
1. Train with `contraction_aware_gradients = True`
2. Compare background quality to previous results
3. Check for elimination of spatial distortions
4. Monitor gradient magnitudes in confidence field

The background rendering should now match the quality of foreground regions. 