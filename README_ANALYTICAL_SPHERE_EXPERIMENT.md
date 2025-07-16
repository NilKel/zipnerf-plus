# Analytical Sphere Experiment for Vector Potential Testing

This document describes the setup and implementation of the analytical sphere experiment to test the refined vector potential formulation using perfect ground truth data.

## 🎯 Experiment Overview

This experiment implements the refined analytical approach that bypasses learned parameters and numerical approximations by using:

1. **Direct Binary Occupancy**: Store final occupancy values (0/1) instead of logits requiring sigmoid
2. **Perfect Analytical Gradients**: Use exact sphere normals instead of finite difference approximations
3. **Pure Analytical Test**: Remove all sources of numerical error to isolate the V_feat formulation

## 📁 Generated Assets

### Sphere Dataset (`../data/nerf_synthetic/sphere_analytical/`)
- **Format**: Standard NeRF synthetic dataset structure
- **Images**: 800x800 PNG files with sphere renderings
- **Splits**: 100 train, 100 val, 200 test images
- **Camera Setup**: Spherical camera positions at distance 4.0 looking at origin
- **Sphere**: Radius 1.0, centered at origin with Lambertian shading

### Analytical Grids (`sphere_analytical_grids/`)
- **Occupancy Grid**: `analytical_occupancy_grid_128.pt` (128³ binary values)
- **Gradient Grid**: `analytical_gradient_grid_128.pt` (3×128³ perfect normals)
- **Metadata**: Complete parameter specifications and statistics

## 🔧 Key Features

### 1. Direct Value Storage
```python
# Instead of learned logits + sigmoid:
# occupancy = torch.sigmoid(learned_logits)

# We directly store analytical values:
occupancy = (distance_to_sphere <= 0).float()  # Perfect binary {0,1}
```

### 2. Perfect Gradients
```python
# Instead of conv3d finite differences:
# gradient = conv3d(occupancy_field, stencil_kernels)

# We store exact analytical normals:
gradient = (point - sphere_center) / sphere_radius  # Perfect unit normals
```

### 3. Clean V_feat Calculation
```python
# The volume integral feature computation remains:
V_feat = X(p) * (G(p) · ∇X(p))

# Where:
# - X(p) = analytical binary occupancy (gating term)
# - G(p) = learned potential features (N, D, 3)
# - ∇X(p) = analytical gradient/normal (N, 3)
```

## 🔄 Implementation Plan

### Phase 1: Analytical Grid Integration ✅
- [x] Generate sphere dataset in NeRF synthetic format
- [x] Generate analytical occupancy and gradient grids
- [x] Verify coordinate system consistency
- [x] Test grid properties and visualization

### Phase 2: Model Integration (Next Steps)

#### Step 1: Create Analytical Confidence Field
```python
class AnalyticalConfidenceField(nn.Module):
    def __init__(self, occupancy_grid_path, gradient_grid_path):
        super().__init__()
        # Load pre-computed grids
        occupancy_data = torch.load(occupancy_grid_path, weights_only=False)
        gradient_data = torch.load(gradient_grid_path, weights_only=False)
        
        # Store as non-trainable buffers
        self.register_buffer('occupancy_grid', occupancy_data['grid'])
        self.register_buffer('gradient_grid', gradient_data['grid'])
        
    def query(self, coords):
        # Interpolate occupancy and gradient at query points
        sampled_occupancy = F.grid_sample(self.occupancy_grid[None, None], ...)
        sampled_gradient = F.grid_sample(self.gradient_grid[None], ...)
        return sampled_occupancy, sampled_gradient
    
    def compute_gradient(self):
        # No-op since gradients are pre-computed
        pass
```

#### Step 2: Modify Model Configuration
```python
# Add to configs/analytical_sphere.gin
Config.use_potential = True
Config.analytical_test = True
Config.analytical_occupancy_path = "sphere_analytical_grids/analytical_occupancy_grid_128.pt"
Config.analytical_gradient_path = "sphere_analytical_grids/analytical_gradient_grid_128.pt"
```

#### Step 3: Update Model Initialization
```python
# In internal/models.py
if self.config.analytical_test:
    self.confidence_field = AnalyticalConfidenceField(
        occupancy_grid_path=self.config.analytical_occupancy_path,
        gradient_grid_path=self.config.analytical_gradient_path
    )
else:
    # Normal learned confidence field
    self.confidence_field = ConfidenceField(...)
```

### Phase 3: Training and Validation

#### Expected Results
If the V_feat formulation is correct:
1. **Perfect Geometry**: Analytical grids provide exact sphere geometry
2. **Clean Features**: Only the MLP learns, potential encoder gets perfect gradients  
3. **Superior Performance**: Should exceed baseline since geometry is "oracle"

#### Training Command
```bash
python train_potential.py analytical_sphere_test \
    --scene sphere_analytical \
    --model_type analytical \
    --max_steps 10000
```

## 📊 Verification Steps

### 1. Grid Quality Verification ✅
```bash
python visualize_sphere_dataset.py \
    --dataset_dir ../data/nerf_synthetic/sphere_analytical \
    --grids_dir sphere_analytical_grids
```

**Results**: 
- ✅ Binary occupancy: values ∈ {0, 1}
- ✅ Perfect gradients: magnitude = 1.0 everywhere  
- ✅ Coordinate consistency: dataset ↔ grids match

### 2. Dataset Quality Verification ✅
- ✅ 400 total images (100+100+200)
- ✅ 800×800 resolution matching NeRF synthetic
- ✅ Valid JSON transforms with proper camera poses
- ✅ Sphere visible and properly lit in all images

### 3. Analytical Properties ✅
- ✅ Gradients point radially outward from sphere center
- ✅ Occupancy boundary matches analytical sphere surface
- ✅ No numerical artifacts from finite differences

## 🧪 Scientific Value

This experiment provides:

1. **Formulation Validation**: Tests whether `V_feat = X(p) * (G(p) · ∇X(p))` is fundamentally sound
2. **Numerical Isolation**: Removes conv3d approximation errors 
3. **Learning Isolation**: Only MLP parameters are optimized
4. **Clean Comparison**: Perfect analytical baseline for performance evaluation

## 📁 File Structure

```
zipnerf-pytorch/
├── generate_sphere_dataset.py           # Dataset generation
├── generate_analytical_grids.py         # Grid generation  
├── visualize_sphere_dataset.py          # Visualization tools
├── README_ANALYTICAL_SPHERE_EXPERIMENT.md  # This document
│
├── ../data/nerf_synthetic/sphere_analytical/
│   ├── train/r_*.png                   # Training images
│   ├── val/r_*.png                     # Validation images
│   ├── test/r_*.png                    # Test images
│   ├── transforms_train.json           # Camera poses
│   ├── transforms_val.json
│   ├── transforms_test.json
│   └── metadata.json                   # Dataset metadata
│
└── sphere_analytical_grids/
    ├── analytical_occupancy_grid_128.pt  # Binary occupancy
    ├── analytical_gradient_grid_128.pt   # Perfect gradients
    └── analytical_grids_metadata.json    # Grid metadata
```

## 🚀 Next Steps

1. **Implement AnalyticalConfidenceField** class
2. **Add analytical mode** to model configuration
3. **Create training configuration** for analytical experiment
4. **Run training** and compare with baseline
5. **Analyze results** to validate V_feat formulation

This analytical experiment will provide definitive validation of whether your volume integral feature formulation is correct, independent of learning dynamics or numerical approximations.

---

**Generated**: 2024-12-14  
**Author**: Analytical Sphere Experiment Setup  
**Purpose**: Vector Potential Field Validation 