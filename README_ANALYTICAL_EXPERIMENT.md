# Analytical Oracle Experiment

This experiment implements a controlled test of the refined vector potential formulation using perfect analytical functions. It tests whether the feature **V_feat = -G⋅∇O** can be used by an MLP to reconstruct a scene when all other components are mathematically perfect.

## 🎯 Experiment Overview

### Mathematical Setup
- **F(p) = 1**: Constant scalar field to integrate
- **G(p) = p/3**: Vector potential satisfying div(G) = F perfectly  
- **O(p) = 1** if abs(||p|| - r) ≤ ε else **0**: Hollow sphere occupancy
- **∇O(p) = normalize(p)** if on sphere surface else **(0,0,0)**: Perfect normals

### Test Hypothesis
Can an MLP learn to reconstruct a sphere using only:
1. **V_feat = -G⋅∇O** (for density prediction)
2. **∇O** (surface normals for color prediction)

### Expected Outcome
Since all components are analytically perfect, we expect:
- **Very fast convergence** (< 1000 steps)
- **High PSNR** (> 35 dB) 
- **Perfect reconstruction** of the uniform-colored sphere

## 📁 Files Created

### Core Components
- `analytical_oracles.py` - Perfect analytical functions (F, G, O, ∇O)
- `analytical_models.py` - MLP classes using oracle components  
- `train_analytical_experiment.py` - Training script for the experiment
- `generate_analytical_sphere_dataset.py` - Simple sphere dataset (no shading)
- `configs/analytical_experiment.gin` - Configuration file

### Generated Assets
- `../data/nerf_synthetic/sphere_analytical_simple/` - Dataset with uniform sphere
- `analytical_experiment_vis/` - Training visualizations

## 🚀 How to Run

### 1. Generate Dataset (if needed)
```bash
python generate_analytical_sphere_dataset.py \
    --output_dir ../data/nerf_synthetic/sphere_analytical_simple \
    --n_train 50 --n_val 25 --n_test 50 \
    --image_size 800 --visualize
```

### 2. Test Oracle Functions
```bash
python analytical_oracles.py
```

### 3. Run the Experiment  
```bash
python train_analytical_experiment.py
```

### 4. With Custom Config (optional)
```bash
python train_analytical_experiment.py --config configs/analytical_experiment.gin
```

## 📊 Expected Results

### Training Progress
```
🔮 Starting Analytical Oracle Experiment
🧪 Testing analytical oracles...
✅ Oracle tests passed!

📋 Experiment Configuration:
   Device: cuda
   Max steps: 3000
   Sphere radius: 1.0
   Expected V_feat on sphere: -0.3333

🚀 Starting training...
Step     0 | Loss: 0.156432 | PSNR: 8.06 | SSIM: 0.234 | LR: 0.020000
Step    50 | Loss: 0.023156 | PSNR: 16.35 | SSIM: 0.667 | LR: 0.019121
Step   100 | Loss: 0.003245 | PSNR: 24.89 | SSIM: 0.834 | LR: 0.018274
Step   150 | Loss: 0.001456 | PSNR: 28.36 | SSIM: 0.901 | LR: 0.017458
🎉 Excellent convergence detected! PSNR = 31.24
Step   200 | Loss: 0.000876 | PSNR: 30.57 | SSIM: 0.942 | LR: 0.016673
```

### Final Validation
```
🏁 Training Complete!
📊 Final Metrics:
   PSNR: 38.45 dB
   SSIM (approx): 0.967

🔍 Analytical Validation:
   Expected V_feat on sphere: -0.3333

✅ Excellent! PSNR > 35 indicates the formulation works perfectly
🔮 Analytical Oracle Experiment Complete!
```

## 🔍 What This Proves

### If Successful (PSNR > 35)
- ✅ The mathematical formulation **V_feat = -G⋅∇O** is sound
- ✅ MLPs can effectively use this feature for scene reconstruction
- ✅ The pipeline architecture handles multi-resolution/multi-dimensional features correctly
- ✅ Any failures in full end-to-end training are due to optimization challenges, not fundamental flaws

### If Unsuccessful (PSNR < 25)
- ⚠️ Potential issues with the feature formulation
- ⚠️ Problems with coordinate transformation or feature concatenation
- ⚠️ Architecture incompatibilities

## 🔧 Key Implementation Details

### Analytical Override Strategy
Instead of learning hash grid features, we directly compute:
```python
# Override grid encoder
def forward(self, points, bound=1):
    g_values = self.oracles.analytical_g_query(points, self.level_dim)
    # Expand to multi-resolution (all levels identical)
    return g_values.flatten(-3, -1)

# Override confidence field  
def query(self, points):
    occupancy = self.oracles.analytical_o_query(points)
    gradients = self.oracles.analytical_grad_o_query(points)
    return occupancy, gradients
```

### Feature Computation
```python
# In MLP forward pass:
v_feat = self.oracles.compute_v_feat(points, feature_dim)  # -G⋅∇O
grad_o = self.oracles.analytical_grad_o_query(points)     # ∇O  
mlp_input = torch.cat([v_feat, grad_o], dim=-1)          # Combined features
```

## 📈 Debugging Guide

### Common Issues
1. **Import errors**: Ensure all files are in project root
2. **Dataset missing**: Run dataset generation script first  
3. **Low PSNR**: Check oracle functions with `python analytical_oracles.py`
4. **GPU memory**: Reduce batch_size in config if needed

### Validation Checks
- Oracle tests pass: `✅ All tests passed!`
- V_feat value: Should be exactly `-sphere_radius/3 = -0.3333`
- Convergence speed: Should reach PSNR > 20 within 200 steps

## 🎓 Educational Value

This experiment demonstrates:
- How to create **controlled scientific tests** in deep learning
- The power of **analytical ground truth** for validating mathematical formulations  
- Proper **separation of concerns** between mathematical correctness and optimization challenges
- How to design experiments that **definitively prove or disprove** core hypotheses

If successful, this gives us complete confidence to focus optimization efforts on the full end-to-end model! 