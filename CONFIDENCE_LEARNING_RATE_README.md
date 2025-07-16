# Separate Confidence Field Learning Rate

This feature allows you to use different learning rates for the confidence field parameters versus the main model parameters (hash grids, MLPs, etc.). This can be useful for fine-tuning the optimization dynamics of the potential field formulation.

## 🎯 Use Cases

- **Higher confidence LR**: When confidence field needs to adapt faster than geometry
- **Lower confidence LR**: When confidence field should change more conservatively  
- **Different warmup**: When confidence field needs different initialization dynamics
- **Ablation studies**: To understand the impact of confidence field learning rate on results

## 🚀 Quick Start

### Method 1: Multiplier (Easiest)

Use `confidence_lr_multiplier` to scale the main learning rate:

```bash
# 2x higher confidence LR
accelerate launch train.py \
    --gin_configs=configs/potential_binary.gin \
    --gin_bindings="Config.confidence_lr_multiplier = 2.0"

# 0.5x lower confidence LR  
accelerate launch train.py \
    --gin_configs=configs/potential_binary.gin \
    --gin_bindings="Config.confidence_lr_multiplier = 0.5"
```

### Method 2: Explicit Values (Full Control)

Set specific learning rate values for confidence field:

```bash
accelerate launch train.py \
    --gin_configs=configs/potential_binary.gin \
    --gin_bindings="Config.confidence_lr_init = 0.05" \
    --gin_bindings="Config.confidence_lr_final = 0.005"
```

## 📋 Configuration Parameters

### Basic Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `confidence_lr_multiplier` | Multiplier applied to main LR for confidence field | `1.0` |
| `confidence_lr_init` | Initial learning rate for confidence field | `None` (uses main LR) |
| `confidence_lr_final` | Final learning rate for confidence field | `None` (uses main LR) |

### Advanced Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `confidence_lr_delay_steps` | Warmup steps for confidence field | `None` (uses main warmup) |
| `confidence_lr_delay_mult` | Warmup multiplier for confidence field | `None` (uses main warmup) |

## 🔧 Configuration Examples

### Example 1: High Confidence Learning Rate

```gin
include 'configs/potential_binary.gin'

# 5x higher learning rate for confidence field
Config.confidence_lr_init = 0.05     # vs 0.01 for main model
Config.confidence_lr_final = 0.005   # vs 0.001 for main model

Config.exp_name = "lego_high_conf_lr"
```

### Example 2: Conservative Confidence Learning

```gin
include 'configs/potential_binary.gin'

# Lower and more conservative confidence field learning
Config.confidence_lr_init = 0.002    # 5x lower than main
Config.confidence_lr_final = 0.0002  # 5x lower than main
Config.confidence_lr_delay_steps = 10000  # Longer warmup
Config.confidence_lr_delay_mult = 1e-9    # More conservative start

Config.exp_name = "lego_conservative_conf_lr"
```

### Example 3: Using Multiplier

```gin
include 'configs/potential_binary.gin'

# Simple 3x multiplier approach
Config.confidence_lr_multiplier = 3.0

Config.exp_name = "lego_conf_mult_3x"
```

## 📊 Monitoring Learning Rates

Both learning rates are logged to TensorBoard and Weights & Biases:

- **TensorBoard**: `train_learning_rate` and `train_learning_rate_confidence`
- **Weights & Biases**: `training/learning_rate` and `training/learning_rate_confidence`

## 🧪 Testing

Run the test script to validate functionality:

```bash
python test_separate_confidence_lr.py
```

Expected output:
```
🚀 Testing Separate Confidence Field Learning Rate
============================================================
🧪 Testing parameter separation...
   ✅ Confidence field parameters: 1
   ✅ Main model parameters: 2847
   ✅ Total parameters: 2848

🧪 Testing unified learning rate...
   Step     0: LR = 1.00e-10
   Step  1000: LR = 1.93e-03
   Step  5000: LR = 1.00e-02
   Step 15000: LR = 5.49e-03
   Step 25000: LR = 1.00e-03
   ✅ Unified learning rate working correctly
...
✅ All tests passed! Separate confidence LR is working correctly.
```

## 🎛️ Pre-made Configurations

Ready-to-use configurations are provided:

| Config File | Description | Confidence LR |
|-------------|-------------|---------------|
| `configs/potential_binary_high_conf_lr.gin` | High confidence LR | 5x higher |
| `configs/potential_binary_low_conf_lr.gin` | Low confidence LR | 5x lower |
| `configs/potential_binary_conf_multiplier.gin` | Multiplier approach | 2x higher |

### Usage:

```bash
# High confidence LR experiment
accelerate launch train.py \
    --gin_configs=configs/potential_binary_high_conf_lr.gin \
    --gin_bindings="Config.data_dir = '/path/to/lego'"

# Low confidence LR experiment  
accelerate launch train.py \
    --gin_configs=configs/potential_binary_low_conf_lr.gin \
    --gin_bindings="Config.data_dir = '/path/to/lego'"
```

## 🔄 Learning Rate Schedules

Both main and confidence learning rates follow the same schedule structure:

```python
# Warmup phase (0 to delay_steps)
delay_rate = delay_mult + (1 - delay_mult) * sin(0.5 * π * step / delay_steps)

# Main phase (delay_steps to max_steps)  
lr = delay_rate * log_lerp(step / max_steps, lr_init, lr_final)
```

Example schedules:

| Step | Main LR | Conf LR (2x) | Conf LR (0.5x) |
|------|---------|--------------|----------------|
| 0 | 1e-10 | 2e-10 | 5e-11 |
| 2500 | 0.007 | 0.014 | 0.0035 |
| 5000 | 0.01 | 0.02 | 0.005 |
| 15000 | 0.0055 | 0.011 | 0.00275 |
| 25000 | 0.001 | 0.002 | 0.0005 |

## 🧠 Best Practices

### When to Use Higher Confidence LR

- Confidence field is lagging behind geometry learning
- Want faster initial scene structure discovery
- Testing if confidence needs more aggressive optimization

### When to Use Lower Confidence LR  

- Confidence field is converging too quickly and overfitting
- Want more stable, conservative confidence evolution
- Geometry and confidence need to co-evolve slowly

### Experimental Guidelines

1. **Start with multiplier**: Use `confidence_lr_multiplier` for initial experiments
2. **Monitor both LRs**: Watch both learning rates in TensorBoard/wandb
3. **Compare baselines**: Run unified LR baseline first
4. **Ablation study**: Test 0.5x, 1x, 2x, 5x multipliers
5. **Fine-tune**: Use explicit values for best performing multiplier

## ⚠️ Important Notes

- **Only works with potential field**: Requires `Config.use_potential = True`
- **Automatic fallback**: Falls back to unified LR when confidence field not present
- **Parameter group ordering**: Main model = group 0, confidence field = group 1
- **Memory usage**: No additional memory overhead
- **Backward compatibility**: Existing configs work unchanged (multiplier defaults to 1.0)

## 🔍 Implementation Details

The implementation uses PyTorch optimizer parameter groups:

```python
# Parameter separation
confidence_params = list(model.confidence_field.parameters())
main_params = [p for p in model.parameters() if id(p) not in confidence_param_ids]

# Optimizer with separate parameter groups
param_groups = [
    {'params': main_params, 'lr': lr_init},
    {'params': confidence_params, 'lr': conf_lr_init}
]
optimizer = torch.optim.Adam(param_groups, **adam_kwargs)

# Training loop applies different LRs
optimizer.param_groups[0]['lr'] = lr_main(step)  # Main model
optimizer.param_groups[1]['lr'] = lr_conf(step)  # Confidence field
```

This approach ensures clean separation while maintaining compatibility with existing training infrastructure. 