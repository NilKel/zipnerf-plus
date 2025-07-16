#!/usr/bin/env python3
"""
Analytical Model Components for Oracle Experiment

This module implements modified MLP classes that use the analytical oracle components
instead of learned hash grid encoders. This allows testing the mathematical formulation
with perfect ground truth data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from internal import coord, ref_utils, render, stepfun
from internal.models import set_kwargs
from analytical_oracles import create_analytical_model_components


class AnalyticalMLP(nn.Module):
    """
    MLP that uses analytical oracle components instead of learned encoders.
    
    This tests whether the feature V_feat = -G⋅∇O can be used by an MLP
    to reconstruct the scene when all other components are perfect.
    """
    
    # Default parameters (matching the base MLP class)
    bottleneck_width: int = 256
    net_depth_viewdirs: int = 2
    net_width_viewdirs: int = 256
    skip_layer_dir: int = 0
    num_rgb_channels: int = 3
    deg_view: int = 4
    use_reflections: bool = False
    use_directional_enc: bool = False
    enable_pred_roughness: bool = False
    roughness_bias: float = -1.
    use_diffuse_color: bool = False
    use_specular_tint: bool = False
    use_n_dot_v: bool = False
    bottleneck_noise: float = 0.0
    density_bias: float = -1.
    density_noise: float = 0.
    rgb_premultiplier: float = 1.
    rgb_bias: float = 0.
    rgb_padding: float = 0.001
    enable_pred_normals: bool = False
    disable_density_normals: bool = False
    disable_rgb: bool = False
    warp_fn = 'contract'
    num_glo_features: int = 0
    num_glo_embeddings: int = 1000
    grid_num_levels: int = 8
    grid_level_dim: int = 2
    
    def __init__(self, config=None, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        self.config = config
        
        # Initialize analytical oracles
        sphere_radius = getattr(config, 'sphere_radius', 1.0)
        sphere_center = getattr(config, 'sphere_center', [0.0, 0.0, 0.0])
        sphere_epsilon = getattr(config, 'sphere_epsilon', 0.05)
        
        self.oracles, self.encoder, self.confidence_field = create_analytical_model_components(
            sphere_radius=sphere_radius,
            sphere_center=sphere_center,
            epsilon=sphere_epsilon,
            num_levels=self.grid_num_levels,
            level_dim=self.grid_level_dim
        )
        
        # Set up view direction encoding
        if self.use_directional_enc:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), torch.zeros(1, 1)).shape[-1]
        else:
            def dir_enc_fn(direction, _):
                return coord.pos_enc(
                    direction, min_deg=0, max_deg=self.deg_view, append_identity=True)
            self.dir_enc_fn = dir_enc_fn
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), None).shape[-1]
        
        # The feature dimension: V_feat + ∇O
        # V_feat has shape (..., grid_level_dim) 
        # ∇O has shape (..., 3)
        # Total input dimension: grid_level_dim + 3
        feature_dim = self.grid_level_dim + 3
        
        # Density prediction network
        self.density_layer = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1 if self.disable_rgb else self.bottleneck_width)
        )
        
        # Normal prediction (if enabled)
        last_dim = 1 if self.disable_rgb and not self.enable_pred_normals else self.bottleneck_width
        if self.enable_pred_normals:
            self.normal_layer = nn.Linear(last_dim, 3)
        
        # RGB prediction network (if not disabled)
        if not self.disable_rgb:
            if self.use_diffuse_color:
                self.diffuse_layer = nn.Linear(last_dim, self.num_rgb_channels)
            
            if self.use_specular_tint:
                self.specular_layer = nn.Linear(last_dim, 3)
            
            if self.enable_pred_roughness:
                self.roughness_layer = nn.Linear(last_dim, 1)
            
            # RGB MLP
            last_dim_rgb = self.bottleneck_width + dim_dir_enc
            if self.use_n_dot_v:
                last_dim_rgb += 1
            
            input_dim_rgb = last_dim_rgb
            for i in range(self.net_depth_viewdirs):
                lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                torch.nn.init.kaiming_uniform_(lin.weight)
                self.register_module(f"lin_second_stage_{i}", lin)
                last_dim_rgb = self.net_width_viewdirs
                if i == self.skip_layer_dir:
                    last_dim_rgb += input_dim_rgb
            
            self.rgb_layer = nn.Linear(last_dim_rgb, self.num_rgb_channels)
        
        print(f"🔮 AnalyticalMLP initialized:")
        print(f"   Feature dim: {feature_dim} (V_feat: {self.grid_level_dim} + ∇O: 3)")
        print(f"   Expected V_feat on sphere: {self.oracles.expected_v_feat_on_sphere():.4f}")
    
    def predict_density(self, means, stds, rand=False, no_warp=False):
        """Predict density using analytical features."""
        
        # Apply coordinate warping if needed
        if self.warp_fn is not None and not no_warp:
            means, stds = coord.track_linearize(self.warp_fn, means, stds)
            bound = 2
            means = means / bound
            stds = stds / bound
        
        # Get analytical features
        batch_shape = means.shape[:-1]
        means_flat = means.view(-1, 3)
        
        # Compute V_feat = -G⋅∇O using oracles
        v_feat = self.oracles.compute_v_feat(means_flat, self.grid_level_dim)  # (N, grid_level_dim)
        
        # Get occupancy gradients ∇O
        grad_o = self.oracles.analytical_grad_o_query(means_flat)  # (N, 3)
        
        # Concatenate features: [V_feat, ∇O]
        mlp_input = torch.cat([v_feat, grad_o], dim=-1)  # (N, grid_level_dim + 3)
        
        # Reshape back to batch dimensions
        mlp_input = mlp_input.view(*batch_shape, -1)
        
        # Pass through density network
        x = self.density_layer(mlp_input)
        
        raw_density = x[..., 0] if not self.disable_rgb else x.squeeze(-1)
        
        # Add noise if training
        if rand and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn_like(raw_density)
        
        return raw_density, x, means.mean(dim=-2)
    
    def forward(self, rand, means, stds, viewdirs=None, imageplane=None, 
                glo_vec=None, exposure=None, no_warp=False, **kwargs):
        """
        Forward pass of analytical MLP.
        
        The MLP learns to use:
        1. V_feat = -G⋅∇O (for density prediction)
        2. ∇O (surface normals for color prediction)
        
        Expected behavior:
        - High density when V_feat ≠ 0 (on sphere surface)
        - Color based on surface normal direction
        """
        
        # Compute density with analytical features
        if self.disable_density_normals:
            raw_density, x, means_contract = self.predict_density(
                means, stds, rand=rand, no_warp=no_warp)
            raw_grad_density = None
            normals = None
        else:
            # Compute gradients of density w.r.t. input coordinates
            with torch.enable_grad():
                means.requires_grad_(True)
                raw_density, x, means_contract = self.predict_density(
                    means, stds, rand=rand, no_warp=no_warp)
                d_output = torch.ones_like(raw_density, requires_grad=False, device=raw_density.device)
                raw_grad_density = torch.autograd.grad(
                    outputs=raw_density,
                    inputs=means,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
            if raw_grad_density is None:
                # If gradients can't be computed, create zero gradients
                raw_grad_density = torch.zeros_like(means)
            raw_grad_density = raw_grad_density.mean(-2)
            normals = -ref_utils.l2_normalize(raw_grad_density)
        
        # Predicted normals (if enabled)
        if self.enable_pred_normals:
            grad_pred = self.normal_layer(x)
            normals_pred = -ref_utils.l2_normalize(grad_pred)
            normals_to_use = normals_pred
        else:
            grad_pred = None
            normals_pred = None
            normals_to_use = normals
        
        # Apply activation to density
        density = F.softplus(raw_density + self.density_bias)
        
        # RGB prediction
        if self.disable_rgb:
            rgb = torch.ones(*density.shape[:-1], 3, device=density.device)
        else:
            # Get bottleneck features
            if self.bottleneck_width > 0:
                bottleneck = x[..., 1:(self.bottleneck_width + 1)]
                
                # Add bottleneck noise if training
                if rand and (self.bottleneck_noise > 0):
                    bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)
            else:
                bottleneck = torch.zeros(*x.shape[:-1], 0, device=x.device)
            
            # Prepare inputs for RGB network
            x_rgb = [bottleneck]
            
            if viewdirs is not None:
                # Encode view directions
                if self.use_reflections:
                    refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
                    dir_enc = self.dir_enc_fn(refdirs, None)
                else:
                    dir_enc = self.dir_enc_fn(viewdirs, None)
                    dir_enc = torch.broadcast_to(
                        dir_enc[..., None, :],
                        bottleneck.shape[:-1] + (dir_enc.shape[-1],))
                
                x_rgb.append(dir_enc)
                
                # Add n⋅v term if enabled
                if self.use_n_dot_v:
                    dotprod = torch.sum(
                        normals_to_use * viewdirs[..., None, :], dim=-1, keepdim=True)
                    x_rgb.append(dotprod)
            
            # Concatenate all RGB inputs
            x_rgb = torch.cat(x_rgb, dim=-1)
            
            # Pass through RGB network
            inputs = x_rgb
            for i in range(self.net_depth_viewdirs):
                x_rgb = self.get_submodule(f"lin_second_stage_{i}")(x_rgb)
                x_rgb = F.relu(x_rgb)
                if i == self.skip_layer_dir:
                    x_rgb = torch.cat([x_rgb, inputs], dim=-1)
            
            # Final RGB prediction
            rgb = torch.sigmoid(self.rgb_premultiplier * self.rgb_layer(x_rgb) + self.rgb_bias)
        
        # Prepare return values
        ret = {
            'rgb': rgb,
            'density': density,
        }
        
        if normals is not None:
            ret['normals'] = normals
        if normals_pred is not None:
            ret['normals_pred'] = normals_pred
        
        return ret


class AnalyticalModel(nn.Module):
    """
    Complete model for analytical oracle experiment.
    
    This replaces the normal Model class but uses analytical MLPs instead
    of learned grid encoders.
    """
    
    # Default parameters (matching Model class)
    num_prop_samples: int = 32
    num_nerf_samples: int = 16
    num_levels: int = 2
    bg_intensity_range = (1., 1.)
    anneal_slope: float = 10
    stop_level_grad: bool = True
    use_viewdirs: bool = True
    raydist_fn = None
    single_jitter: bool = True
    dilation_multiplier: float = 0.5
    dilation_bias: float = 0.0025
    num_glo_features: int = 0
    num_glo_embeddings: int = 1000
    learned_exposure_scaling: bool = False
    near_anneal_rate = None
    near_anneal_init: float = 0.95
    single_mlp: bool = False
    distinct_prop: bool = True
    resample_padding: float = 0.0
    opaque_background: bool = False
    power_lambda: float = -1.5
    
    def __init__(self, config=None, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        self.config = config
        
        print(f"🔮 AnalyticalModel initializing...")
        
        # Create analytical MLPs
        self.nerf_mlp = AnalyticalMLP(config=config)
        
        if self.single_mlp:
            self.prop_mlp = self.nerf_mlp
        elif not self.distinct_prop:
            # Single proposal MLP
            self.prop_mlp = AnalyticalMLP(config=config, disable_rgb=True)
        else:
            # Multiple proposal MLPs
            for i in range(self.num_levels - 1):
                prop_mlp = AnalyticalMLP(config=config, disable_rgb=True)
                self.register_module(f'prop_mlp_{i}', prop_mlp)
        
        # GLO embeddings (if enabled)
        if self.num_glo_features > 0:
            self.glo_vecs = nn.Embedding(self.num_glo_embeddings, self.num_glo_features)
        
        # Learned exposure scaling (if enabled)
        if self.learned_exposure_scaling:
            self.exposure_scaling_offsets = nn.Embedding(self.num_glo_embeddings, 3)
            torch.nn.init.zeros_(self.exposure_scaling_offsets.weight)
        
        print(f"✅ AnalyticalModel initialized with {self.num_levels} levels")
    
    def forward(self, rand, batch, train_frac, compute_extras, zero_glo=True, **kwargs):
        """Forward pass using analytical components."""
        
        device = batch['origins'].device
        
        # GLO vector handling
        if self.num_glo_features > 0:
            if not zero_glo:
                cam_idx = batch['cam_idx'][..., 0]
                glo_vec = self.glo_vecs(cam_idx.long())
            else:
                glo_vec = torch.zeros(batch['origins'].shape[:-1] + (self.num_glo_features,), device=device)
        else:
            glo_vec = None
        
        # Set up ray warping
        _, s_to_t = coord.construct_ray_warps(self.raydist_fn, batch['near'], batch['far'], self.power_lambda)
        
        # Initialize sampling intervals
        if self.near_anneal_rate is None:
            init_s_near = 0.
        else:
            init_s_near = np.clip(1 - train_frac / self.near_anneal_rate, 0, self.near_anneal_init)
        
        # Initial uniform sampling
        init_s_far = 1.
        sdist = torch.cat([
            torch.full_like(batch['near'], init_s_near),
            torch.full_like(batch['far'], init_s_far)
        ], dim=-1)
        weights = torch.ones_like(sdist[..., :-1])
        
        renderings = []
        ray_history = []
        
        # Hierarchical sampling
        for i_level in range(self.num_levels):
            is_prop = i_level < (self.num_levels - 1)
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples
            
            # Get MLP for this level
            if self.single_mlp:
                mlp = self.nerf_mlp
            elif not self.distinct_prop and is_prop:
                mlp = self.prop_mlp
            elif is_prop:
                mlp = self.get_submodule(f'prop_mlp_{i_level}')
            else:
                mlp = self.nerf_mlp
            
            # Sample points along rays
            if i_level == 0:
                # Uniform sampling for first level
                # Use uniform weights for initial sampling
                w_logits = torch.zeros_like(weights)
                sdist = stepfun.sample_intervals(
                    rand, sdist, w_logits, num_samples, self.single_jitter)
            else:
                # Importance sampling for subsequent levels
                # Convert weights to logits for sampling
                eps = torch.finfo(weights.dtype).eps
                w_logits = torch.log(weights.clamp_min(eps))
                if self.stop_level_grad:
                    w_logits = w_logits.detach()
                sdist = stepfun.sample_intervals(
                    rand, sdist, w_logits, num_samples, self.single_jitter)
            
            # Convert to metric coordinates
            tdist = s_to_t(sdist)
            
            # Compute sample positions and directions
            means, stds, _ = render.cast_rays(
                tdist, batch['origins'], batch['directions'], 
                batch['cam_dirs'], batch['radii'], rand=rand is not None)
            
            # Evaluate MLP
            ray_results = mlp(
                rand, means, stds,
                viewdirs=batch['viewdirs'] if self.use_viewdirs else None,
                imageplane=batch.get('imageplane'),
                glo_vec=None if is_prop else glo_vec,
                exposure=batch.get('exposure_values')
            )
            
            # Compute alpha weights for volumetric rendering
            weights = render.compute_alpha_weights(
                ray_results['density'], tdist, batch['directions'],
                opaque_background=self.opaque_background)[0]
            
            # Background color
            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                bg_rgbs = self.bg_intensity_range[0]
            elif rand is None:
                bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
            else:
                minval, maxval = self.bg_intensity_range
                bg_rgbs = torch.rand(weights.shape[:-1] + (3,), device=device) * (maxval - minval) + minval
            
            # Apply exposure scaling if needed
            if batch.get('exposure_idx') is not None:
                ray_results['rgb'] *= batch['exposure_values'][..., None, :]
                if self.learned_exposure_scaling:
                    exposure_idx = batch['exposure_idx'][..., 0]
                    mask = exposure_idx > 0
                    scaling = 1 + mask[..., None] * self.exposure_scaling_offsets(exposure_idx.long())
                    ray_results['rgb'] *= scaling[..., None, :]
            
            # Volumetric rendering
            rendering = render.volumetric_rendering(
                ray_results['rgb'], weights, tdist, bg_rgbs, batch['far'],
                compute_extras, extras={
                    k: v for k, v in ray_results.items()
                    if k.startswith('normals') or k in ['roughness']
                })
            
            # Add visualization rays if requested
            if compute_extras:
                n = getattr(self.config, 'vis_num_rays', 1024)
                rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
                rendering['ray_weights'] = weights.reshape([-1, weights.shape[-1]])[:n, :]
                rgb = ray_results['rgb']
                rendering['ray_rgbs'] = rgb.reshape((-1,) + rgb.shape[-2:])[:n, :, :]
            
            renderings.append(rendering)
            ray_history.append(ray_results)
        
        # Post-process visualization
        if compute_extras:
            weights = [r['ray_weights'] for r in renderings]
            rgbs = [r['ray_rgbs'] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], dim=-2)
            avg_rgbs = [torch.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]]
            for i in range(len(avg_rgbs)):
                renderings[i]['ray_rgbs'] = avg_rgbs[i]
        
        return renderings, ray_history 