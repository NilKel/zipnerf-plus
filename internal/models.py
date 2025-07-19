import accelerate
import gin
from internal import coord
from internal import geopoly
from internal import image
from internal import math
from internal import ref_utils
from internal import train_utils
from internal import render
from internal import stepfun
from internal import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from tqdm import tqdm
from gridencoder import GridEncoder, PotentialEncoder
from internal.tri_mip import TriMipEncoding, PotentialTriMipEncoding
from internal.field import ConfidenceField
from posencoder import PositionalEncoder
try:
    from torch_scatter import segment_coo
except:
    pass

gin.config.external_configurable(math.safe_exp, module='math')


def set_kwargs(self, kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)


class DivergenceMLP(nn.Module):
    """A simple MLP that predicts divergence from flattened feature vectors."""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=None):
        """
        Initialize the divergence MLP.
        
        Args:
            input_dim: Input dimension (D * 3 where D is level_dim)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            output_dim: Output dimension (D, defaults to input_dim // 3)
        """
        super().__init__()
        
        if output_dim is None:
            output_dim = input_dim // 3
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build the network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (..., D*3)
            
        Returns:
            Predicted divergence of shape (..., D)
        """
        return self.network(x)


@gin.configurable
class Model(nn.Module):
    """A mip-Nerf360 model containing all MLPs."""
    num_prop_samples: int = 64  # The number of samples for each proposal level.
    num_nerf_samples: int = 32  # The number of samples the final nerf level.
    num_levels: int = 3  # The number of sampling levels (3==2 proposals, 1 nerf).
    bg_intensity_range = (1., 1.)  # The range of background colors.
    anneal_slope: float = 10  # Higher = more rapid annealing.
    stop_level_grad: bool = True  # If True, don't backprop across levels.
    use_viewdirs: bool = True  # If True, use view directions as input.
    raydist_fn = None  # The curve used for ray dists.
    single_jitter: bool = True  # If True, jitter whole rays instead of samples.
    use_potential: bool = False # If True, use potential grid encoder
    confidence_grid_resolution: tuple[int, int, int] = (128, 128, 128)
    dilation_multiplier: float = 0.5  # How much to dilate intervals relatively.
    dilation_bias: float = 0.0025  # How much to dilate intervals absolutely.
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    learned_exposure_scaling: bool = False  # Learned exposure scaling (RawNeRF).
    near_anneal_rate = None  # How fast to anneal in near bound.
    near_anneal_init: float = 0.95  # Where to initialize near bound (in [0, 1]).
    single_mlp: bool = False  # Use the NerfMLP for all rounds of sampling.
    distinct_prop: bool = True  # Use the NerfMLP for all rounds of sampling.
    resample_padding: float = 0.0  # Dirichlet/alpha "padding" on the histogram.
    opaque_background: bool = False  # If true, make the background opaque.
    power_lambda: float = -1.5
    std_scale: float = 0.5
    prop_desired_grid_size = [512, 2048]

    def __init__(self, config=None, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        self.config = config

        # Initialize confidence field for potential encoders or positional encoders
        # This is needed because positional encoders require confidence field for occupancy
        needs_confidence_field = self.config.use_potential
        
        # Check if positional encoders will be used (before MLPs are created)
        # We need to explicitly check gin config state since MLPs aren't created yet
        try:
            import gin
            # Check if NerfMLP or PropMLP are configured to use positional encoders
            nerf_mlp_config = gin.get_bindings('NerfMLP')
            prop_mlp_config = gin.get_bindings('PropMLP')
            uses_positional = (
                nerf_mlp_config.get('use_positional_encoder', False) or
                prop_mlp_config.get('use_positional_encoder', False)
            )
            needs_confidence_field = needs_confidence_field or uses_positional
        except:
            # If gin inspection fails, check if debug_confidence_grid_path is set
            # as a fallback indicator that confidence field is needed
            needs_confidence_field = needs_confidence_field or bool(getattr(config, 'debug_confidence_grid_path', None))
        
        if needs_confidence_field:
            self.confidence_field = ConfidenceField(
                resolution=self.config.confidence_grid_resolution, 
                device='cuda' if not self.config.dpcpp_backend else 'xpu',
                pretrained_grid_path=self.config.debug_confidence_grid_path,
                freeze_pretrained=self.config.freeze_debug_confidence,
                binary_occupancy=self.config.binary_occupancy,
                analytical_gradient=self.config.analytical_gradient,
                use_admm_pruner=self.config.use_admm_pruner,
                contraction_aware_gradients=self.config.contraction_aware_gradients
            )
        else:
            self.confidence_field = None

        from extensions import Backend
        Backend.set_backend('dpcpp' if self.config.dpcpp_backend else 'cuda')
        self.backend = Backend.get_backend()
        self.generator = self.backend.get_generator()

        # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
        # being regularized.
        self.nerf_mlp = NerfMLP(config=config,
                                num_glo_features=self.num_glo_features,
                                num_glo_embeddings=self.num_glo_embeddings)
        if self.config.dpcpp_backend:
            self.generator = self.nerf_mlp.encoder.backend.get_generator()
        else:
            self.generator = None

        if self.single_mlp:
            self.prop_mlp = self.nerf_mlp
        elif not self.distinct_prop:
            self.prop_mlp = PropMLP(config=config)
        else:
            for i in range(self.num_levels - 1):
                self.register_module(f'prop_mlp_{i}', PropMLP(config=config, grid_disired_resolution=self.prop_desired_grid_size[i]))
        if self.num_glo_features > 0 and not config.zero_glo:
            # Construct/grab GLO vectors for the cameras of each input ray.
            self.glo_vecs = nn.Embedding(self.num_glo_embeddings, self.num_glo_features)

        if self.learned_exposure_scaling:
            # Setup learned scaling factors for output colors.
            max_num_exposures = self.num_glo_embeddings
            # Initialize the learned scaling offsets at 0.
            self.exposure_scaling_offsets = nn.Embedding(max_num_exposures, 3)
            torch.nn.init.zeros_(self.exposure_scaling_offsets.weight)

        # Add triplane components - always create them to ensure parameters are registered
        self.tri_mip_encoding = TriMipEncoding(n_levels=8, plane_size=512, feature_dim=16)
        self.tri_mip_projection = nn.Sequential(
            nn.Linear(self.tri_mip_encoding.dim_out, self.nerf_mlp.encoder.output_dim),
            nn.ReLU()
        )
        
        # Add divergence MLP for regularization if enabled
        # Note: divergence regularization is only applicable to grid encoders, not positional encoders
        if (config.use_divergence_regularization and config.use_potential and 
            not getattr(self.nerf_mlp, 'use_positional_encoder', False)):
            # Input dimension is level_dim * 3 (for x, y, z components)
            div_input_dim = self.nerf_mlp.encoder.level_dim * 3
            self.div_mlp = DivergenceMLP(
                input_dim=div_input_dim,
                hidden_dim=config.div_mlp_hidden_dim,
                num_layers=config.div_mlp_num_layers,
                output_dim=self.nerf_mlp.encoder.level_dim
            )
        else:
            self.div_mlp = None

    def forward(
            self,
            rand,
            batch,
            train_frac,
            compute_extras,
            zero_glo=True,
            training_step=None,
    ):
        """The mip-NeRF Model.

    Args:
      rand: random number generator (or None for deterministic output).
      batch: util.Rays, a pytree of ray origins, directions, and viewdirs.
      train_frac: float in [0, 1], what fraction of training is complete.
      compute_extras: bool, if True, compute extra quantities besides color.
      zero_glo: bool, if True, when using GLO pass in vector of zeros.

    Returns:
      ret: list, [*(rgb, distance, acc)]
    """
        if self.config.use_potential and not self.config.analytical_gradient:
            # Only pre-compute gradients for stencil-based approach
            self.confidence_field.compute_gradient()

        device = batch['origins'].device
        if self.num_glo_features > 0:
            if not zero_glo:
                # Construct/grab GLO vectors for the cameras of each input ray.
                cam_idx = batch['cam_idx'][..., 0]
                glo_vec = self.glo_vecs(cam_idx.long())
            else:
                glo_vec = torch.zeros(batch['origins'].shape[:-1] + (self.num_glo_features,), device=device)
        else:
            glo_vec = None

        # Define the mapping from normalized to metric ray distance.
        _, s_to_t = coord.construct_ray_warps(self.raydist_fn, batch['near'], batch['far'], self.power_lambda)

        # Initialize the range of (normalized) distances for each ray to [0, 1],
        # and assign that single interval a weight of 1. These distances and weights
        # will be repeatedly updated as we proceed through sampling levels.
        # `near_anneal_rate` can be used to anneal in the near bound at the start
        # of training, eg. 0.1 anneals in the bound over the first 10% of training.
        if self.near_anneal_rate is None:
            init_s_near = 0.
        else:
            init_s_near = np.clip(1 - train_frac / self.near_anneal_rate, 0,
                                  self.near_anneal_init)
        init_s_far = 1.
        sdist = torch.cat([
            torch.full_like(batch['near'], init_s_near),
            torch.full_like(batch['far'], init_s_far)
        ], dim=-1)
        weights = torch.ones_like(batch['near'])
        prod_num_samples = 1

        ray_history = []
        renderings = []
        for i_level in range(self.num_levels):
            is_prop = i_level < (self.num_levels - 1)
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

            # Dilate by some multiple of the expected span of each current interval,
            # with some bias added in.
            dilation = self.dilation_bias + self.dilation_multiplier * (
                    init_s_far - init_s_near) / prod_num_samples

            # Record the product of the number of samples seen so far.
            prod_num_samples *= num_samples

            # After the first level (where dilation would be a no-op) optionally
            # dilate the interval weights along each ray slightly so that they're
            # overestimates, which can reduce aliasing.
            use_dilation = self.dilation_bias > 0 or self.dilation_multiplier > 0
            if i_level > 0 and use_dilation:
                sdist, weights = stepfun.max_dilate_weights(
                    sdist,
                    weights,
                    dilation,
                    domain=(init_s_near, init_s_far),
                    renormalize=True)
                sdist = sdist[..., 1:-1]
                weights = weights[..., 1:-1]

            # Optionally anneal the weights as a function of training iteration.
            if self.anneal_slope > 0:
                # Schlick's bias function, see https://arxiv.org/abs/2010.09714
                bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
                anneal = bias(train_frac, self.anneal_slope)
            else:
                anneal = 1.

            # A slightly more stable way to compute weights**anneal. If the distance
            # between adjacent intervals is zero then its weight is fixed to 0.
            logits_resample = torch.where(
                sdist[..., 1:] > sdist[..., :-1],
                anneal * torch.log(weights + self.resample_padding),
                torch.full_like(sdist[..., :-1], -torch.inf))

            # Draw sampled intervals from each ray's current weights.
            if self.config.importance_sampling:
                sdist = self.backend.funcs.sample_intervals(
                    rand,
                    sdist.contiguous(),
                    stepfun.integrate_weights(torch.softmax(logits_resample, dim=-1)).contiguous(),
                    num_samples,
                    self.single_jitter)
            else:
                sdist = stepfun.sample_intervals(
                    rand,
                    sdist,
                    logits_resample,
                    num_samples,
                    single_jitter=self.single_jitter,
                    domain=(init_s_near, init_s_far))

            # Optimization will usually go nonlinear if you propagate gradients
            # through sampling.
            if self.stop_level_grad:
                sdist = sdist.detach()

            # Convert normalized distances to metric distances.
            tdist = s_to_t(sdist)

            # Cast our rays, by turning our distance intervals into Gaussians.
            means, stds, ts = render.cast_rays(
                tdist,
                batch['origins'],
                batch['directions'],
                batch['cam_dirs'],
                batch['radii'],
                rand,
                std_scale=self.std_scale)

            # Push our Gaussians through one of our two MLPs.
            mlp = (self.get_submodule(
                f'prop_mlp_{i_level}') if self.distinct_prop else self.prop_mlp) if is_prop else self.nerf_mlp
            # Pass confidence field to MLPs that need it (potential encoders or positional encoders)
            mlp_needs_confidence = (
                self.config.use_potential or 
                getattr(mlp, 'use_positional_encoder', False)
            )
            ray_results = mlp(
                rand,
                means, stds,
                viewdirs=batch['viewdirs'] if self.use_viewdirs else None,
                imageplane=batch.get('imageplane'),
                glo_vec=None if is_prop else glo_vec,
                exposure=batch.get('exposure_values'),
                confidence_field=self.confidence_field if mlp_needs_confidence else None,
                training_step=training_step,
            )
            if self.config.gradient_scaling:
                ray_results['rgb'], ray_results['density'] = train_utils.GradientScaler.apply(
                    ray_results['rgb'], ray_results['density'], ts.mean(dim=-1))

            # Get the weights used by volumetric rendering (and our other losses).
            
            weights = render.compute_alpha_weights(
                ray_results['density'],
                tdist,
                batch['directions'],
                opaque_background=self.opaque_background,
            )[0]
            
            if ray_results['sampled_confidence'].shape[-1]!= ray_results['density'].shape[-1]:
                ray_results['sampled_confidence'] = ray_results['sampled_confidence'].mean(-1)
            weights_conf = render.compute_alpha_weights(
                ray_results['sampled_confidence'],
                tdist,
                batch['directions'],
                opaque_background=self.opaque_background,
            )[0]

            # Define or sample the background color for each ray.
            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                # If the min and max of the range are equal, just take it.
                bg_rgbs = self.bg_intensity_range[0]
            elif rand is None:
                # If rendering is deterministic, use the midpoint of the range.
                bg_rgbs = (self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2
            else:
                # Sample RGB values from the range for each ray.
                minval = self.bg_intensity_range[0]
                maxval = self.bg_intensity_range[1]
                bg_rgbs = torch.rand(weights.shape[:-1] + (3,), device=device) * (maxval - minval) + minval

            # RawNeRF exposure logic.
            if batch.get('exposure_idx') is not None:
                # Scale output colors by the exposure.
                ray_results['rgb'] *= batch['exposure_values'][..., None, :]
                if self.learned_exposure_scaling:
                    exposure_idx = batch['exposure_idx'][..., 0]
                    # Force scaling offset to always be zero when exposure_idx is 0.
                    # This constraint fixes a reference point for the scene's brightness.
                    mask = exposure_idx > 0
                    # Scaling is parameterized as an offset from 1.
                    scaling = 1 + mask[..., None] * self.exposure_scaling_offsets(exposure_idx.long())
                    ray_results['rgb'] *= scaling[..., None, :]

            # Render each ray.
            rendering = render.volumetric_rendering(
                ray_results['rgb'],
                weights,
                tdist,
                bg_rgbs,
                batch['far'],
                compute_extras,
                extras={
                    k: v
                    for k, v in ray_results.items()
                    if k.startswith('normals') or k in ['roughness']
                })

            if compute_extras:
                # Collect some rays to visualize directly. By naming these quantities
                # with `ray_` they get treated differently downstream --- they're
                # treated as bags of rays, rather than image chunks.
                n = self.config.vis_num_rays
                rendering['ray_sdist'] = sdist.reshape([-1, sdist.shape[-1]])[:n, :]
                rendering['ray_weights'] = (
                    weights.reshape([-1, weights.shape[-1]])[:n, :])
                rgb = ray_results['rgb']
                rendering['ray_rgbs'] = (rgb.reshape((-1,) + rgb.shape[-2:]))[:n, :, :]

            if self.training:
                # Compute the hash decay loss for this level.
                if isinstance(mlp.encoder, PotentialEncoder):
                    encoders = [mlp.encoder.encoder_x, mlp.encoder.encoder_y, mlp.encoder.encoder_z]
                    total_loss = 0
                    for enc in encoders:
                        param = enc.embeddings
                        if self.config.dpcpp_backend:
                            total_loss += (param ** 2).mean()
                        else:
                            idx = enc.idx.to(param.device)
                            loss = segment_coo(
                                param ** 2,
                                idx,
                                torch.zeros(idx.max() + 1, param.shape[-1], device=param.device),
                                reduce='mean'
                            ).mean()
                            total_loss += loss
                    ray_results['loss_hash_decay'] = total_loss / 3
                elif isinstance(mlp.encoder, PositionalEncoder):
                    # Positional encoders don't have hash embeddings, so no hash decay loss
                    ray_results['loss_hash_decay'] = torch.tensor(0.0, device=means.device)
                else:
                    idx = mlp.encoder.idx
                    param = mlp.encoder.embeddings
                    if self.config.dpcpp_backend:
                        ray_results['loss_hash_decay'] = (param ** 2).mean()
                    else:
                        loss_hash_decay = segment_coo(param ** 2,
                                                      idx.to(param.device),
                                                      torch.zeros(idx.max() + 1, param.shape[-1], device=param.device),
                                                      reduce='mean'
                                                      ).mean()
                        ray_results['loss_hash_decay'] = loss_hash_decay

            renderings.append(rendering)
            ray_results['sdist'] = sdist.clone()
            ray_results['weights'] = weights.clone()
            ray_results['weights_conf'] = weights_conf.clone()
            ray_history.append(ray_results)

        if compute_extras:
            # Because the proposal network doesn't produce meaningful colors, for
            # easier visualization we replace their colors with the final average
            # color.
            weights = [r['ray_weights'] for r in renderings]
            rgbs = [r['ray_rgbs'] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], dim=-2)
            avg_rgbs = [
                torch.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]
            ]
            for i in range(len(avg_rgbs)):
                renderings[i]['ray_rgbs'] = avg_rgbs[i]

        return renderings, ray_history

    def densify_grid(self, mlp, bound=1):
        """
        Densify the G-Grid by evaluating it at a dense set of points.
        
        Args:
            mlp: The MLP containing the encoder to densify
            bound: The bound for the grid coordinates
            
        Returns:
            G_dense: Dense grid of shape (D, H, W, num_levels, level_dim, 3)
        """
        if not hasattr(mlp.encoder, 'level_dim'):
            raise ValueError("MLP encoder must have level_dim attribute for divergence regularization")
        
        # Get grid resolution from confidence field
        D, H, W = self.confidence_field.resolution
        
        # Create dense coordinate grid
        z_coords = torch.linspace(-bound, bound, D, device=self.confidence_field.c_grid.device)
        y_coords = torch.linspace(-bound, bound, H, device=self.confidence_field.c_grid.device)
        x_coords = torch.linspace(-bound, bound, W, device=self.confidence_field.c_grid.device)
        
        # Create meshgrid
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        grid_coords = torch.stack([zz, yy, xx], dim=-1)  # (D, H, W, 3)
        
        # Flatten for encoder
        grid_coords_flat = grid_coords.view(-1, 3)  # (D*H*W, 3)
        
        # Evaluate encoder at dense grid
        if self.config.use_potential:
            # For potential encoder, get features and reshape
            features_flat = mlp.encoder(grid_coords_flat, bound=bound)  # (D*H*W, num_levels * level_dim * 3)
            features_reshaped = features_flat.view(D, H, W, mlp.encoder.num_levels, mlp.encoder.level_dim, 3)
            return features_reshaped
        else:
            raise ValueError("Divergence regularization requires potential encoder")

    def compute_numerical_divergence(self, G_dense):
        """
        Compute numerical divergence using finite differences.
        
        Args:
            G_dense: Dense grid of shape (D, H, W, num_levels, level_dim, 3)
            
        Returns:
            div_grid: Divergence grid of shape (D, H, W, num_levels, level_dim)
        """
        D, H, W, num_levels, level_dim, _ = G_dense.shape
        
        # Extract components
        G_x = G_dense[..., 0]  # (D, H, W, num_levels, level_dim)
        G_y = G_dense[..., 1]  # (D, H, W, num_levels, level_dim)
        G_z = G_dense[..., 2]  # (D, H, W, num_levels, level_dim)
        
        # Compute gradients using central differences
        # For boundaries, use forward/backward differences
        
        # d(G_x)/dx
        dGx_dx = torch.zeros_like(G_x)
        dGx_dx[:, :, 1:-1] = (G_x[:, :, 2:] - G_x[:, :, :-2]) / 2.0
        dGx_dx[:, :, 0] = G_x[:, :, 1] - G_x[:, :, 0]
        dGx_dx[:, :, -1] = G_x[:, :, -1] - G_x[:, :, -2]
        
        # d(G_y)/dy
        dGy_dy = torch.zeros_like(G_y)
        dGy_dy[:, 1:-1, :] = (G_y[:, 2:, :] - G_y[:, :-2, :]) / 2.0
        dGy_dy[:, 0, :] = G_y[:, 1, :] - G_y[:, 0, :]
        dGy_dy[:, -1, :] = G_y[:, -1, :] - G_y[:, -2, :]
        
        # d(G_z)/dz
        dGz_dz = torch.zeros_like(G_z)
        dGz_dz[1:-1, :, :] = (G_z[2:, :, :] - G_z[:-2, :, :]) / 2.0
        dGz_dz[0, :, :] = G_z[1, :, :] - G_z[0, :, :]
        dGz_dz[-1, :, :] = G_z[-1, :, :] - G_z[-2, :, :]
        
        # Compute divergence
        div_grid = dGx_dx + dGy_dy + dGz_dz
        
        return div_grid

    def compute_grid_divergence_loss(self, mlp):
        """
        Compute the grid-level divergence loss (L_grid).
        
        Args:
            mlp: The MLP to compute divergence loss for
            
        Returns:
            loss: Grid divergence loss
        """
        if self.div_mlp is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Densify the grid
        G_dense = self.densify_grid(mlp)  # (D, H, W, num_levels, level_dim, 3)
        
        # Compute numerical divergence
        div_stencil_grid = self.compute_numerical_divergence(G_dense)  # (D, H, W, num_levels, level_dim)
        
        # Flatten G_dense for MLP input
        D, H, W, num_levels, level_dim, _ = G_dense.shape
        G_flat = G_dense.view(-1, num_levels, level_dim, 3)  # (D*H*W, num_levels, level_dim, 3)
        
        # Process each level separately
        total_loss = 0.0
        for level in range(num_levels):
            # Get features for this level
            G_level = G_flat[:, level, :, :].flatten(-2)  # (D*H*W, level_dim * 3)
            
            # Predict divergence
            div_pred_level = self.div_mlp(G_level)  # (D*H*W, level_dim)
            
            # Reshape back to grid
            div_pred_grid_level = div_pred_level.view(D, H, W, level_dim)  # (D, H, W, level_dim)
            
            # Get target divergence for this level
            div_target_level = div_stencil_grid[:, :, :, level, :]  # (D, H, W, level_dim)
            
            # Compute MSE loss with stop gradient on target
            loss_level = F.mse_loss(div_pred_grid_level, div_target_level.detach())
            total_loss += loss_level
        
        return total_loss / num_levels

    def compute_ray_divergence_regularization(self, G_features):
        """
        Compute the ray-level divergence regularization (part of L_ray).
        
        Args:
            G_features: Vector potential features of shape (..., num_levels, level_dim, 3)
            
        Returns:
            reg_loss: Regularization loss
        """
        if self.div_mlp is None:
            return torch.tensor(0.0, device=G_features.device)
        
        # Flatten features for MLP input
        original_shape = G_features.shape[:-3]  # Everything except (num_levels, level_dim, 3)
        num_levels = G_features.shape[-3]
        level_dim = G_features.shape[-2]
        
        G_flat = G_features.view(-1, num_levels, level_dim, 3)  # (N, num_levels, level_dim, 3)
        
        total_reg = 0.0
        for level in range(num_levels):
            # Get features for this level
            G_level = G_flat[:, level, :, :].flatten(-2)  # (N, level_dim * 3)
            
            # Predict divergence
            div_pred = self.div_mlp(G_level)  # (N, level_dim)
            
            # Compute regularization: encourage low divergence
            reg_level = torch.mean(div_pred ** 2)
            total_reg += reg_level
        
        return total_reg / num_levels

    def clear_divergence_cache(self):
        """Clear cached divergence features for inference."""
        if hasattr(self.nerf_mlp, '_stored_g_features'):
            del self.nerf_mlp._stored_g_features
        for i in range(self.num_levels - 1):
            if hasattr(self, f'prop_mlp_{i}'):
                prop_mlp = getattr(self, f'prop_mlp_{i}')
                if hasattr(prop_mlp, '_stored_g_features'):
                    del prop_mlp._stored_g_features


class MLP(nn.Module):
    """A PosEnc MLP."""
    bottleneck_width: int = 256  # The width of the bottleneck vector.
    net_depth_viewdirs: int = 2  # The depth of the second part of ML.
    net_width_viewdirs: int = 256  # The width of the second part of MLP.
    skip_layer_dir: int = 0  # Add a skip connection to 2nd MLP after Nth layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
    use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
    use_directional_enc: bool = False  # If True, use IDE to encode directions.
    # If False and if use_directional_enc is True, use zero roughness in IDE.
    enable_pred_roughness: bool = False
    roughness_bias: float = -1.  # Shift added to raw roughness pre-activation.
    use_diffuse_color: bool = False  # If True, predict diffuse & specular colors.
    use_specular_tint: bool = False  # If True, predict tint.
    use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.
    bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
    density_bias: float = -1.  # Shift added to raw densities pre-activation.
    density_noise: float = 0.  # Standard deviation of noise added to raw density.
    rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
    rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    enable_pred_normals: bool = False  # If True compute predicted normals.
    disable_density_normals: bool = False  # If True don't compute normals.
    disable_rgb: bool = False  # If True don't output RGB.
    warp_fn = 'contract'
    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.
    scale_featurization: bool = False
    grid_num_levels: int = 10
    grid_level_interval: int = 2
    grid_level_dim: int = 4
    grid_base_resolution: int = 16
    grid_disired_resolution: int = 8192
    grid_log2_hashmap_size: int = 21
    net_width_glo: int = 128  # The width of the second part of MLP.
    net_depth_glo: int = 2  # The width of the second part of MLP.
    use_potential: bool = False # If true, use potential encoder
    use_positional_encoder: bool = False  # If true, use positional encoder instead of grid encoder
    pos_enc_num_freqs: int = 10  # Number of frequency bands for positional encoder
    pos_enc_log_sampling: bool = True  # Use log sampling for positional encoder frequencies
    feature_mlp_hidden_dim: int = 64  # Hidden dimension for feature MLP
    feature_mlp_num_layers: int = 2  # Number of layers in feature MLP

    def __init__(self, config=None, **kwargs):
        super().__init__()
        set_kwargs(self, kwargs)
        # Store config for later use
        self.config = config
        
        # Make sure that normals are computed if reflection direction is used.
        if self.use_reflections and not (self.enable_pred_normals or
                                         not self.disable_density_normals):
            raise ValueError('Normals must be computed for reflection directions.')

        # Precompute and define viewdir or refdir encoding function.
        if self.use_directional_enc:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), torch.zeros(1, 1)).shape[-1]
        else:

            def dir_enc_fn(direction, _):
                return coord.pos_enc(
                    direction, min_deg=0, max_deg=self.deg_view, append_identity=True)

            self.dir_enc_fn = dir_enc_fn
            dim_dir_enc = self.dir_enc_fn(torch.zeros(1, 3), None).shape[-1]
        # Choose between grid encoder and positional encoder
        use_positional_encoder = getattr(self, 'use_positional_encoder', False)
        
        if use_positional_encoder:
            # Use positional encoder
            use_potential = config is not None and getattr(config, 'use_potential', False)
            encoding_type = 'P_ENCODER' if use_potential else 'F_ENCODER'
            
            self.encoder = PositionalEncoder(
                encoding_type=encoding_type,
                num_dims=3,
                num_freqs=self.pos_enc_num_freqs,
                log_sampling=self.pos_enc_log_sampling
            )
            
            # Set output dimension for positional encoder
            if use_potential:
                # P_ENCODER outputs [..., output_dim_F, 3]
                self.encoder_output_dim = self.encoder.encoder.output_dim_F
            else:
                # F_ENCODER outputs [..., output_dim]
                self.encoder_output_dim = self.encoder.encoder.output_dim
        else:
            # Use grid encoder (original logic)
            self.grid_num_levels = int(
                np.log(self.grid_disired_resolution / self.grid_base_resolution) / np.log(self.grid_level_interval)) + 1
            
            use_potential = config is not None and getattr(config, 'use_potential', False)
            Encoder = PotentialEncoder if use_potential else GridEncoder
            
            # Prepare encoder arguments
            encoder_kwargs = {
                'input_dim': 3,
                'num_levels': self.grid_num_levels,
                'level_dim': self.grid_level_dim,
                'per_level_scale': self.grid_level_interval,  # Add missing per_level_scale parameter
                'base_resolution': self.grid_base_resolution,
                'desired_resolution': self.grid_disired_resolution,
                'log2_hashmap_size': self.grid_log2_hashmap_size,
                'gridtype': 'hash',
                'align_corners': False,
            }
            
            # Add sphere initialization parameters if this is a sphere experiment
            if use_potential and config is not None and getattr(config, 'sphere_experiment', False):
                encoder_kwargs.update({
                    'sphere_init': True,
                    'sphere_radius': getattr(config, 'sphere_radius', 1.0),
                    'sphere_center': getattr(config, 'sphere_center', [0.0, 0.0, 0.0]),
                })
            
            self.encoder = Encoder(**encoder_kwargs)
            self.encoder_output_dim = self.encoder.output_dim
        
        # Add feature MLP - required when using positional encoders
        if use_positional_encoder:
            layers = []
            input_dim = self.encoder_output_dim
            
            # Hidden layers
            for i in range(self.feature_mlp_num_layers):
                layers.append(nn.Linear(input_dim, self.feature_mlp_hidden_dim))
                layers.append(nn.ReLU())
                input_dim = self.feature_mlp_hidden_dim
            
            # Output layer - convert to grid encoder compatible dimension
            # This ensures density MLP gets same input dim regardless of encoder type
            grid_output_dim = self.grid_num_levels * self.grid_level_dim
            layers.append(nn.Linear(input_dim, grid_output_dim))
            
            self.feature_mlp = nn.Sequential(*layers)
            # Update encoder_output_dim to match grid encoder for downstream compatibility
            self.encoder_output_dim = grid_output_dim
        else:
            self.feature_mlp = None
        
        # Add triplane components
        TriplaneEncoder = PotentialTriMipEncoding if use_potential else TriMipEncoding
        self.tri_mip_encoding = TriplaneEncoder(n_levels=8, plane_size=512, feature_dim=16)

        projection_in_dim = self.tri_mip_encoding.dim_out
        if use_potential:
            # For potential encoders, multiply by 3 for the vector potential dimension
            if use_positional_encoder:
                projection_out_dim = self.encoder_output_dim * 3
            else:
                projection_out_dim = self.encoder.output_dim * 3
        else:
            # For standard encoders
            if use_positional_encoder:
                projection_out_dim = self.encoder_output_dim
            else:
                projection_out_dim = self.encoder.output_dim

        self.tri_mip_projection = nn.Sequential(
            nn.Linear(projection_in_dim, projection_out_dim),
            nn.ReLU()
        )
        
        last_dim = self.encoder_output_dim
        if self.scale_featurization and not use_positional_encoder:
            last_dim += self.encoder.num_levels
        self.density_layer = nn.Sequential(nn.Linear(last_dim, 64),
                                           nn.ReLU(),
                                           nn.Linear(64,
                                                     1 if self.disable_rgb else self.bottleneck_width))  # Hardcoded to a single channel.
        last_dim = 1 if self.disable_rgb and not self.enable_pred_normals else self.bottleneck_width
        if self.enable_pred_normals:
            self.normal_layer = nn.Linear(last_dim, 3)

        if not self.disable_rgb:
            if self.use_diffuse_color:
                self.diffuse_layer = nn.Linear(last_dim, self.num_rgb_channels)

            if self.use_specular_tint:
                self.specular_layer = nn.Linear(last_dim, 3)

            if self.enable_pred_roughness:
                self.roughness_layer = nn.Linear(last_dim, 1)

            # Output of the first part of MLP.
            if self.bottleneck_width > 0:
                last_dim_rgb = self.bottleneck_width
            else:
                last_dim_rgb = 0

            last_dim_rgb += dim_dir_enc

            if self.use_n_dot_v:
                last_dim_rgb += 1

            if self.num_glo_features > 0:
                last_dim_glo = self.num_glo_features
                for i in range(self.net_depth_glo - 1):
                    self.register_module(f"lin_glo_{i}", nn.Linear(last_dim_glo, self.net_width_glo))
                    last_dim_glo = self.net_width_glo
                self.register_module(f"lin_glo_{self.net_depth_glo - 1}",
                                     nn.Linear(last_dim_glo, self.bottleneck_width * 2))

            input_dim_rgb = last_dim_rgb
            for i in range(self.net_depth_viewdirs):
                lin = nn.Linear(last_dim_rgb, self.net_width_viewdirs)
                torch.nn.init.kaiming_uniform_(lin.weight)
                self.register_module(f"lin_second_stage_{i}", lin)
                last_dim_rgb = self.net_width_viewdirs
                if i == self.skip_layer_dir:
                    last_dim_rgb += input_dim_rgb
            self.rgb_layer = nn.Linear(last_dim_rgb, self.num_rgb_channels)

    def predict_density(self, means, stds, rand=False, no_warp=False, confidence_field=None, training_step=None, store_g_features=False):
        """Helper function to output density."""
        # Initialize sampled_conf to None - will be filled if using potential encoder
        sampled_conf = None
        
        # Encode input positions
        if self.warp_fn is not None and not no_warp:
            means, stds = coord.track_linearize(self.warp_fn, means, stds)
            # contract [-2, 2] to [-1, 1]
            bound = 2
            means = means / bound
            stds = stds / bound
        
        # Check if triplane is enabled using stored config
        use_triplane = self.config is not None and getattr(self.config, 'use_triplane', False)
        use_positional_encoder = getattr(self, 'use_positional_encoder', False)
        
        if not use_positional_encoder:
            # Move grid_sizes to the correct device and calculate weights once.
            grid_sizes = self.encoder.grid_sizes.to(stds.device)
            weights = torch.erf(1 / torch.sqrt(8 * stds[..., None] ** 2 * grid_sizes ** 2 + 1e-8))

        if use_positional_encoder:
            # Path for positional encoder
            use_potential = self.config is not None and getattr(self.config, 'use_potential', False)
            
            if use_potential:
                # P_ENCODER case: Get tensor potential and dot product with grad_occ
                if confidence_field is None:
                    raise ValueError("Confidence field must be provided when using potential with positional encoder.")
                
                # Get tensor potential G of shape [..., output_dim_F, 3]
                G = self.encoder(means)
                
                # Store G_features for divergence regularization if requested
                if store_g_features:
                    self._stored_g_features = G.clone()
                
                # Get occupancy gradient
                means_for_conf = means.view(-1, 3)
                sampled_conf_raw, sampled_grad = confidence_field.query(means_for_conf)
                
                sampled_conf = sampled_conf.view(*hash_features_per_level.shape[:-3], 1, 1, 1) # (..., 1, 1, 1)
                sampled_grad = sampled_grad.view(*hash_features_per_level.shape[:-3], 1, 1, 3) # (..., 1, 1, 3)
                
                # Dot product: G · grad_occ
                dot_product = -torch.sum(G * sampled_grad, dim=-1)  # [..., output_dim_F]
                features = dot_product
                
            else:
                # F_ENCODER case: Get positional features and multiply with occupancy
                if confidence_field is None:
                    raise ValueError("Confidence field must be provided when using positional encoder.")
                
                # Get positional features F of shape [..., output_dim]
                F = self.encoder(means)
                
                # Get occupancy (sampled_conf is actually occupancy)
                means_for_conf = means.view(-1, 3)
                sampled_conf_raw, _ = confidence_field.query(means_for_conf)
                
                # Store the raw confidence values for distortion loss (shaped like means)
                sampled_conf_unaveraged = sampled_conf_raw.view(*means.shape[:-1], 1)  # (..., num_samples, 1)
                
                # Reshape for broadcasting in feature computation
                sampled_conf_broadcast = sampled_conf_raw.view(*F.shape[:-1], 1)  # [..., 1]
                
                # Element-wise multiplication: F * occ
                features = F * sampled_conf_broadcast  # [..., output_dim]
            
            # Average both features and confidence along the sample dimension to match density computation
            features = features.mean(dim=-2)
            
            
        elif self.config is not None and getattr(self.config, 'use_potential', False):
            # Path for potential field computation (grid encoder)
            if confidence_field is None:
                raise ValueError("Confidence field must be provided when using potential.")

            # 1. Get potential features from PotentialEncoder
            potential_features_raw = self.encoder(means, bound=1)
            # Reshape to (..., num_levels, level_dim, 3)
            hash_features_per_level = potential_features_raw.view(
                *potential_features_raw.shape[:-2], self.encoder.num_levels, self.encoder.level_dim, 3
            )
            
            # Store G_features for divergence regularization if requested
            if store_g_features:
                self._stored_g_features = hash_features_per_level.clone()
            
            # 2. Interpolate confidence and gradient
            # The `means` are already in [-1, 1] for the encoder, which is what query expects.
            # aabb is (-2, -2, -2) to (2, 2, 2). So means is in (-1,1)
            # so we normalize means to be in [-1, 1]
            
            means_for_conf = means.view(-1, 3)
            sampled_conf, sampled_grad = confidence_field.query(means_for_conf)
            
            sampled_conf = sampled_conf.view(*hash_features_per_level.shape[:-3], 1, 1, 1) # (..., 1, 1, 1)
            sampled_grad = sampled_grad.view(*hash_features_per_level.shape[:-3], 1, 1, 3) # (..., 1, 1, 3)
            
            

            if use_triplane:
                # Normalize coordinates for triplane - clamp to ensure valid range
                normalized_means = torch.clamp((means + 1.0) / 2.0, 0.0, 1.0)  # Convert from [-1,1] to [0,1]
                
                # Calculate mip level from stds
                stds_mean = stds.mean(dim=-1, keepdim=True)
                trimip_level = torch.log2(stds_mean.clamp(min=1e-8, max=1e2)) + 1.0
                trimip_level = torch.clamp(trimip_level, 0, self.tri_mip_encoding.n_levels - 1)

                tri_mip_features = self.tri_mip_encoding(normalized_means, trimip_level.unsqueeze(-1))
                projected_trimip_features = self.tri_mip_projection(tri_mip_features)

                trimip_features_per_level = projected_trimip_features.view(
                    *projected_trimip_features.shape[:-1], self.encoder.num_levels, self.grid_level_dim, 3
                )
                
                blended_features_per_level = (weights[..., None, None] * hash_features_per_level) + ((1 - weights[..., None, None]) * trimip_features_per_level)
                
            else:
                blended_features_per_level = hash_features_per_level
            
            # 4. Compute dot product and multiply by confidence/occupancy
            # Dot product between feature and occupancy gradient
            
            dot_product = -torch.sum(blended_features_per_level * sampled_grad, dim=-1)
            
            # Apply the appropriate formulation based on binary_occupancy flag
            # if self.config.binary_occupancy:
            #     # Binary occupancy formulation: -binary_occ * (potential ⋅ occ_grad)
            #     # where sampled_conf_broadcast contains binary values (0 or 1) with threshold 0.001
            #     # and sampled_grad is computed from binary occupancy
            #     features = sampled_conf.squeeze(-1) * dot_product
            # else:
            #     # Smooth sigmoid formulation: -sigmoid(conf) * (potential ⋅ occ_grad)
            #     # where sampled_conf_broadcast contains continuous sigmoid values [0,1]
            #     # and sampled_grad is computed from sigmoid confidence
            #     features = sampled_conf.squeeze(-1) * dot_product

            features = dot_product
            
            if not use_triplane:
                # The shape is now (..., num_levels, level_dim)
                features = (features * weights[..., None]).mean(dim=-3).flatten(-2, -1)
            else:
                features = features.mean(dim=-3).flatten(-2, -1)

        elif use_triplane:
            # Triplane + Hashgrid blending - match original ZipNeRF processing exactly
            hash_features_raw = self.encoder(means, bound=1)
            hash_features_per_level = hash_features_raw.unflatten(-1, (self.encoder.num_levels, -1))

            # Normalize coordinates for triplane - clamp to ensure valid range
            normalized_means = torch.clamp((means + 1.0) / 2.0, 0.0, 1.0)  # Convert from [-1,1] to [0,1]
            
            # Calculate mip level from stds with better numerical stability
            stds_mean = stds.mean(dim=-1, keepdim=True)
            # Use more stable mip level calculation
            trimip_level = torch.log2(stds_mean.clamp(min=1e-8, max=1e2)) + 1.0
            trimip_level = torch.clamp(trimip_level, 0, self.tri_mip_encoding.n_levels - 1)
            
            # Get triplane features
            tri_mip_features = self.tri_mip_encoding(normalized_means, trimip_level.unsqueeze(-1))
            projected_trimip_features = self.tri_mip_projection(tri_mip_features)
            trimip_features_per_level = projected_trimip_features.unflatten(-1, (self.encoder.num_levels, -1))

            # Blend the unweighted features: zipnerf weight for hashgrid, (1-zipnerf_weight) for triplane
            # This creates a weighted combination of hash and triplane features at each level
            blended_features_per_level = (weights[..., None] * hash_features_per_level) + ((1 - weights[..., None]) * trimip_features_per_level)
            # Apply the same ZipNeRF processing as original: multiply by weights, mean across levels, flatten
            features = (blended_features_per_level).mean(dim=-3).flatten(-2, -1).squeeze(-1)
            
        else:
            # Original hashgrid-only path
            features = self.encoder(means, bound=1).unflatten(-1, (self.encoder.num_levels, -1))
            features = (features * weights[..., None]).mean(dim=-3).flatten(-2, -1)
        
        # Apply feature MLP (required for positional encoders)
        if self.feature_mlp is not None:
            features = self.feature_mlp(features)
        
        if self.scale_featurization and not use_positional_encoder:
            with torch.no_grad():
                vl2mean = segment_coo((self.encoder.embeddings ** 2).sum(-1),
                                      self.encoder.idx,
                                      torch.zeros(self.grid_num_levels, device=weights.device),
                                      self.grid_num_levels,
                                      reduce='mean'
                                      )
            featurized_w = (2 * weights.mean(dim=-2) - 1) * (self.encoder.init_std ** 2 + vl2mean).sqrt()
            features = torch.cat([features, featurized_w], dim=-1)
        x = self.density_layer(features)
        
        raw_density = x[..., 0]  # Hardcoded to a single channel.
        
        # Apply tanh(α∥v∥2) scaling for potential encoder case
        # This implements the density modulation described in the paper where:
        # - v is the feature vector after potential field processing  
        # - α is scheduled: 10^4 for steps < 1000, 10^5 for steps >= 1000
        # - The scaling helps with fast convergence initially and fine-tuning later
        if self.config.gating and self.config.use_potential:
            #Compute 2-norm squared of features (v vector after potential processing)
            v_norm_squared = torch.sum(features ** 2, dim=-1)
            
            #Set alpha according to paper: 1e4 for steps < 1000, 1e5 afterward
            if training_step is not None and training_step >= 1000:
                alpha = 1e5  # Higher alpha for fine-tuning phase
            else:
                alpha = 1e4  # Lower alpha for fast convergence phase
            
            #Allow config override for experimentation
            alpha = getattr(self.config, 'potential_alpha', alpha)
            
            #Apply tanh(α∥v∥2) scaling to modulate density
            tanh_scaling = torch.tanh(alpha * v_norm_squared)
            raw_density = raw_density * tanh_scaling
        
        # Add noise to regularize the density predictions if needed.
        if rand and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn_like(raw_density)
        if not self.config.use_potential:
            sampled_conf = torch.zeros_like(raw_density)
        else:
            sampled_conf = sampled_conf.mean(4).squeeze(-1).squeeze(-1).squeeze(-1)
        return raw_density, x, means.mean(dim=-2), sampled_conf

    def forward(self,
                rand,
                means, stds,
                viewdirs=None,
                imageplane=None,
                glo_vec=None,
                exposure=None,
                no_warp=False,
                confidence_field=None,
                training_step=None):
        """Evaluate the MLP.

    Args:
      rand: if random .
      means: [..., n, 3], coordinate means.
      stds: [..., n], coordinate stds.
      viewdirs: [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
      imageplane:[batch, 2], xy image plane coordinates
        for each ray in the batch. Useful for image plane operations such as a
        learned vignette mapping.
      glo_vec: [..., num_glo_features], The GLO vector for each ray.
      exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.
      confidence_field: The confidence field module.

    Returns:
      rgb: [..., num_rgb_channels].
      density: [...].
      normals: [..., 3], or None.
      normals_pred: [..., 3], or None.
      roughness: [..., 1], or None.
    """
        # Check if we should store G_features for divergence regularization
        store_g_features = (self.config is not None and 
                           getattr(self.config, 'use_divergence_regularization', False) and
                           self.training)
        
        if self.disable_density_normals:
            raw_density, x, means_contract, sampled_conf = self.predict_density(means, stds, rand=rand, no_warp=no_warp, confidence_field=confidence_field, training_step=training_step, store_g_features=store_g_features)
            raw_grad_density = None
            normals = None
        else:
            with torch.enable_grad():
                means.requires_grad_(True)
                raw_density, x, means_contract, sampled_conf = self.predict_density(means, stds, rand=rand, no_warp=no_warp, confidence_field=confidence_field, training_step=training_step, store_g_features=store_g_features)
                d_output = torch.ones_like(raw_density, requires_grad=False, device=raw_density.device)
                raw_grad_density = torch.autograd.grad(
                    outputs=raw_density,
                    inputs=means,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
            raw_grad_density = raw_grad_density.mean(-2)
            # Compute normal vectors as negative normalized density gradient.
            # We normalize the gradient of raw (pre-activation) density because
            # it's the same as post-activation density, but is more numerically stable
            # when the activation function has a steep or flat gradient.
            normals = -ref_utils.l2_normalize(raw_grad_density)

        if self.enable_pred_normals:
            grad_pred = self.normal_layer(x)

            # Normalize negative predicted gradients to get predicted normal vectors.
            normals_pred = -ref_utils.l2_normalize(grad_pred)
            normals_to_use = normals_pred
        else:
            grad_pred = None
            normals_pred = None
            normals_to_use = normals

        # Apply bias and activation to raw density
        density = F.softplus(raw_density + self.density_bias)

        roughness = None
        if self.disable_rgb:
            rgb = torch.zeros(density.shape + (3,), device=density.device)
        else:
            if viewdirs is not None:
                # Predict diffuse color.
                if self.use_diffuse_color:
                    raw_rgb_diffuse = self.diffuse_layer(x)

                if self.use_specular_tint:
                    tint = torch.sigmoid(self.specular_layer(x))

                if self.enable_pred_roughness:
                    raw_roughness = self.roughness_layer(x)
                    roughness = (F.softplus(raw_roughness + self.roughness_bias))

                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = x
                    # Add bottleneck noise.
                    if rand and (self.bottleneck_noise > 0):
                        bottleneck += self.bottleneck_noise * torch.randn_like(bottleneck)

                    # Append GLO vector if used.
                    if glo_vec is not None:
                        for i in range(self.net_depth_glo):
                            glo_vec = self.get_submodule(f"lin_glo_{i}")(glo_vec)
                            if i != self.net_depth_glo - 1:
                                glo_vec = F.relu(glo_vec)
                        glo_vec = torch.broadcast_to(glo_vec[..., None, :],
                                                     bottleneck.shape[:-1] + glo_vec.shape[-1:])
                        scale, shift = glo_vec.chunk(2, dim=-1)
                        bottleneck = bottleneck * torch.exp(scale) + shift

                    x = [bottleneck]
                else:
                    x = []

                # Encode view (or reflection) directions.
                if self.use_reflections:
                    # Compute reflection directions. Note that we flip viewdirs before
                    # reflecting, because they point from the camera to the point,
                    # whereas ref_utils.reflect() assumes they point toward the camera.
                    # Returned refdirs then point from the point to the environment.
                    refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
                    # Encode reflection directions.
                    dir_enc = self.dir_enc_fn(refdirs, roughness)
                else:
                    # Encode view directions.
                    dir_enc = self.dir_enc_fn(viewdirs, roughness)
                    # Check if dir_enc already has the sample dimension
                    if dir_enc.shape[:-1] == bottleneck.shape[:-1]:
                        # Already has correct shape, no need to broadcast
                        pass
                    else:
                        # Need to broadcast to match bottleneck shape
                        dir_enc = torch.broadcast_to(
                            dir_enc[..., None, :],
                            bottleneck.shape[:-1] + (dir_enc.shape[-1],))

                # Append view (or reflection) direction encoding to bottleneck vector.
                x.append(dir_enc)

                # Append dot product between normal vectors and view directions.
                if self.use_n_dot_v:
                    dotprod = torch.sum(
                        normals_to_use * viewdirs[..., None, :], dim=-1, keepdim=True)
                    x.append(dotprod)

                # Concatenate bottleneck, directional encoding, and GLO.
                x = torch.cat(x, dim=-1)
                # Output of the second part of MLP.
                inputs = x
                for i in range(self.net_depth_viewdirs):
                    x = self.get_submodule(f"lin_second_stage_{i}")(x)
                    x = F.relu(x)
                    if i == self.skip_layer_dir:
                        x = torch.cat([x, inputs], dim=-1)
            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            rgb = torch.sigmoid(self.rgb_premultiplier *
                                self.rgb_layer(x) +
                                self.rgb_bias)

            if self.use_diffuse_color:
                # Initialize linear diffuse color around 0.25, so that the combined
                # linear color is initialized around 0.5.
                diffuse_linear = torch.sigmoid(raw_rgb_diffuse - np.log(3.0))
                if self.use_specular_tint:
                    specular_linear = tint * rgb
                else:
                    specular_linear = 0.5 * rgb

                # Combine specular and diffuse components and tone map to sRGB.
                rgb = torch.clip(image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)

            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        output_dict = dict(
            coord=means_contract,
            density=density,
            rgb=rgb,
            raw_grad_density=raw_grad_density,
            grad_pred=grad_pred,
            normals=normals,
            normals_pred=normals_pred,
            roughness=roughness,
        )
        
        # Add sampled confidence if it was computed (for confidence distortion loss)
        if sampled_conf is not None:
            output_dict['sampled_confidence'] = sampled_conf.squeeze(-1)  # Remove last dimension
            
        return output_dict


@gin.configurable
class NerfMLP(MLP):
    pass


@gin.configurable
class PropMLP(MLP):
    pass


@torch.no_grad()
def render_image(model,
                 accelerator: accelerate.Accelerator,
                 batch,
                 rand,
                 train_frac,
                 config,
                 verbose=True,
                 return_weights=False):
    """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function mapping (rand, batch) -> pytree.
    accelerator: used for DDP.
    batch: a `Rays` pytree, the rays to be rendered.
    rand: if random
    config: A Config class.

  Returns:
    rgb: rendered color image.
    disp: rendered disparity image.
    acc: rendered accumulated weights per pixel.
  """
    model.eval()

    # Clear divergence cache for inference
    accelerator.unwrap_model(model).clear_divergence_cache()

    height, width = batch['origins'].shape[:2]
    num_rays = height * width
    batch = {k: v.reshape((num_rays, -1)) for k, v in batch.items() if v is not None}

    global_rank = accelerator.process_index
    chunks = []
    idx0s = tqdm(range(0, num_rays, config.render_chunk_size),
                 desc="Rendering chunk", leave=False,
                 disable=not (accelerator.is_main_process and verbose))

    for i_chunk, idx0 in enumerate(idx0s):
        chunk_batch = tree_map(lambda r: r[idx0:idx0 + config.render_chunk_size], batch)
        actual_chunk_size = chunk_batch['origins'].shape[0]
        rays_remaining = actual_chunk_size % accelerator.num_processes
        if rays_remaining != 0:
            padding = accelerator.num_processes - rays_remaining
            chunk_batch = tree_map(lambda v: torch.cat([v, torch.zeros_like(v[-padding:])], dim=0), chunk_batch)
        else:
            padding = 0
        # After padding the number of chunk_rays is always divisible by host_count.
        rays_per_host = chunk_batch['origins'].shape[0] // accelerator.num_processes
        start, stop = global_rank * rays_per_host, (global_rank + 1) * rays_per_host
        chunk_batch = tree_map(lambda r: r[start:stop], chunk_batch)

        with accelerator.autocast():
            chunk_renderings, ray_history = model(rand,
                                                  chunk_batch,
                                                  train_frac=train_frac,
                                                  compute_extras=True,
                                                  zero_glo=True,
                                                  training_step=None)

        gather = lambda v: accelerator.gather(v.contiguous())[:-padding] \
            if padding > 0 else accelerator.gather(v.contiguous())
        # Unshard the renderings.
        chunk_renderings = tree_map(gather, chunk_renderings)

        # Gather the final pass for 2D buffers and all passes for ray bundles.
        chunk_rendering = chunk_renderings[-1]
        for k in chunk_renderings[0]:
            if k.startswith('ray_'):
                chunk_rendering[k] = [r[k] for r in chunk_renderings]

        if return_weights:
            chunk_rendering['weights'] = gather(ray_history[-1]['weights'])
            chunk_rendering['coord'] = gather(ray_history[-1]['coord'])
        chunks.append(chunk_rendering)

    # Concatenate all chunks within each leaf of a single pytree.
    rendering = {}
    for k in chunks[0].keys():
        if isinstance(chunks[0][k], list):
            rendering[k] = []
            for i in range(len(chunks[0][k])):
                rendering[k].append(torch.cat([item[k][i] for item in chunks]))
        else:
            rendering[k] = torch.cat([item[k] for item in chunks])

    for k, z in rendering.items():
        if not k.startswith('ray_'):
            # Reshape 2D buffers into original image shape.
            rendering[k] = z.reshape((height, width) + z.shape[1:])

    # After all of the ray bundles have been concatenated together, extract a
    # new random bundle (deterministically) from the concatenation that is the
    # same size as one of the individual bundles.
    keys = [k for k in rendering if k.startswith('ray_')]
    if keys:
        num_rays = rendering[keys[0]][0].shape[0]
        ray_idx = torch.randperm(num_rays)
        ray_idx = ray_idx[:config.vis_num_rays]
        for k in keys:
            rendering[k] = [r[ray_idx] for r in rendering[k]]
    model.train()
    return rendering
