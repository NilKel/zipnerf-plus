import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 


_gridtype_to_id = {
    'hash': 0,
    'tiled': 1,
}

_interp_to_id = {
    'linear': 0,
    'smoothstep': 1,
}

class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, backend, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, gridtype=0, align_corners=False, interpolation=0):
        # inputs: [B, D], float in [0, 1]
        # embeddings: [sO, C], float
        # offsets: [L + 1], int
        # RETURN: [B, F], float

        inputs = inputs.contiguous()

        B, D = inputs.shape # batch size, coord dim
        L = offsets.shape[0] - 1 # level
        C = embeddings.shape[1] # embedding dim for each level
        S = np.log2(per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = base_resolution # base resolution

        # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # if C % 2 != 0, force float, since half for atomicAdd is very slow.
        if torch.is_autocast_enabled() and C % 2 == 0:
            embeddings = embeddings.to(torch.half)

        # L first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        backend.synchronize()
        backend.funcs.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, gridtype, align_corners, interpolation)

        # permute back to [B, L * C]
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)

        ctx.backend = backend
        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype, interpolation]
        ctx.align_corners = align_corners

        return outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        backend = ctx.backend
        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype, interpolation = ctx.dims
        align_corners = ctx.align_corners

        # grad: [B, L * C] --> [L, B, C]
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        backend.synchronize()
        backend.funcs.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation)

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return None, grad_inputs, grad_embeddings, None, None, None, None, None, None, None


grid_encode = _grid_encode.apply


class GridEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2,
                 per_level_scale=2, base_resolution=16,
                 log2_hashmap_size=19, desired_resolution=None,
                 gridtype='hash', align_corners=False,
                 interpolation='linear', init_std=1e-4):
        super().__init__()

        # the finest resolution desired at the last level, if provided, overridee per_level_scale
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))

        self.input_dim = input_dim # coord dims, 2 or 3
        self.num_levels = num_levels # num levels, each level multiply resolution by 2
        self.level_dim = level_dim # encode channels per level
        self.per_level_scale = per_level_scale # multiply resolution by this scale at each level.
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype] # "tiled" or "hash"
        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation] # "linear" or "smoothstep"
        self.align_corners = align_corners
        self.init_std = init_std

        from extensions import Backend
        self.backend = Backend.get_backend()
        self.save_iteration = 0  # Flag to track if the file has been saved

        # allocate parameters
        resolutions = []
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            resolution = (resolution if align_corners else resolution + 1)
            params_in_level = min(self.max_params, resolution ** input_dim) # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
            resolutions.append(resolution)
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        idx = torch.empty(offset, dtype=torch.long)
        for i in range(self.num_levels):
            idx[offsets[i]:offsets[i+1]] = i
        self.register_buffer('idx', idx)
        self.register_buffer('grid_sizes', torch.from_numpy(np.array(resolutions, dtype=np.int32)))
        
        self.n_params = offsets[-1] * level_dim

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))

        self.reset_parameters()
    
    def reset_parameters(self):
        std = self.init_std
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners} interpolation={self.interpolation}"
    
    def forward(self, inputs, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim]

        inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
        # inputs = inputs.clamp(0, 1)
        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)

        outputs = grid_encode(self.backend, inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad, self.gridtype_id, self.align_corners, self.interp_id)
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs

    # always run in float precision!
    @torch.cuda.amp.autocast(enabled=False)
    def grad_total_variation(self, weight=1e-7, inputs=None, bound=1, B=1000000):
        # inputs: [..., input_dim], float in [-b, b], location to calculate TV loss.
        
        D = self.input_dim
        C = self.embeddings.shape[1] # embedding dim for each level
        L = self.offsets.shape[0] - 1 # level
        S = np.log2(self.per_level_scale) # resolution multiplier at each level, apply log2 for later CUDA exp2f
        H = self.base_resolution # base resolution

        if inputs is None:
            # randomized in [0, 1]
            inputs = torch.rand(B, self.input_dim, device=self.embeddings.device)
        else:
            inputs = (inputs + bound) / (2 * bound) # map to [0, 1]
            inputs = inputs.view(-1, self.input_dim)
            B = inputs.shape[0]

        if self.embeddings.grad is None:
            raise ValueError('grad is None, should be called after loss.backward() and before optimizer.step()!')

        self.backend.funcs.grad_total_variation(inputs, self.embeddings, self.embeddings.grad, self.offsets, weight, B, D, C, L, S, H, self.gridtype_id, self.align_corners)


class PotentialEncoder(nn.Module):
    def __init__(self, input_dim=3, num_levels=16, level_dim=2,
                 per_level_scale=2, base_resolution=16,
                 log2_hashmap_size=19, desired_resolution=None,
                 gridtype='hash', align_corners=False,
                 interpolation='linear', init_std=1e-4,
                 sphere_init=False, sphere_radius=1.0, sphere_center=None):
        super().__init__()

        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.vector_dim = 3
        self.output_dim = self.num_levels * self.level_dim
        self.sphere_init = sphere_init
        self.sphere_radius = sphere_radius
        self.sphere_center = sphere_center if sphere_center is not None else [0.0, 0.0, 0.0]

        common_kwargs = {
            'input_dim': input_dim,
            'num_levels': num_levels,
            'level_dim': level_dim,
            'per_level_scale': per_level_scale,
            'base_resolution': base_resolution,
            'log2_hashmap_size': log2_hashmap_size,
            'desired_resolution': desired_resolution,
            'gridtype': gridtype,
            'align_corners': align_corners,
            'interpolation': interpolation,
            'init_std': init_std,
        }

        self.encoder_x = GridEncoder(**common_kwargs)
        self.encoder_y = GridEncoder(**common_kwargs)
        self.encoder_z = GridEncoder(**common_kwargs)
        
        # Apply sphere-based initialization if requested
        if self.sphere_init:
            print("🌟 Sphere initialization temporarily disabled due to memory constraints")
            print("   Using random initialization instead")
            # self._initialize_with_sphere()  # Disabled for memory reasons

        # For compatibility
        self.n_params = self.encoder_x.n_params * 3
        # The following are for compatibility with code that inspects the encoder
        self.embeddings = self.encoder_x.embeddings
        self.offsets = self.encoder_x.offsets
        self.grid_sizes = self.encoder_x.grid_sizes
        self.backend = self.encoder_x.backend
        self.gridtype = self.encoder_x.gridtype
        self.gridtype_id = self.encoder_x.gridtype_id
        self.align_corners = self.encoder_x.align_corners
        self.interpolation = self.encoder_x.interpolation
        self.interp_id = self.encoder_x.interp_id
        self.base_resolution = self.encoder_x.base_resolution
        self.per_level_scale = self.encoder_x.per_level_scale
        self.init_std = self.encoder_x.init_std
        self.idx = self.encoder_x.idx


    def reset_parameters(self):
        self.encoder_x.reset_parameters()
        self.encoder_y.reset_parameters()
        self.encoder_z.reset_parameters()

    def __repr__(self):
        base_repr = self.encoder_x.__repr__()
        return f"PotentialEncoder (wraps 3 GridEncoders):\n - {base_repr}"
    
    def forward(self, inputs, bound=1):
        # inputs: [..., input_dim], normalized real world positions in [-bound, bound]
        # return: [..., num_levels * level_dim, 3]

        x_out = self.encoder_x(inputs, bound=bound)
        y_out = self.encoder_y(inputs, bound=bound)
        z_out = self.encoder_z(inputs, bound=bound)

        return torch.stack([x_out, y_out, z_out], dim=-1)

    # always run in float precision!
    @torch.cuda.amp.autocast(enabled=False)
    def grad_total_variation(self, weight=1e-7, inputs=None, bound=1, B=1000000):
        self.encoder_x.grad_total_variation(weight, inputs, bound, B)
        self.encoder_y.grad_total_variation(weight, inputs, bound, B)
        self.encoder_z.grad_total_variation(weight, inputs, bound, B)
    
    def _initialize_with_sphere(self):
        """
        Initialize the potential encoder embeddings based on sphere geometry.
        This creates potential fields that are geometrically meaningful for the sphere.
        """
        print(f"🌟 Initializing PotentialEncoder with sphere geometry")
        print(f"   Sphere radius: {self.sphere_radius}")
        print(f"   Sphere center: {self.sphere_center}")
        
        sphere_center_tensor = torch.tensor(self.sphere_center, dtype=torch.float32)
        
        # Initialize embeddings for each level
        for level in range(self.num_levels):
            # Get resolution for this level
            resolution = int(self.encoder_x.base_resolution * (self.encoder_x.per_level_scale ** level))
            
            # Create coordinate grid for this level
            coords = torch.linspace(-1, 1, resolution)
            if self.input_dim == 3:
                Z, Y, X = torch.meshgrid(coords, coords, coords, indexing='ij')
                grid_coords = torch.stack([X, Y, Z], dim=-1)  # (res, res, res, 3)
            else:
                raise NotImplementedError("Only 3D input supported for sphere initialization")
            
            # Compute sphere-based potential values
            # Distance from sphere center
            distances = torch.norm(grid_coords - sphere_center_tensor, dim=-1)
            
            # Different potential formulations for X, Y, Z components:
            
            # X component: radial potential (distance-based)
            x_potential = self._compute_radial_potential(distances, level)
            
            # Y component: angular potential (based on angle from sphere center)
            y_potential = self._compute_angular_potential(grid_coords, sphere_center_tensor, level)
            
            # Z component: height potential (based on Z coordinate relative to sphere)
            z_potential = self._compute_height_potential(grid_coords, sphere_center_tensor, level)
            
            # Flatten potentials for this level
            x_flat = x_potential.reshape(-1)
            y_flat = y_potential.reshape(-1)  
            z_flat = z_potential.reshape(-1)
            
            # Get indices for this level in the embeddings
            start_idx = self.encoder_x.offsets[level].item()
            end_idx = self.encoder_x.offsets[level + 1].item()
            level_size = end_idx - start_idx
            
            # Sample or interpolate to match embedding size
            if len(x_flat) >= level_size:
                # Subsample if we have more values than needed
                indices = torch.linspace(0, len(x_flat) - 1, level_size, dtype=torch.long)
                x_values = x_flat[indices]
                y_values = y_flat[indices]
                z_values = z_flat[indices]
            else:
                # Interpolate if we need more values
                x_values = torch.nn.functional.interpolate(
                    x_flat.unsqueeze(0).unsqueeze(0), size=level_size, mode='linear', align_corners=True
                ).squeeze()
                y_values = torch.nn.functional.interpolate(
                    y_flat.unsqueeze(0).unsqueeze(0), size=level_size, mode='linear', align_corners=True
                ).squeeze()
                z_values = torch.nn.functional.interpolate(
                    z_flat.unsqueeze(0).unsqueeze(0), size=level_size, mode='linear', align_corners=True
                ).squeeze()
            
            # Initialize embeddings for this level
            with torch.no_grad():
                for dim in range(self.level_dim):
                    # Cycle through the potential components
                    if dim % 3 == 0:
                        values = x_values
                    elif dim % 3 == 1:
                        values = y_values
                    else:
                        values = z_values
                    
                    # Apply to the embeddings with some scaling
                    scale = 0.1 / (level + 1)  # Smaller values for higher levels
                    self.encoder_x.embeddings.data[start_idx:end_idx, dim] = values * scale
                    self.encoder_y.embeddings.data[start_idx:end_idx, dim] = values * scale * 0.8
                    self.encoder_z.embeddings.data[start_idx:end_idx, dim] = values * scale * 0.6
        
        print(f"   ✅ Initialized {self.num_levels} levels with sphere-based potentials")
    
    def _compute_radial_potential(self, distances, level):
        """Compute radial potential based on distance from sphere center."""
        # Potential decreases with distance from sphere surface
        surface_distance = torch.abs(distances - self.sphere_radius)
        potential = torch.exp(-surface_distance * (level + 1))
        return potential
    
    def _compute_angular_potential(self, coords, sphere_center, level):
        """Compute angular potential based on angular position."""
        # Vector from sphere center to each point
        vectors = coords - sphere_center
        # Angular component (normalized)
        angles = torch.atan2(vectors[..., 1], vectors[..., 0])  # XY plane angle
        potential = torch.sin(angles * (level + 1)) * 0.5
        return potential
    
    def _compute_height_potential(self, coords, sphere_center, level):
        """Compute height-based potential."""
        # Height relative to sphere center
        height = coords[..., 2] - sphere_center[2]
        # Height-based potential
        potential = torch.tanh(height * (level + 1)) * 0.5
        return potential
