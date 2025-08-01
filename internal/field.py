import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from internal import coord

class ConfidenceField(nn.Module):
    def __init__(self, resolution=(128, 128, 128), init_val=-1, init_rand_mag=2.0, post_mult=0.01, device='cuda', 
                 stencil_type='central_difference_4th_order', pretrained_grid_path=None, freeze_pretrained=True,
                 binary_occupancy=False, analytical_gradient=False, use_admm_pruner=False, contraction_aware_gradients=True,
                 non_spherical_contraction=False, non_uniform_cells=False):
        """
        Initialize ConfidenceField.
        
        Args:
            resolution: Grid resolution (D, H, W)
            init_val: Initial value for random initialization
            init_rand_mag: Magnitude of random initialization
            device: Device to place tensors on
            stencil_type: Type of finite difference stencil
            pretrained_grid_path: Optional path to pretrained confidence grid (.pt file)
            freeze_pretrained: If True and pretrained_grid_path is provided, freeze the grid (no gradients)
            binary_occupancy: If True, use binary occupancy with STE instead of smooth sigmoid
            analytical_gradient: If True, use analytical gradient (autograd) instead of stencil-based finite differences
            use_admm_pruner: If True, enable ADMM pruning functionality
            contraction_aware_gradients: If True, account for spatial contraction in gradient computation
            non_spherical_contraction: If True, use cubic contraction; if False, use spherical contraction
            non_uniform_cells: If True, use non-uniform finite differences for variable cell spacing
        """
        super().__init__()
        self.resolution = resolution
        self.pretrained_grid_path = pretrained_grid_path
        self.freeze_pretrained = freeze_pretrained
        self.binary_occupancy = binary_occupancy
        self.analytical_gradient = analytical_gradient
        self.use_admm_pruner = use_admm_pruner
        self.contraction_aware_gradients = contraction_aware_gradients
        self.non_spherical_contraction = non_spherical_contraction
        self.non_uniform_cells = non_uniform_cells
        
        # Initialize logits to be slightly negative on average
        self.c_grid = nn.Parameter(torch.randn(*resolution, device=device) * init_rand_mag + init_val)
        # self.c_grid = nn.Parameter(torch.nn.init.uniform_(*resolution, -0.01, 0.01))
        
        # ADMM Pruner initialization
        if self.use_admm_pruner:
            # Dual variable (γ) for ADMM optimization - Lagrange multiplier
            self.dual_variable = nn.Parameter(torch.tensor(0.0, device=device))
            self.dual_variable.requires_grad_(False)  # Updated manually, not by optimizer
            
            # Cache for total grid elements (computed once)
            self.total_grid_elements = self.c_grid.nelement()
        
        # Load pretrained grid if provided
        if pretrained_grid_path is not None and pretrained_grid_path != '':
            self._load_pretrained_grid(pretrained_grid_path, device, stencil_type)
        
        self.grad_c_grid = None
        self.binary_c_grid = None  # Store binary occupancy grid when using STE
        
        # Only build stencil kernels if not using analytical gradients
        if not self.analytical_gradient:
            self._build_kernels(device, stencil_type)

    def _load_pretrained_grid(self, pretrained_grid_path, device, stencil_type):
        """
        Load pretrained confidence grid from file.
        
        Args:
            pretrained_grid_path: Path to the .pt file containing confidence logits or probabilities
            device: Device to place the grid on
        """
        grid_path = Path(pretrained_grid_path)
        if not grid_path.exists():
            raise FileNotFoundError(f"Pretrained confidence grid not found: {pretrained_grid_path}")
        
        print(f"🔧 Loading pretrained confidence grid from: {pretrained_grid_path}")
        
        # Load the pretrained grid
        pretrained_data = torch.load(pretrained_grid_path, map_location='cpu')
        
        # Handle both formats: raw tensor (logits) or dict with metadata (probabilities)
        if isinstance(pretrained_data, dict):
            # New format with metadata - contains probabilities
            if 'occupancy_probabilities' in pretrained_data:
                pretrained_probs = pretrained_data['occupancy_probabilities']
                # Convert probabilities back to logits using inverse sigmoid (logit function)
                # logit(p) = log(p / (1 - p)), with clamping to avoid numerical issues
                eps = 1e-8
                pretrained_probs_clamped = torch.clamp(pretrained_probs, eps, 1 - eps)
                pretrained_logits = torch.log(pretrained_probs_clamped / (1 - pretrained_probs_clamped))
                
                print(f"✅ Loaded probability grid, converted to logits")
                print(f"   Original probabilities range: [{pretrained_probs.min():.6f}, {pretrained_probs.max():.6f}]")
                print(f"   Converted logits range: [{pretrained_logits.min():.3f}, {pretrained_logits.max():.3f}]")
                
                # Print additional metadata if available
                if 'statistics' in pretrained_data:
                    stats = pretrained_data['statistics']
                    print(f"   High confidence ratio: {stats.get('high_confidence_ratio', 'N/A'):.4f}")
            else:
                raise ValueError(f"Unknown dict format in {pretrained_grid_path}")
        else:
            # Old format - raw tensor assumed to be logits
            pretrained_logits = pretrained_data
            print(f"✅ Loaded logits grid (legacy format)")
            print(f"   Logits range: [{pretrained_logits.min():.3f}, {pretrained_logits.max():.3f}]")
        
        # Validate grid shape
        if len(pretrained_logits.shape) != 3:
            raise ValueError(f"Expected 3D grid, got shape {pretrained_logits.shape}")
        
        # Check if resolution matches
        pretrained_resolution = pretrained_logits.shape
        if pretrained_resolution != self.resolution:
            print(f"⚠️  Warning: Pretrained grid resolution {pretrained_resolution} != expected {self.resolution}")
            print(f"    Updating resolution to match pretrained grid")
            self.resolution = pretrained_resolution
            
            # Recreate the parameter with correct size
            self.c_grid = nn.Parameter(torch.zeros(*pretrained_resolution, device=device))
        
        # Load the pretrained values
        with torch.no_grad():
            self.c_grid.data.copy_(pretrained_logits.to(device))
        
        print(f"✅ Loaded {pretrained_resolution[0]}³ confidence grid")
        
        # Convert to confidence and print final stats
        conf = torch.sigmoid(pretrained_logits)
        print(f"   Final confidence range: [{conf.min():.6f}, {conf.max():.6f}]")
        print(f"   Mean confidence: {conf.mean():.6f}")
        print(f"   High confidence voxels (>0.5): {(conf > 0.5).sum().item()}/{conf.numel()}")
        
        # Optionally freeze the grid for debugging/sanity check
        if self.freeze_pretrained:
            self.c_grid.requires_grad_(False)
            print(f"🔒 Confidence grid frozen (no gradients will be computed)")
        else:
            print(f"🔓 Confidence grid is trainable (gradients will be computed)")

        self.stencil_type = stencil_type

    def _build_kernels(self, device, stencil_type):
        """
        Builds finite difference kernels.
        """
        if stencil_type == 'central_difference_2nd_order':
            # Genus D* = 1 in the user request. Using corrected signs.
            coeffs = torch.tensor([-0.5, 0, 0.5], dtype=torch.float32, device=device)
            self.k_size = 3
        elif stencil_type == 'central_difference_4th_order':
            # Genus D* = 2 in the user request. Using corrected signs.
            coeffs = torch.tensor([1/12, -2/3, 0, 2/3, -1/12], dtype=torch.float32, device=device)
            self.k_size = 5
        else:
            raise ValueError(f"Unknown stencil_type: {stencil_type}")

        self.kernel_dx = coeffs.view(1, 1, 1, 1, self.k_size)
        self.kernel_dy = coeffs.view(1, 1, 1, self.k_size, 1)
        self.kernel_dz = coeffs.view(1, 1, self.k_size, 1, 1)

    def get_confidence(self):
        """Returns the confidence values by applying sigmoid to the grid logits."""
        return torch.sigmoid(self.c_grid)

    def _compute_world_space_intervals(self, grid_coords, axis):
        """
        Compute world-space intervals between adjacent grid points for non-uniform finite differences.
        
        Args:
            grid_coords: (D, H, W, 3) tensor of contracted space coordinates
            axis: 0=z, 1=y, 2=x - which axis to compute intervals for
            
        Returns:
            h_left, h_right: (D, H, W) tensors of world-space distances to left/right neighbors
        """
        D, H, W = self.resolution
        device = grid_coords.device
        
        # Create coordinate shifts for left and right neighbors
        shift = torch.zeros(3, device=device)
        if axis == 0:  # z-axis
            shift[0] = 2.0 / (D - 1)
        elif axis == 1:  # y-axis  
            shift[1] = 2.0 / (H - 1)
        elif axis == 2:  # x-axis
            shift[2] = 2.0 / (W - 1)
        
        # Get contracted coordinates of neighbors
        coord_left = grid_coords - shift
        coord_right = grid_coords + shift
        
        # Convert to world space using inverse contraction
        if self.non_spherical_contraction:
            world_center = coord.inv_contract_cubic(grid_coords)
            world_left = coord.inv_contract_cubic(coord_left)
            world_right = coord.inv_contract_cubic(coord_right)
        else:
            world_center = coord.inv_contract(grid_coords)
            world_left = coord.inv_contract(coord_left)
            world_right = coord.inv_contract(coord_right)
        
        # Compute world-space distances
        h_left = torch.norm(world_center - world_left, dim=-1)
        h_right = torch.norm(world_right - world_center, dim=-1)
        
        return h_left, h_right
    
    def _apply_sundqvist_veronis_stencil(self, conf_values, h_left, h_right, axis):
        """
        Apply the Sundqvist & Veronis (1970) finite difference formula for non-uniform grids.
        
        Formula: f'(x_i) = (f_{i+1} * h_{i-1}^2 - f_{i-1} * h_i^2 + f_i * (h_i^2 - h_{i-1}^2)) / (h_i * h_{i-1} * (h_i + h_{i-1}))
        
        Args:
            conf_values: (D, H, W) tensor of confidence values
            h_left, h_right: (D, H, W) tensors of world-space intervals
            axis: 0=z, 1=y, 2=x - which axis to compute derivative for
            
        Returns:
            grad: (D, H, W) tensor of gradients along the specified axis
        """
        D, H, W = self.resolution
        eps = 1e-8
        
        # Get neighbor values by padding and shifting
        if axis == 0:  # z-axis
            conf_left = F.pad(conf_values, (0, 0, 0, 0, 1, 0), mode='replicate')[:-1, :, :]
            conf_right = F.pad(conf_values, (0, 0, 0, 0, 0, 1), mode='replicate')[1:, :, :]
            h_left = h_left[1:-1] if D > 2 else h_left  # Exclude boundary points
            h_right = h_right[1:-1] if D > 2 else h_right
            conf_center = conf_values[1:-1] if D > 2 else conf_values
            conf_left = conf_left[1:-1] if D > 2 else conf_left
            conf_right = conf_right[1:-1] if D > 2 else conf_right
        elif axis == 1:  # y-axis
            conf_left = F.pad(conf_values, (0, 0, 1, 0), mode='replicate')[:, :-1, :]
            conf_right = F.pad(conf_values, (0, 0, 0, 1), mode='replicate')[:, 1:, :]
            h_left = h_left[:, 1:-1] if H > 2 else h_left
            h_right = h_right[:, 1:-1] if H > 2 else h_right
            conf_center = conf_values[:, 1:-1] if H > 2 else conf_values
            conf_left = conf_left[:, 1:-1] if H > 2 else conf_left
            conf_right = conf_right[:, 1:-1] if H > 2 else conf_right
        elif axis == 2:  # x-axis
            conf_left = F.pad(conf_values, (1, 0), mode='replicate')[:, :, :-1]
            conf_right = F.pad(conf_values, (0, 1), mode='replicate')[:, :, 1:]
            h_left = h_left[:, :, 1:-1] if W > 2 else h_left
            h_right = h_right[:, :, 1:-1] if W > 2 else h_right
            conf_center = conf_values[:, :, 1:-1] if W > 2 else conf_values
            conf_left = conf_left[:, :, 1:-1] if W > 2 else conf_left
            conf_right = conf_right[:, :, 1:-1] if W > 2 else conf_right
        
        # Apply Sundqvist & Veronis formula
        h_left_safe = torch.clamp(h_left, min=eps)
        h_right_safe = torch.clamp(h_right, min=eps)
        
        numerator = (conf_right * h_left_safe**2 - 
                    conf_left * h_right_safe**2 + 
                    conf_center * (h_right_safe**2 - h_left_safe**2))
        
        denominator = h_right_safe * h_left_safe * (h_right_safe + h_left_safe)
        denominator = torch.clamp(denominator, min=eps)
        
        grad_center = numerator / denominator
        
        # Handle boundaries with one-sided differences
        grad_full = torch.zeros_like(conf_values)
        
        if axis == 0 and D > 2:
            grad_full[1:-1, :, :] = grad_center
            # Boundary conditions
            grad_full[0, :, :] = (conf_values[1, :, :] - conf_values[0, :, :]) / h_right[0, :, :].clamp(min=eps)
            grad_full[-1, :, :] = (conf_values[-1, :, :] - conf_values[-2, :, :]) / h_left[-1, :, :].clamp(min=eps)
        elif axis == 1 and H > 2:
            grad_full[:, 1:-1, :] = grad_center
            grad_full[:, 0, :] = (conf_values[:, 1, :] - conf_values[:, 0, :]) / h_right[:, 0, :].clamp(min=eps)
            grad_full[:, -1, :] = (conf_values[:, -1, :] - conf_values[:, -2, :]) / h_left[:, -1, :].clamp(min=eps)
        elif axis == 2 and W > 2:
            grad_full[:, :, 1:-1] = grad_center
            grad_full[:, :, 0] = (conf_values[:, :, 1] - conf_values[:, :, 0]) / h_right[:, :, 0].clamp(min=eps)
            grad_full[:, :, -1] = (conf_values[:, :, -1] - conf_values[:, :, -2]) / h_left[:, :, -1].clamp(min=eps)
        else:
            # For small grids, use simple differences
            grad_full = grad_center
        
        return grad_full

    def _compute_gradient_non_uniform(self):
        """
        Compute gradients using non-uniform finite differences for contracted space.
        Uses the Sundqvist & Veronis (1970) formula to account for variable cell spacing.
        """
        D, H, W = self.resolution
        
        # Create coordinate grid in contracted space [-1, 1]
        z_coords = torch.linspace(-1, 1, D, device=self.c_grid.device)
        y_coords = torch.linspace(-1, 1, H, device=self.c_grid.device)
        x_coords = torch.linspace(-1, 1, W, device=self.c_grid.device)
        
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        grid_coords = torch.stack([zz, yy, xx], dim=-1)  # (D, H, W, 3)
        
        if self.binary_occupancy:
            # STE Implementation with non-uniform finite differences
            conf_continuous = self.get_confidence()
            
            with torch.no_grad():
                conf_binary = (torch.sigmoid(self.c_grid) > 0.5).float()
            
            self.binary_c_grid = conf_binary
            
            # Compute gradients for both continuous and binary versions
            grad_x_cont = self._compute_gradient_axis_non_uniform(conf_continuous, grid_coords, axis=2)
            grad_y_cont = self._compute_gradient_axis_non_uniform(conf_continuous, grid_coords, axis=1)
            grad_z_cont = self._compute_gradient_axis_non_uniform(conf_continuous, grid_coords, axis=0)
            
            grad_x_bin = self._compute_gradient_axis_non_uniform(conf_binary, grid_coords, axis=2)
            grad_y_bin = self._compute_gradient_axis_non_uniform(conf_binary, grid_coords, axis=1)
            grad_z_bin = self._compute_gradient_axis_non_uniform(conf_binary, grid_coords, axis=0)
            
            # Apply STE: binary values with continuous gradients
            grad_x = grad_x_bin.detach() + (grad_x_cont - grad_x_cont.detach())
            grad_y = grad_y_bin.detach() + (grad_y_cont - grad_y_cont.detach())
            grad_z = grad_z_bin.detach() + (grad_z_cont - grad_z_cont.detach())
        else:
            # Standard non-uniform finite differences on smooth sigmoid
            conf = self.get_confidence()
            self.binary_c_grid = None
            
            grad_x = self._compute_gradient_axis_non_uniform(conf, grid_coords, axis=2)
            grad_y = self._compute_gradient_axis_non_uniform(conf, grid_coords, axis=1) 
            grad_z = self._compute_gradient_axis_non_uniform(conf, grid_coords, axis=0)
        
        # Add batch dimensions for consistency with convolution-based method
        grad_x = grad_x.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        grad_y = grad_y.unsqueeze(0).unsqueeze(0)
        grad_z = grad_z.unsqueeze(0).unsqueeze(0)
        
        # Store gradient grid
        self.grad_c_grid = torch.cat([grad_x, grad_y, grad_z], dim=1)
    
    def _compute_gradient_axis_non_uniform(self, conf_values, grid_coords, axis):
        """
        Compute gradient along a specific axis using non-uniform finite differences.
        
        Args:
            conf_values: (D, H, W) tensor of confidence values
            grid_coords: (D, H, W, 3) tensor of contracted space coordinates  
            axis: 0=z, 1=y, 2=x - which axis to compute gradient for
            
        Returns:
            grad: (D, H, W) tensor of gradients along the specified axis
        """
        # Compute world-space intervals
        h_left, h_right = self._compute_world_space_intervals(grid_coords, axis)
        
        # Apply Sundqvist & Veronis formula
        grad = self._apply_sundqvist_veronis_stencil(conf_values, h_left, h_right, axis)
        
        return grad

    def compute_gradient(self):
        """
        Computes the gradient of the confidence grid using either uniform or non-uniform finite differences.
        If binary_occupancy is enabled, uses Straight-Through Estimator (STE) to 
        compute gradients from binary occupancy values while maintaining gradient flow.
        This is a pre-computation step that should be done once per training iteration.
        """
        # Check if we should use non-uniform finite differences
        if self.non_uniform_cells and not self.analytical_gradient:
            self._compute_gradient_non_uniform()
            return
            
        # Original uniform finite difference computation
        padding = (self.k_size - 1) // 2
        
        if self.binary_occupancy:
            # STE Implementation: Binary values in forward pass, continuous gradients in backward pass
            
            # Step 1: Compute continuous confidence field (for backward pass)
            conf_continuous = self.get_confidence().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            
            # Step 2: Compute binary confidence field (for forward pass)
            with torch.no_grad():
                conf_binary = (torch.sigmoid(self.c_grid) > 0.5).float().unsqueeze(0).unsqueeze(0)
            
            # Store binary occupancy grid for query method
            self.binary_c_grid = conf_binary.squeeze(0).squeeze(0)  # (D, H, W)
            
            # Step 3: Compute gradients from both versions
            
            # Continuous gradients (for backward pass)
            conf_padded_x_cont = F.pad(conf_continuous, (padding, padding, 0, 0, 0, 0), mode='replicate')
            grad_x_cont = F.conv3d(conf_padded_x_cont, self.kernel_dx, padding=0)
            
            conf_padded_y_cont = F.pad(conf_continuous, (0, 0, padding, padding, 0, 0), mode='replicate')
            grad_y_cont = F.conv3d(conf_padded_y_cont, self.kernel_dy, padding=0)
            
            conf_padded_z_cont = F.pad(conf_continuous, (0, 0, 0, 0, padding, padding), mode='replicate')
            grad_z_cont = F.conv3d(conf_padded_z_cont, self.kernel_dz, padding=0)
            
            # Binary gradients (for forward pass)
            conf_padded_x_bin = F.pad(conf_binary, (padding, padding, 0, 0, 0, 0), mode='replicate')
            grad_x_bin = F.conv3d(conf_padded_x_bin, self.kernel_dx, padding=0)
            
            conf_padded_y_bin = F.pad(conf_binary, (0, 0, padding, padding, 0, 0), mode='replicate')
            grad_y_bin = F.conv3d(conf_padded_y_bin, self.kernel_dy, padding=0)
            
            conf_padded_z_bin = F.pad(conf_binary, (0, 0, 0, 0, padding, padding), mode='replicate')
            grad_z_bin = F.conv3d(conf_padded_z_bin, self.kernel_dz, padding=0)
            
            # Step 4: Apply STE - combine binary values with continuous gradients
            grad_x = grad_x_bin.detach() + (grad_x_cont - grad_x_cont.detach())
            grad_y = grad_y_bin.detach() + (grad_y_cont - grad_y_cont.detach())
            grad_z = grad_z_bin.detach() + (grad_z_cont - grad_z_cont.detach())
            
            # Apply grid spacing correction
            if self.contraction_aware_gradients:
                grad_x, grad_y, grad_z = self._apply_contraction_aware_scaling(grad_x, grad_y, grad_z)
            else:
                # Original uniform spacing correction for [-1, 1] coordinates
                D, H, W = self.resolution
                scale_z = (D - 1) / 2.0  # 1 / grid_spacing_z
                scale_y = (H - 1) / 2.0  # 1 / grid_spacing_y  
                scale_x = (W - 1) / 2.0  # 1 / grid_spacing_x
                
                grad_x = grad_x * scale_x
                grad_y = grad_y * scale_y
                grad_z = grad_z * scale_z
            
            # Store gradient grid of shape (1, 3, D, H, W)
            self.grad_c_grid = torch.cat([grad_x, grad_y, grad_z], dim=1)
            
        else:
            # Original smooth sigmoid implementation
            # (D, H, W) -> (1, 1, D, H, W)
            conf = self.get_confidence().unsqueeze(0).unsqueeze(0) # Sigmoidal values
            self.binary_c_grid = None  # Not used in smooth mode

            # Manually pad and then convolve, as padding_mode is not supported with tuple-based padding in this PyTorch version.
            conf_padded_x = F.pad(conf, (padding, padding, 0, 0, 0, 0), mode='replicate')
            grad_x = F.conv3d(conf_padded_x, self.kernel_dx, padding=0)

            conf_padded_y = F.pad(conf, (0, 0, padding, padding, 0, 0), mode='replicate')
            grad_y = F.conv3d(conf_padded_y, self.kernel_dy, padding=0)

            conf_padded_z = F.pad(conf, (0, 0, 0, 0, padding, padding), mode='replicate')
            grad_z = F.conv3d(conf_padded_z, self.kernel_dz, padding=0)
            
            # Apply grid spacing correction
            if self.contraction_aware_gradients:
                grad_x, grad_y, grad_z = self._apply_contraction_aware_scaling(grad_x, grad_y, grad_z)
            else:
                # Original uniform spacing correction for [-1, 1] coordinates
                D, H, W = self.resolution
                scale_z = (D - 1) / 2.0  # 1 / grid_spacing_z
                scale_y = (H - 1) / 2.0  # 1 / grid_spacing_y  
                scale_x = (W - 1) / 2.0  # 1 / grid_spacing_x
                
                grad_x = grad_x * scale_x
                grad_y = grad_y * scale_y
                grad_z = grad_z * scale_z
            
            # Store gradient grid of shape (1, 3, D, H, W)
            self.grad_c_grid = torch.cat([grad_x, grad_y, grad_z], dim=1)

    def _apply_contraction_aware_scaling(self, grad_x, grad_y, grad_z):
        """
        Apply contraction-aware scaling to gradients computed via finite differences.
        
        The key insight: after spatial contraction, grid cells don't represent uniform
        world-space distances. We need to account for the contraction function's Jacobian.
        
        For spherical contraction (mip-NeRF 360):
        - Points inside unit sphere: mapped nearly linearly  
        - Points outside unit sphere: compressed into shell [1, 2]
        
        For cubic contraction (MeRF):
        - Points inside unit cube: mapped linearly
        - Points outside unit cube: compressed using L∞ norm scaling
        
        Args:
            grad_x, grad_y, grad_z: Raw gradients from finite differences
            
        Returns:
            Scaled gradients that account for non-uniform spacing
        """
        D, H, W = self.resolution
        
        # Create coordinate grids in [-1, 1] (contracted space)
        z_coords = torch.linspace(-1, 1, D, device=grad_x.device)
        y_coords = torch.linspace(-1, 1, H, device=grad_x.device)  
        x_coords = torch.linspace(-1, 1, W, device=grad_x.device)
        
        # Create meshgrid for all grid points
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        grid_coords = torch.stack([zz, yy, xx], dim=-1)  # (D, H, W, 3)
        
        if self.non_spherical_contraction:
            # Cubic contraction scaling (MeRF) following coord.py formulation
            # contract𝜋(x)𝑗 = {
            #   𝑥𝑗                                if ∥x∥∞≤1
            #   𝑥𝑗/∥x∥∞                          if 𝑥𝑗 ≠ ∥x∥∞> 1  
            #   (2 - 1/|𝑥𝑗|) * 𝑥𝑗/|𝑥𝑗|         if 𝑥𝑗= ∥x∥∞> 1
            # }
            
            eps = 1e-8
            abs_coords = torch.abs(grid_coords)  # (D, H, W, 3)
            coord_norm_inf = torch.max(abs_coords, dim=-1, keepdim=True)[0]  # (D, H, W, 1)
            
            # Case 1: Inside unit cube ∥x∥∞≤1
            mask_inside = coord_norm_inf <= 1.0
            
            # Case 2 & 3: Outside unit cube ∥x∥∞> 1
            # Determine which coordinates are the max coordinate
            max_coord_mask = (abs_coords == coord_norm_inf) & (coord_norm_inf > 1.0)  # (D, H, W, 3)
            
            # Safe clamping for numerical stability
            coord_norm_inf_safe = torch.clamp(coord_norm_inf, min=eps)
            abs_coords_safe = torch.clamp(abs_coords, min=eps)
            
            # Scaling calculations for finite differences in contracted space:
            # We need the inverse Jacobian to convert contracted-space gradients to world-space
            # Case 1: df/dx = 1 → scaling = 1/1 = 1
            # Case 2: df/dx ≈ 1/∥x∥∞ → scaling = 1/(1/∥x∥∞) = ∥x∥∞  
            # Case 3: df/dx = 1/|𝑥𝑗|² → scaling = 1/(1/|𝑥𝑗|²) = |𝑥𝑗|²
            
            scale_identity = torch.ones_like(abs_coords)  # Case 1: identity
            scale_non_max = coord_norm_inf_safe.expand_as(abs_coords)  # Case 2: ||x||∞
            scale_max = abs_coords_safe ** 2  # Case 3: |x_j|²
            
            # Apply the appropriate scaling based on the region
            scaling_factor_per_coord = torch.where(
                mask_inside.expand_as(abs_coords),
                scale_identity,  # Inside: identity scaling
                torch.where(
                    max_coord_mask,
                    scale_max,      # Max coordinate outside: 1/|x_j|²
                    scale_non_max   # Non-max coordinate outside: 1/∥x∥∞
                )
            )  # (D, H, W, 3)
            
            # For gradient scaling, use geometric mean of coordinate scalings
            # This preserves the relative scaling relationships better than arithmetic mean
            scaling_factor_log = torch.log(scaling_factor_per_coord + eps)
            scaling_factor = torch.exp(torch.mean(scaling_factor_log, dim=-1, keepdim=True))  # (D, H, W, 1)
            
        else:
            # Spherical contraction scaling (mip-NeRF 360)
            # Use L2 norm
            coord_norm = torch.norm(grid_coords, dim=-1, keepdim=True)  # (D, H, W, 1)
            
            # For spherical contraction: f(x) = x if |x|≤1, else (2√|x| - 1)/|x|² * x
            # The dominant scaling factor for |x| > 1 is approximately |x|²
            # We need the inverse Jacobian: scaling ≈ |x|² for contracted regions
            eps = 1e-8
            coord_norm_safe = torch.clamp(coord_norm, min=eps)
            
            scaling_factor = torch.where(
                coord_norm <= 1.0,
                torch.ones_like(coord_norm),  # Linear region: no scaling needed
                coord_norm_safe ** 2  # Contracted region: |x|² scaling
            )
        
        # Apply scaling to gradients
        # Note: This is a simplified approach. A full solution would compute the exact Jacobian.
        scaling_factor = scaling_factor.squeeze(-1)  # (D, H, W)
        scaling_factor = scaling_factor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        
        # Scale gradients by the contraction factor
        grad_x_scaled = grad_x * scaling_factor
        grad_y_scaled = grad_y * scaling_factor  
        grad_z_scaled = grad_z * scaling_factor
        
        # Still apply the basic grid spacing correction for [-1, 1] coordinates
        base_scale_z = (D - 1) / 2.0
        base_scale_y = (H - 1) / 2.0
        base_scale_x = (W - 1) / 2.0
        
        grad_x_final = grad_x_scaled * base_scale_x
        grad_y_final = grad_y_scaled * base_scale_y
        grad_z_final = grad_z_scaled * base_scale_z
        
        return grad_x_final, grad_y_final, grad_z_final

    def get_analytical_gradient_at_points(self, points_normalized):
        """
        Compute exact analytical gradients at specified points using explicit trilinear interpolation.
        This avoids the grid_sample autograd limitation by implementing trilinear interpolation manually
        and computing its analytical derivatives.
        
        Args:
            points_normalized: (N, 3) tensor of points in [-1, 1] range
            
        Returns:
            analytical_grad: (N, 3) tensor of gradients ∇occupancy(p) 
            occupancy: (N, 1) tensor of sigmoid(logits(p)) values
        """
        N = points_normalized.shape[0]
        D, H, W = self.resolution
        
        # Convert [-1, 1] coordinates to grid indices [0, D-1], [0, H-1], [0, W-1]
        # grid_sample uses: coord = -1 maps to index 0, coord = 1 maps to index D-1
        coords_grid = (points_normalized + 1.0) * 0.5  # [-1, 1] -> [0, 1]
        coords_grid[:, 0] *= (D - 1)  # z coordinate
        coords_grid[:, 1] *= (H - 1)  # y coordinate  
        coords_grid[:, 2] *= (W - 1)  # x coordinate
        
        # Get integer and fractional parts for trilinear interpolation
        coords_floor = torch.floor(coords_grid).long()
        coords_frac = coords_grid - coords_floor.float()
        
        # Clamp to valid grid range
        z0 = torch.clamp(coords_floor[:, 0], 0, D - 2)
        y0 = torch.clamp(coords_floor[:, 1], 0, H - 2)
        x0 = torch.clamp(coords_floor[:, 2], 0, W - 2)
        z1 = z0 + 1
        y1 = y0 + 1
        x1 = x0 + 1
        
        # Get fractional coordinates for interpolation weights
        dz = coords_frac[:, 0]  # [0, 1]
        dy = coords_frac[:, 1]  # [0, 1]
        dx = coords_frac[:, 2]  # [0, 1]
        
        # Get the 8 corner values of the grid cube
        # Access logits directly from the confidence grid
        c_grid = self.c_grid  # (D, H, W)
        
        v000 = c_grid[z0, y0, x0]  # (N,)
        v001 = c_grid[z0, y0, x1]
        v010 = c_grid[z0, y1, x0]
        v011 = c_grid[z0, y1, x1]
        v100 = c_grid[z1, y0, x0]
        v101 = c_grid[z1, y0, x1]
        v110 = c_grid[z1, y1, x0]
        v111 = c_grid[z1, y1, x1]
        
        # Trilinear interpolation formula
        # f(x,y,z) = v000*(1-dx)*(1-dy)*(1-dz) + v001*dx*(1-dy)*(1-dz) + ... (8 terms)
        
        interpolated_logits = (
            v000 * (1 - dx) * (1 - dy) * (1 - dz) +
            v001 * dx * (1 - dy) * (1 - dz) +
            v010 * (1 - dx) * dy * (1 - dz) +
            v011 * dx * dy * (1 - dz) +
            v100 * (1 - dx) * (1 - dy) * dz +
            v101 * dx * (1 - dy) * dz +
            v110 * (1 - dx) * dy * dz +
            v111 * dx * dy * dz
        )
        
        # Compute analytical derivatives of trilinear interpolation
        # ∂f/∂x = (v001 - v000)*(1-dy)*(1-dz) + (v011 - v010)*dy*(1-dz) + 
        #         (v101 - v100)*(1-dy)*dz + (v111 - v110)*dy*dz
        grad_dx = (
            (v001 - v000) * (1 - dy) * (1 - dz) +
            (v011 - v010) * dy * (1 - dz) +
            (v101 - v100) * (1 - dy) * dz +
            (v111 - v110) * dy * dz
        )
        
        grad_dy = (
            (v010 - v000) * (1 - dx) * (1 - dz) +
            (v011 - v001) * dx * (1 - dz) +
            (v110 - v100) * (1 - dx) * dz +
            (v111 - v101) * dx * dz
        )
        
        grad_dz = (
            (v100 - v000) * (1 - dx) * (1 - dy) +
            (v101 - v001) * dx * (1 - dy) +
            (v110 - v010) * (1 - dx) * dy +
            (v111 - v011) * dx * dy
        )
        
        # Apply chain rule for coordinate transformation
        # We interpolated w.r.t grid coordinates, but need gradient w.r.t normalized coordinates
        # d_logits/d_normalized = d_logits/d_grid * d_grid/d_normalized
        # d_grid/d_normalized = (resolution - 1) / 2 for each axis
        scale_z = (D - 1) / 2.0
        scale_y = (H - 1) / 2.0  
        scale_x = (W - 1) / 2.0
        
        grad_logits_normalized = torch.stack([
            grad_dz * scale_z,  # z gradient
            grad_dy * scale_y,  # y gradient  
            grad_dx * scale_x   # x gradient
        ], dim=1)  # (N, 3)
        
        # Apply sigmoid to get occupancy values
        occupancy = torch.sigmoid(interpolated_logits).unsqueeze(1)  # (N, 1)
        
        # Apply chain rule for sigmoid: ∇occupancy = sigmoid'(logits) * ∇logits
        sigmoid_derivative = occupancy * (1 - occupancy)  # (N, 1)
        analytical_grad = sigmoid_derivative * grad_logits_normalized  # (N, 3)
        
        return analytical_grad, occupancy
    
    def query(self, points):
        """
        Interpolates the confidence and its gradient at given points.
        When binary_occupancy is enabled, returns binary occupancy values (0 or 1).
        
        Args:
            points: (N, 3) tensor of points in the range [-1, 1].
        Returns:
            sampled_conf: (N, 1) tensor of confidence/occupancy values.
            sampled_grad: (N, 3) tensor of gradient values.
        """
        # `grid_sample` expects coordinates in [-1, 1]
        # points should be (N, 1, 1, 1, 3) for 3D grid_sample
        points_for_grid_sample = points.view(1, -1, 1, 1, 3)

        # Interpolate confidence/occupancy
        if self.binary_occupancy and self.binary_c_grid is not None:
            # Use STE: binary values in forward pass, continuous gradients in backward pass
            binary_grid = self.binary_c_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            continuous_grid = self.get_confidence().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            
            # Sample from both grids
            sampled_binary = F.grid_sample(binary_grid, points_for_grid_sample, align_corners=True, mode='bilinear')
            sampled_continuous = F.grid_sample(continuous_grid, points_for_grid_sample, align_corners=True, mode='bilinear')
            
            # Apply STE: binary values + (continuous - continuous.detach())
            sampled_conf = sampled_binary.detach() + (sampled_continuous - sampled_continuous.detach())
            sampled_conf = sampled_conf.view(-1, 1)  # (N, 1)
        else:
            # Use continuous confidence values
            conf_grid = self.get_confidence().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            sampled_conf = F.grid_sample(conf_grid, points_for_grid_sample, align_corners=True, mode='bilinear')
            sampled_conf = sampled_conf.view(-1, 1) # (N, 1)

        # Compute gradient using either stencil-based or analytical method
        if self.analytical_gradient:
            # Use analytical gradient computation (autograd-based)
            analytical_grad, analytical_occupancy = self.get_analytical_gradient_at_points(points)
            sampled_grad = analytical_grad  # (N, 3)
            
            # For consistency, we could optionally use the analytical occupancy instead of sampled_conf
            # but for now we keep the existing confidence computation to maintain compatibility
            
        else:
            # Use pre-computed stencil-based gradients
            if self.grad_c_grid is None:
                raise RuntimeError("Gradient must be computed before querying when using stencil-based gradients.")
                
            # self.grad_c_grid is (1, 3, D, H, W)
            sampled_grad = F.grid_sample(self.grad_c_grid, points_for_grid_sample, align_corners=True, mode='bilinear')
            
            # (1, 3, N, 1, 1) -> (N, 3)
            sampled_grad = sampled_grad.view(3, -1).permute(1, 0)
        
        return sampled_conf, sampled_grad

    def get_regularization_loss(self):
        """Computes the binarity-promoting regularization loss."""
        C = self.get_confidence()
        loss_reg = torch.mean(-C * torch.log(C + 1e-8) - (1-C) * torch.log(1-C + 1e-8))
        return loss_reg 
    
    # ADMM Pruner Methods
    def get_current_sparsity(self):
        """
        Compute the current sparsity of the confidence grid.
        Returns the L1 norm of sigmoid-activated grid (number of "active" voxels).
        """
        if not self.use_admm_pruner:
            raise RuntimeError("ADMM pruner not enabled. Set use_admm_pruner=True in initialization.")
        return torch.sigmoid(self.c_grid).sum()
    
    def get_sparsity_fraction(self):
        """Get the current sparsity as a fraction of total grid elements."""
        if not self.use_admm_pruner:
            raise RuntimeError("ADMM pruner not enabled. Set use_admm_pruner=True in initialization.")
        return self.get_current_sparsity() / self.total_grid_elements
    
    def get_admm_loss_components(self, sparsity_constraint_absolute, penalty_rho):
        """
        Compute ADMM augmented Lagrangian loss components.
        
        Args:
            sparsity_constraint_absolute: Absolute sparsity constraint (C * total_elements)
            penalty_rho: Quadratic penalty coefficient
            
        Returns:
            dict containing:
                - current_sparsity: Current L1 norm of sigmoid(c_grid)
                - constraint_violation: current_sparsity - sparsity_constraint_absolute
                - lagrangian_term: γ * constraint_violation
                - penalty_term: (ρ/2) * constraint_violation^2
                - total_admm_loss: lagrangian_term + penalty_term
        """
        if not self.use_admm_pruner:
            raise RuntimeError("ADMM pruner not enabled. Set use_admm_pruner=True in initialization.")
        
        # Compute current sparsity (L1 norm of sigmoid activations)
        current_sparsity = self.get_current_sparsity()
        
        # Constraint violation: g(x) = ||sigmoid(c_grid)||_1 - C
        constraint_violation = current_sparsity - sparsity_constraint_absolute
        
        # Lagrangian term: γ * g(x)
        # Detach dual_variable to treat it as constant during primal update
        lagrangian_term = self.dual_variable.detach() * constraint_violation
        
        # Quadratic penalty term: (ρ/2) * g(x)^2
        penalty_term = (penalty_rho / 2) * (constraint_violation ** 2)
        
        # Total ADMM loss contribution
        total_admm_loss = lagrangian_term + penalty_term
        
        return {
            'current_sparsity': current_sparsity,
            'constraint_violation': constraint_violation,
            'lagrangian_term': lagrangian_term,
            'penalty_term': penalty_term,
            'total_admm_loss': total_admm_loss,
        }
    
    def update_dual_variable(self, sparsity_constraint_absolute, dual_lr):
        """
        Update the dual variable γ using gradient ascent.
        This should be called after the primal optimization step.
        
        Args:
            sparsity_constraint_absolute: Absolute sparsity constraint (C * total_elements)
            dual_lr: Learning rate for dual variable update
        """
        if not self.use_admm_pruner:
            raise RuntimeError("ADMM pruner not enabled. Set use_admm_pruner=True in initialization.")
        
        with torch.no_grad():
            # Current sparsity (detached to avoid gradient computation)
            current_sparsity = self.get_current_sparsity().detach()
            
            # Gradient of Lagrangian w.r.t. γ is just the constraint violation
            grad_dual = current_sparsity - sparsity_constraint_absolute
            
            # Gradient ascent update: γ += lr * grad
            self.dual_variable.add_(dual_lr * grad_dual)
            
            # Enforce non-negativity constraint: γ = max(0, γ)
            self.dual_variable.clamp_(min=0.0)
    
    def get_admm_metrics(self, sparsity_constraint_fraction):
        """
        Get comprehensive ADMM metrics for logging.
        
        Args:
            sparsity_constraint_fraction: Target sparsity fraction
            
        Returns:
            dict with metrics for logging
        """
        if not self.use_admm_pruner:
            return {}
        
        with torch.no_grad():
            current_sparsity = self.get_current_sparsity()
            current_fraction = current_sparsity / self.total_grid_elements
            
            # Additional useful metrics
            confidence_sigmoid = torch.sigmoid(self.c_grid)
            
            metrics = {
                'admm/current_sparsity_absolute': current_sparsity.item(),
                'admm/current_sparsity_fraction': current_fraction.item(),
                'admm/target_sparsity_fraction': sparsity_constraint_fraction,
                'admm/sparsity_error': (current_fraction - sparsity_constraint_fraction).item(),
                'admm/dual_variable': self.dual_variable.item(),
                'admm/confidence_mean': confidence_sigmoid.mean().item(),
                'admm/confidence_std': confidence_sigmoid.std().item(),
                'admm/confidence_min': confidence_sigmoid.min().item(),
                'admm/confidence_max': confidence_sigmoid.max().item(),
                'admm/high_confidence_voxels_01': (confidence_sigmoid > 0.1).float().mean().item(),
                'admm/high_confidence_voxels_05': (confidence_sigmoid > 0.5).float().mean().item(),
                'admm/high_confidence_voxels_09': (confidence_sigmoid > 0.9).float().mean().item(),
            }
            
            return metrics 