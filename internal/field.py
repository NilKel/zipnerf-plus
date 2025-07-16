import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class ConfidenceField(nn.Module):
    def __init__(self, resolution=(128, 128, 128), init_val=-0.5, init_rand_mag=1.0, device='cuda', 
                 stencil_type='central_difference_4th_order', pretrained_grid_path=None, freeze_pretrained=True,
                 binary_occupancy=False, analytical_gradient=False):
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
        """
        super().__init__()
        self.resolution = resolution
        self.pretrained_grid_path = pretrained_grid_path
        self.freeze_pretrained = freeze_pretrained
        self.binary_occupancy = binary_occupancy
        self.analytical_gradient = analytical_gradient
        
        # Initialize logits to be slightly negative on average
        self.c_grid = nn.Parameter(torch.randn(*resolution, device=device) * init_rand_mag + init_val)
        
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

    def compute_gradient(self):
        """
        Computes the gradient of the confidence grid using 3D convolution.
        If binary_occupancy is enabled, uses Straight-Through Estimator (STE) to 
        compute gradients from binary occupancy values while maintaining gradient flow.
        This is a pre-computation step that should be done once per training iteration.
        """
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
            
            # Apply grid spacing correction for normalized coordinates [-1, 1]
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
            conf = self.get_confidence().unsqueeze(0).unsqueeze(0)
            self.binary_c_grid = None  # Not used in smooth mode

            # Manually pad and then convolve, as padding_mode is not supported with tuple-based padding in this PyTorch version.
            conf_padded_x = F.pad(conf, (padding, padding, 0, 0, 0, 0), mode='replicate')
            grad_x = F.conv3d(conf_padded_x, self.kernel_dx, padding=0)

            conf_padded_y = F.pad(conf, (0, 0, padding, padding, 0, 0), mode='replicate')
            grad_y = F.conv3d(conf_padded_y, self.kernel_dy, padding=0)

            conf_padded_z = F.pad(conf, (0, 0, 0, 0, padding, padding), mode='replicate')
            grad_z = F.conv3d(conf_padded_z, self.kernel_dz, padding=0)
            
            # Apply grid spacing correction for normalized coordinates [-1, 1]
            D, H, W = self.resolution
            scale_z = (D - 1) / 2.0  # 1 / grid_spacing_z
            scale_y = (H - 1) / 2.0  # 1 / grid_spacing_y  
            scale_x = (W - 1) / 2.0  # 1 / grid_spacing_x
            
            grad_x = grad_x * scale_x
            grad_y = grad_y * scale_y
            grad_z = grad_z * scale_z
            
            # Store gradient grid of shape (1, 3, D, H, W)
            self.grad_c_grid = torch.cat([grad_x, grad_y, grad_z], dim=1)

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