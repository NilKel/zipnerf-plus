import torch
from torch import nn
import torch.nn.functional as F


class TriMipEncoding(nn.Module):
    def __init__(
        self,
        n_levels: int,
        plane_size: int,
        feature_dim: int,
        include_xyz: bool = False,
    ):
        super(TriMipEncoding, self).__init__()
        self.n_levels = n_levels
        self.plane_size = plane_size
        self.feature_dim = feature_dim
        self.include_xyz = include_xyz

        self.register_parameter(
            "fm",
            nn.Parameter(torch.zeros(3, plane_size, plane_size, feature_dim)),
        )
        self.init_parameters()
        self.dim_out = (
            self.feature_dim * 3 + 3 if include_xyz else self.feature_dim * 3
        )

    def init_parameters(self) -> None:
        # Important for performance
        nn.init.uniform_(self.fm, -1e-2, 1e-2)

    def forward(self, x, level):
        # x in [0,1], level in [0,max_level]
        # x is Nx3, level is Nx1
        if 0 == x.shape[0]:
            return torch.zeros([x.shape[0], self.feature_dim * 3]).to(x)
        
        # Ensure x is 2D [N, 3]
        if x.dim() > 2:
            original_shape = x.shape
            x = x.view(-1, x.shape[-1])  # Flatten to [N, 3]
        else:
            original_shape = None
            
        # Process each plane separately using simple grid_sample
        features = []
        plane_coords = [
            x[:, [1, 2]],  # YZ plane
            x[:, [0, 2]],  # XZ plane  
            x[:, [0, 1]],  # XY plane
        ]
        
        for i, coords in enumerate(plane_coords):
            # Convert [0,1] coordinates to [-1,1] for grid_sample
            coords_grid = coords * 2.0 - 1.0  # [N, 2]
            
            # Reshape for grid_sample: [N, 2] -> [1, 1, N, 2]
            coords_reshaped = coords_grid.unsqueeze(0).unsqueeze(1)  # [1, 1, N, 2]
            
            # Get plane features: [H, W, C] -> [C, H, W] -> [1, C, H, W]
            plane = self.fm[i].permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            
            # Sample from plane using bilinear interpolation
            plane_features = F.grid_sample(
                plane, coords_reshaped, 
                mode='bilinear', padding_mode='border', align_corners=True
            )  # [1, C, 1, N]
            
            # Reshape to [N, C]
            plane_features = plane_features.squeeze(0).squeeze(1).permute(1, 0)  # [N, C]
            features.append(plane_features)
        
        # Concatenate features from all three planes
        enc = torch.cat(features, dim=-1)  # [N, 3*C]
        
        # Restore original shape if needed
        if original_shape is not None and original_shape != enc.shape:
            target_shape = original_shape[:-1] + (enc.shape[-1],)
            enc = enc.view(target_shape)
        
        if self.include_xyz:
            if original_shape is not None:
                x = x.view(original_shape)
            enc = torch.cat([x, enc], dim=-1)
        return enc 