import torch
import torch.nn as nn


class FEncoder(torch.nn.Module):
    """Classic Positional Encoder for NeRF."""
    
    def __init__(self, num_dims: int = 3, num_freqs: int = 10, log_sampling: bool = True):
        super().__init__()
        
        self.num_dims = num_dims
        self.num_freqs = num_freqs
        self.output_dim = num_dims * 2 * num_freqs
        
        # Create frequency bands
        if log_sampling:
            freqs = 2. ** torch.linspace(0., num_freqs - 1, num_freqs)
        else:
            freqs = torch.linspace(1., 2.**(num_freqs - 1), num_freqs)
        
        # Register as buffer so it moves with the model to different devices
        self.register_buffer('freqs', freqs)
    
    def forward(self, x):
        """
        Forward pass of the positional encoder.
        
        Args:
            x: Input coordinates of shape [..., 3]
            
        Returns:
            F: Positional encoding of shape [..., output_dim]
        """
        # Prepare input for broadcasting: [..., 3, num_freqs]
        x_proj = x.unsqueeze(-1) * self.freqs
        
        # Concatenate sine and cosine: [..., 3, 2 * num_freqs]
        encoded = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        # Reshape to flatten the last two dimensions: [..., output_dim]
        encoded = encoded.view(*x.shape[:-1], self.output_dim)
        
        return encoded


class PEncoder(torch.nn.Module):
    """Tensor Potential Encoder whose divergence equals FEncoder output."""
    
    def __init__(self, num_dims: int = 3, num_freqs: int = 10, log_sampling: bool = True):
        super().__init__()
        
        # Instantiate internal FEncoder to reuse frequency bands and output dimension
        self.f_encoder = FEncoder(num_dims, num_freqs, log_sampling)
        self.output_dim_F = self.f_encoder.output_dim
        self.freqs = self.f_encoder.freqs
        self.num_freqs = num_freqs
    
    def forward(self, x):
        """
        Forward pass of the tensor potential encoder using distributed potentials.
        
        Args:
            x: Input coordinates of shape [..., 3]
            
        Returns:
            G: Tensor potential of shape [..., output_dim_F, 3]
        """
        # Get coordinates for each dimension (keep last dim for broadcasting)
        x_coord = x[..., 0:1]  # [..., 1]
        y_coord = x[..., 1:2]  # [..., 1]
        z_coord = x[..., 2:3]  # [..., 1]
        
        G_rows = []
        
        # First, create all x-coordinate potentials (sin and cos for each frequency)
        for l in range(self.num_freqs):
            k = self.freqs[l]
            
            # For f = sin(kx), G = [-(1/3k)cos(kx), (y/3)sin(kx), (z/3)sin(kx)]
            sin_potential_x = torch.cat([
                -(1/(3*k)) * torch.cos(k * x_coord), 
                (y_coord/3) * torch.sin(k * x_coord), 
                (z_coord/3) * torch.sin(k * x_coord)
            ], dim=-1)
            
            # For f = cos(kx), G = [(1/3k)sin(kx), (y/3)cos(kx), (z/3)cos(kx)]
            cos_potential_x = torch.cat([
                (1/(3*k)) * torch.sin(k * x_coord), 
                (y_coord/3) * torch.cos(k * x_coord), 
                (z_coord/3) * torch.cos(k * x_coord)
            ], dim=-1)
            
            G_rows.append(sin_potential_x)
            G_rows.append(cos_potential_x)
        
        # Then, create all y-coordinate potentials (sin and cos for each frequency)
        for l in range(self.num_freqs):
            k = self.freqs[l]
            
            # For f = sin(ky), G = [(x/3)sin(ky), -(1/3k)cos(ky), (z/3)sin(ky)]
            sin_potential_y = torch.cat([
                (x_coord/3) * torch.sin(k * y_coord), 
                -(1/(3*k)) * torch.cos(k * y_coord), 
                (z_coord/3) * torch.sin(k * y_coord)
            ], dim=-1)
            
            # For f = cos(ky), G = [(x/3)cos(ky), (1/3k)sin(ky), (z/3)cos(ky)]
            cos_potential_y = torch.cat([
                (x_coord/3) * torch.cos(k * y_coord), 
                (1/(3*k)) * torch.sin(k * y_coord), 
                (z_coord/3) * torch.cos(k * y_coord)
            ], dim=-1)
            
            G_rows.append(sin_potential_y)
            G_rows.append(cos_potential_y)
        
        # Finally, create all z-coordinate potentials (sin and cos for each frequency)
        for l in range(self.num_freqs):
            k = self.freqs[l]
            
            # For f = sin(kz), G = [(x/3)sin(kz), (y/3)sin(kz), -(1/3k)cos(kz)]
            sin_potential_z = torch.cat([
                (x_coord/3) * torch.sin(k * z_coord), 
                (y_coord/3) * torch.sin(k * z_coord), 
                -(1/(3*k)) * torch.cos(k * z_coord)
            ], dim=-1)
            
            # For f = cos(kz), G = [(x/3)cos(kz), (y/3)cos(kz), (1/3k)sin(kz)]
            cos_potential_z = torch.cat([
                (x_coord/3) * torch.cos(k * z_coord), 
                (y_coord/3) * torch.cos(k * z_coord), 
                (1/(3*k)) * torch.sin(k * z_coord)
            ], dim=-1)
            
            G_rows.append(sin_potential_z)
            G_rows.append(cos_potential_z)
        
        # Stack all potential vectors along a new dimension
        # This creates shape [..., output_dim_F, 3]
        G = torch.stack(G_rows, dim=-2)
        
        return G


class PositionalEncoder(torch.nn.Module):
    """Main switch module to easily switch between FEncoder and PEncoder."""
    
    def __init__(self, encoding_type: str, num_dims: int = 3, num_freqs: int = 10, log_sampling: bool = True):
        super().__init__()
        
        self.encoding_type = encoding_type
        
        if encoding_type == 'F_ENCODER':
            self.encoder = FEncoder(num_dims, num_freqs, log_sampling)
        elif encoding_type == 'P_ENCODER':
            self.encoder = PEncoder(num_dims, num_freqs, log_sampling)
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}. Must be 'F_ENCODER' or 'P_ENCODER'.")
    
    @property
    def output_dim(self):
        """Return the output dimension of the selected encoder."""
        if self.encoding_type == 'F_ENCODER':
            return self.encoder.output_dim
        elif self.encoding_type == 'P_ENCODER':
            return self.encoder.output_dim_F
        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")
    
    def forward(self, x):
        """
        Forward pass through the selected encoder.
        
        Args:
            x: Input coordinates
            
        Returns:
            Encoded output from the selected encoder
        """
        return self.encoder(x) 