#!/usr/bin/env python3
"""
Analytical Oracle Functions for Vector Potential Testing

This module implements the perfect analytical functions for the controlled experiment:
- F(p) = 1 (constant scalar field to integrate)  
- G(p) = p/3 (vector potential with div(G) = 1 = F)
- O(p) = 1 if abs(||p|| - r) <= epsilon else 0 (hollow sphere occupancy)
- ∇O(p) = normalize(p) if on sphere surface, else (0,0,0) (perfect normals)

The goal is to test if the feature V_feat = -G⋅∇O can be used by an MLP to reconstruct the scene.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AnalyticalOracles:
    """
    Container for all analytical oracle functions.
    
    These functions provide ground truth values for:
    1. F(p) - the function to integrate (constant 1)
    2. G(p) - the vector potential (p/3) 
    3. O(p) - the occupancy function (hollow sphere)
    4. ∇O(p) - the occupancy gradient (sphere normals)
    """
    
    def __init__(self, sphere_radius=1.0, sphere_center=None, epsilon=0.05):
        """
        Initialize analytical oracles.
        
        Args:
            sphere_radius: Radius of the hollow sphere
            sphere_center: Center of the sphere [x, y, z]
            epsilon: Thickness of the hollow sphere shell
        """
        if sphere_center is None:
            sphere_center = [0.0, 0.0, 0.0]
            
        self.sphere_radius = sphere_radius
        self.sphere_center = torch.tensor(sphere_center, dtype=torch.float32)
        self.epsilon = epsilon
        
        print(f"🔮 Analytical Oracles initialized:")
        print(f"   Sphere radius: {sphere_radius}")
        print(f"   Sphere center: {sphere_center}")
        print(f"   Shell thickness: {epsilon}")
    
    def analytical_f_query(self, points):
        """
        The function to integrate: F(p) = 1 (constant scalar field).
        
        Args:
            points: (..., 3) tensor of 3D points
            
        Returns:
            f_values: (..., D) tensor where D is the feature dimension
                     All values are 1.0
        """
        batch_shape = points.shape[:-1]
        # Return constant 1 for all points and all feature dimensions
        # The feature dimension D will be determined by the calling context
        return torch.ones(batch_shape, device=points.device, dtype=points.dtype)
    
    def analytical_g_query(self, points, feature_dim=1):
        """
        The vector potential: G(p) = p/3.
        
        This satisfies div(G) = ∂/∂x(x/3) + ∂/∂y(y/3) + ∂/∂z(z/3) = 1 = F(p)
        
        Args:
            points: (..., 3) tensor of 3D points
            feature_dim: Number of feature channels D
            
        Returns:
            g_values: (..., D, 3) tensor of vector potentials
        """
        batch_shape = points.shape[:-1]
        
        # G(p) = p/3 for each point
        g_base = points / 3.0  # (..., 3)
        
        # Expand to multi-dimensional features
        # All D feature channels are identical per the experiment design
        g_values = g_base.unsqueeze(-2).expand(*batch_shape, feature_dim, 3)
        
        return g_values
    
    def analytical_o_query(self, points):
        """
        The occupancy function: O(p) = 1 if abs(||p|| - r) <= epsilon else 0.
        
        This defines a hollow sphere shell with thickness 2*epsilon.
        
        Args:
            points: (..., 3) tensor of 3D points
            
        Returns:
            occupancy: (..., 1) tensor of occupancy values {0, 1}
        """
        batch_shape = points.shape[:-1]
        
        # Move sphere center to same device as points
        sphere_center = self.sphere_center.to(points.device)
        
        # Compute distance from sphere center
        distances = torch.norm(points - sphere_center, dim=-1)
        
        # Compute signed distance function (SDF)
        sdf = distances - self.sphere_radius
        
        # Occupancy: 1 if within shell thickness, 0 otherwise
        occupancy = (torch.abs(sdf) <= self.epsilon).float()
        
        return occupancy.unsqueeze(-1)  # (..., 1)
    
    def analytical_grad_o_query(self, points):
        """
        The occupancy gradient: ∇O(p) = normalize(p) if on sphere surface, else (0,0,0).
        
        For points on the sphere shell, this gives the outward-pointing unit normal.
        For points not on the shell, the gradient is zero.
        
        Args:
            points: (..., 3) tensor of 3D points
            
        Returns:
            gradient: (..., 3) tensor of gradient vectors
        """
        batch_shape = points.shape[:-1]
        
        # Move sphere center to same device as points
        sphere_center = self.sphere_center.to(points.device)
        
        # Get occupancy to determine which points are on the shell
        occupancy = self.analytical_o_query(points).squeeze(-1)  # (...,)
        
        # Compute vectors from sphere center to points
        vectors = points - sphere_center  # (..., 3)
        
        # Compute distances
        distances = torch.norm(vectors, dim=-1, keepdim=True)  # (..., 1)
        
        # Compute unit normals (avoiding division by zero)
        epsilon_dist = 1e-8
        normals = vectors / (distances + epsilon_dist)  # (..., 3)
        
        # Zero out gradients for points not on the sphere shell
        occupancy_mask = occupancy.unsqueeze(-1)  # (..., 1)
        gradient = normals * occupancy_mask  # (..., 3)
        
        return gradient
    
    def analytical_div_g_query(self, points, feature_dim=1):
        """
        The divergence of the vector potential: div(G) = 1.
        
        Since G(p) = p/3, we have:
        div(G) = ∂/∂x(x/3) + ∂/∂y(y/3) + ∂/∂z(z/3) = 1/3 + 1/3 + 1/3 = 1
        
        Args:
            points: (..., 3) tensor of 3D points
            feature_dim: Number of feature channels D
            
        Returns:
            divergence: (..., D) tensor of divergence values (all ones)
        """
        batch_shape = points.shape[:-1]
        
        # Return constant 1 for all points and all feature dimensions
        return torch.ones(*batch_shape, feature_dim, device=points.device, dtype=points.dtype)
    
    def compute_v_feat(self, points, feature_dim=1):
        """
        Compute the test feature: V_feat = -G⋅∇O.
        
        This is the key feature we're testing to see if an MLP can use it
        to reconstruct the scene.
        
        Args:
            points: (..., 3) tensor of 3D points
            feature_dim: Number of feature channels D
            
        Returns:
            v_feat: (..., D) tensor of test features
        """
        # Get analytical values
        g_values = self.analytical_g_query(points, feature_dim)  # (..., D, 3)
        grad_o = self.analytical_grad_o_query(points)  # (..., 3)
        
        # Expand grad_o to match feature dimensions
        grad_o_expanded = grad_o.unsqueeze(-2)  # (..., 1, 3)
        
        # Compute dot product: G⋅∇O for each feature channel
        dot_product = torch.sum(g_values * grad_o_expanded, dim=-1)  # (..., D)
        
        # Apply negative sign: V_feat = -G⋅∇O
        v_feat = -dot_product
        
        return v_feat
    
    def expected_v_feat_on_sphere(self, feature_dim=1):
        """
        Compute the expected value of V_feat on the sphere surface.
        
        On the sphere surface with radius r:
        - G(p) = p/3 
        - ∇O(p) = p/r (unit normal)
        - V_feat = -G⋅∇O = -(p/3)⋅(p/r) = -||p||²/(3r) = -r²/(3r) = -r/3
        
        Returns:
            expected_value: Scalar expected value on sphere surface
        """
        return -self.sphere_radius / 3.0
    
    def verify_divergence_condition(self, points, feature_dim=1, tolerance=1e-6):
        """
        Verify that div(G) = F everywhere.
        
        Args:
            points: (..., 3) tensor of test points
            feature_dim: Number of feature channels
            tolerance: Numerical tolerance for verification
            
        Returns:
            is_satisfied: Boolean indicating if condition is satisfied
            max_error: Maximum absolute error found
        """
        f_values = self.analytical_f_query(points)  # (...,)
        if feature_dim > 1:
            f_values = f_values.unsqueeze(-1).expand(*f_values.shape, feature_dim)  # (..., D)
        
        div_g_values = self.analytical_div_g_query(points, feature_dim)  # (..., D)
        
        # Compute absolute error
        error = torch.abs(div_g_values - f_values)
        max_error = torch.max(error).item()
        
        is_satisfied = max_error < tolerance
        
        return is_satisfied, max_error


class AnalyticalGridEncoder(nn.Module):
    """
    GridEncoder override that returns analytical values instead of learned features.
    
    This bypasses all hash table lookups and directly computes G(p) = p/3
    for any query point.
    """
    
    def __init__(self, oracles, num_levels=10, level_dim=4, **kwargs):
        """
        Initialize analytical grid encoder.
        
        Args:
            oracles: AnalyticalOracles instance
            num_levels: Number of resolution levels (for compatibility)
            level_dim: Feature dimension per level (for compatibility)
            **kwargs: Additional arguments (ignored, for compatibility)
        """
        super().__init__()
        self.oracles = oracles
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.output_dim = num_levels * level_dim * 3  # *3 for vector potential
        
        # Register dummy parameters to ensure model.parameters() works
        # These won't be used or trained
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        
        print(f"🔮 AnalyticalGridEncoder initialized:")
        print(f"   Num levels: {num_levels}")
        print(f"   Level dim: {level_dim}")
        print(f"   Output dim: {self.output_dim}")
    
    def forward(self, points, bound=1):
        """
        Query analytical vector potential G(p) = p/3.
        
        Args:
            points: (..., 3) tensor of query points in [-bound, bound]
            bound: Bounding box size (ignored, for compatibility)
            
        Returns:
            features: (..., num_levels * level_dim * 3) tensor of analytical features
        """
        batch_shape = points.shape[:-1]
        
        # Get analytical vector potential
        g_values = self.oracles.analytical_g_query(points, self.level_dim)  # (..., level_dim, 3)
        
        # Expand to multi-resolution (all levels identical per experiment design)
        g_multi_level = g_values.unsqueeze(-3).expand(*batch_shape, self.num_levels, self.level_dim, 3)
        
        # Flatten to expected output format
        features = g_multi_level.flatten(-3, -1)  # (..., num_levels * level_dim * 3)
        
        return features


class AnalyticalConfidenceField(nn.Module):
    """
    ConfidenceField override that returns analytical occupancy and gradients.
    
    This bypasses all grid interpolation and directly computes:
    - O(p) = 1 if abs(||p|| - r) <= epsilon else 0
    - ∇O(p) = normalize(p) if on sphere surface else (0,0,0)
    """
    
    def __init__(self, oracles, **kwargs):
        """
        Initialize analytical confidence field.
        
        Args:
            oracles: AnalyticalOracles instance
            **kwargs: Additional arguments (ignored, for compatibility)
        """
        super().__init__()
        self.oracles = oracles
        
        # Register dummy parameters for compatibility
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        
        print(f"🔮 AnalyticalConfidenceField initialized")
    
    def query(self, points):
        """
        Query analytical occupancy and gradients.
        
        Args:
            points: (N, 3) tensor of points in [-1, 1]
            
        Returns:
            occupancy: (N, 1) tensor of occupancy values
            gradients: (N, 3) tensor of gradient values
        """
        occupancy = self.oracles.analytical_o_query(points)  # (N, 1)
        gradients = self.oracles.analytical_grad_o_query(points)  # (N, 3)
        
        return occupancy, gradients
    
    def compute_gradient(self):
        """Dummy method for compatibility with existing code."""
        pass


def create_analytical_model_components(sphere_radius=1.0, sphere_center=None, 
                                     epsilon=0.05, num_levels=10, level_dim=4):
    """
    Create all analytical components for the controlled experiment.
    
    Args:
        sphere_radius: Radius of the hollow sphere
        sphere_center: Center of the sphere
        epsilon: Thickness of the sphere shell
        num_levels: Number of hash grid levels
        level_dim: Feature dimension per level
        
    Returns:
        oracles: AnalyticalOracles instance
        encoder: AnalyticalGridEncoder instance
        confidence_field: AnalyticalConfidenceField instance
    """
    # Create oracles
    oracles = AnalyticalOracles(
        sphere_radius=sphere_radius,
        sphere_center=sphere_center,
        epsilon=epsilon
    )
    
    # Create encoder
    encoder = AnalyticalGridEncoder(
        oracles=oracles,
        num_levels=num_levels,
        level_dim=level_dim
    )
    
    # Create confidence field
    confidence_field = AnalyticalConfidenceField(oracles=oracles)
    
    return oracles, encoder, confidence_field


def test_analytical_oracles():
    """Test the analytical oracle functions."""
    print("🧪 Testing Analytical Oracles...")
    
    # Create test points
    test_points = torch.tensor([
        [0.0, 0.0, 0.0],    # Center
        [1.0, 0.0, 0.0],    # On sphere surface
        [0.0, 1.0, 0.0],    # On sphere surface  
        [0.0, 0.0, 1.0],    # On sphere surface
        [2.0, 0.0, 0.0],    # Outside sphere
        [0.5, 0.0, 0.0],    # Inside sphere
    ], dtype=torch.float32)
    
    # Initialize oracles
    oracles = AnalyticalOracles(sphere_radius=1.0, epsilon=0.05)
    
    # Test F function
    f_values = oracles.analytical_f_query(test_points)
    print(f"F values: {f_values}")
    assert torch.allclose(f_values, torch.ones_like(f_values)), "F should be constant 1"
    
    # Test G function
    g_values = oracles.analytical_g_query(test_points, feature_dim=2)
    expected_g = test_points.unsqueeze(-2).expand(-1, 2, -1) / 3.0
    print(f"G values shape: {g_values.shape}")
    assert torch.allclose(g_values, expected_g), "G should equal p/3"
    
    # Test divergence condition
    is_satisfied, max_error = oracles.verify_divergence_condition(test_points, feature_dim=2)
    print(f"Divergence condition satisfied: {is_satisfied}, max error: {max_error}")
    assert is_satisfied, "div(G) should equal F"
    
    # Test V_feat computation
    v_feat = oracles.compute_v_feat(test_points, feature_dim=2)
    print(f"V_feat values: {v_feat}")
    
    # Test expected value on sphere
    expected_val = oracles.expected_v_feat_on_sphere()
    print(f"Expected V_feat on sphere: {expected_val}")
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_analytical_oracles() 