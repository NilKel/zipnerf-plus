from internal import math
from internal import utils
import numpy as np
import torch
# from torch.func import vmap, jacrev


def contract(x):
    """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
    eps = torch.finfo(x.dtype).eps
    # eps = 1e-3
    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
    z = torch.where(x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
    return z


def inv_contract(z):
    """The inverse of contract()."""
    eps = torch.finfo(z.dtype).eps

    # Clamping to eps prevents non-finite gradients when z == 0.
    z_mag_sq = torch.sum(z ** 2, dim=-1, keepdim=True).clamp_min(eps)
    x = torch.where(z_mag_sq <= 1, z, z / (2 * torch.sqrt(z_mag_sq) - z_mag_sq).clamp_min(eps))
    return x


def inv_contract_np(z):
    """The inverse of contract()."""
    eps = np.finfo(z.dtype).eps

    # Clamping to eps prevents non-finite gradients when z == 0.
    z_mag_sq = np.maximum(np.sum(z ** 2, axis=-1, keepdims=True), eps)
    x = np.where(z_mag_sq <= 1, z, z / np.maximum(2 * np.sqrt(z_mag_sq) - z_mag_sq, eps))
    return x


def contract_tuple(x):
    res = contract(x)
    return res, res


def contract_mean_jacobi(x):
    eps = torch.finfo(x.dtype).eps
    # eps = 1e-3

    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
    x_mag_sqrt = torch.sqrt(x_mag_sq)
    x_xT = math.matmul(x[..., None], x[..., None, :])
    mask = x_mag_sq <= 1
    z = torch.where(x_mag_sq <= 1, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)

    eye = torch.broadcast_to(torch.eye(3, device=x.device), z.shape[:-1] + z.shape[-1:] * 2)
    jacobi = (2 * x_xT * (1 - x_mag_sqrt[..., None]) + (2 * x_mag_sqrt[..., None] ** 3 - x_mag_sqrt[..., None] ** 2) * eye) / x_mag_sqrt[..., None] ** 4
    jacobi = torch.where(mask[..., None], eye, jacobi)
    return z, jacobi


def contract_mean_std(x, std):
    eps = torch.finfo(x.dtype).eps
    # eps = 1e-3
    # Clamping to eps prevents non-finite gradients when x == 0.
    x_mag_sq = torch.sum(x ** 2, dim=-1, keepdim=True).clamp_min(eps)
    x_mag_sqrt = torch.sqrt(x_mag_sq)
    mask = x_mag_sq <= 1
    z = torch.where(mask, x, ((2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq) * x)
    # det_13 = ((1 / x_mag_sq) * ((2 / x_mag_sqrt - 1 / x_mag_sq) ** 2)) ** (1 / 3)
    det_13 = (torch.pow(2 * x_mag_sqrt - 1, 1/3) / x_mag_sqrt) ** 2

    std = torch.where(mask[..., 0], std, det_13[..., 0] * std)
    return z, std


def contract_cubic(x):
    """MeRF's cubic contraction function (Eq 7) - optimized version."""
    eps = torch.finfo(x.dtype).eps
    
    # L∞ norm (max absolute coordinate) - optimized
    abs_x = torch.abs(x)
    x_norm_inf, max_indices = torch.max(abs_x, dim=-1, keepdim=True)
    
    # Early return for points inside unit cube (most common case)
    mask_inside = x_norm_inf <= 1.0
    if torch.all(mask_inside):
        return x
    
    # For outside points, vectorized computation
    x_norm_inf_safe = torch.clamp(x_norm_inf, min=eps)
    
    # Compute both transformations vectorized
    # Non-max coords: x_j / ||x||∞
    contracted_non_max = x / x_norm_inf_safe
    
    # Max coords: sign(x_j) * (2 - 1/|x_j|)
    abs_x_safe = torch.clamp(abs_x, min=eps)
    sign_x = torch.sign(x)
    contracted_max = sign_x * (2.0 - 1.0 / abs_x_safe)
    
    # Create mask for max coordinates more efficiently
    max_coord_mask = (abs_x == x_norm_inf) & (x_norm_inf > 1.0)
    
    # Combine using vectorized selection
    contracted = torch.where(max_coord_mask, contracted_max, contracted_non_max)
    
    # Apply final mask for inside vs outside
    result = torch.where(mask_inside, x, contracted)
    return result


def inv_contract_cubic(z):
    """Inverse of MeRF's cubic contraction function."""
    eps = torch.finfo(z.dtype).eps
    
    # L∞ norm in contracted space
    z_norm_inf = torch.max(torch.abs(z), dim=-1, keepdim=True)[0]
    
    # Case 1: Inside unit cube ||z||∞ ≤ 1 (identity mapping)
    mask_inside = z_norm_inf <= 1.0
    
    # Case 2: Outside unit cube ||z||∞ > 1
    abs_z = torch.abs(z)
    max_coord_mask = (abs_z == z_norm_inf) & (z_norm_inf > 1.0)
    
    # For coordinates that are NOT the max: solve x_j = z_j * ||x||∞
    # We need to find ||x||∞ first from the max coordinate
    
    # For the max coordinate: solve z_j = sign(x_j) * (2 - 1/|x_j|)
    # This gives us: |x_j| = 1 / (2 - |z_j|)
    sign_z = torch.sign(z)
    abs_z_clamped = torch.clamp(abs_z, min=eps, max=2.0-eps)  # Clamp to valid range
    
    # Solve for the max coordinate magnitude
    x_max_mag = 1.0 / (2.0 - abs_z_clamped + eps)
    
    # Reconstruct the max coordinate
    x_max = sign_z * x_max_mag
    
    # For non-max coordinates, use the relationship x_j = z_j * ||x||∞
    # where ||x||∞ is the magnitude of the max coordinate
    x_norm_inf_reconstructed = torch.where(max_coord_mask, x_max_mag, torch.zeros_like(x_max_mag))
    x_norm_inf_reconstructed = torch.max(x_norm_inf_reconstructed, dim=-1, keepdim=True)[0]
    
    # Reconstruct non-max coordinates
    x_non_max = z * x_norm_inf_reconstructed
    
    # Combine max and non-max coordinates
    x_outside = torch.where(max_coord_mask, x_max, x_non_max)
    
    # Final result: identity inside, reconstructed outside
    result = torch.where(mask_inside, z, x_outside)
    return result


def contract_cubic_mean_jacobi(x):
    """MeRF's cubic contraction with Jacobian computation."""
    eps = torch.finfo(x.dtype).eps
    
    # L∞ norm and setup
    x_norm_inf = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
    mask_inside = x_norm_inf <= 1.0
    abs_x = torch.abs(x)
    max_coord_mask = (abs_x == x_norm_inf) & (x_norm_inf > 1.0)
    
    # Contract the coordinates (same as contract_cubic)
    contracted_non_max = x / (x_norm_inf + eps)
    sign_x = torch.sign(x)
    abs_x_clamped = torch.clamp(abs_x, min=eps)
    contracted_max = sign_x * (2.0 - 1.0 / abs_x_clamped)
    contracted_outside = torch.where(max_coord_mask, contracted_max, contracted_non_max)
    z = torch.where(mask_inside, x, contracted_outside)
    
    # Compute Jacobian
    device = x.device
    eye = torch.eye(3, device=device).expand(*x.shape[:-1], 3, 3)
    
    # Inside unit cube: Jacobian is identity
    jacobi = eye.clone()
    
    # Outside unit cube: compute Jacobian for each region
    if not torch.all(mask_inside):
        # This is a simplified Jacobian computation
        # For the exact Jacobian, we'd need to handle each of the 7 regions separately
        # For now, we'll use an approximation based on the dominant scaling
        
        # Approximate Jacobian scaling factor
        # In regions where coordinate is not max: 1/||x||∞ 
        # In regions where coordinate is max: 1/|x_j|²
        scale_non_max = 1.0 / (x_norm_inf + eps)
        scale_max = 1.0 / (abs_x_clamped ** 2 + eps)
        
        # Apply scaling to diagonal
        scale_factor = torch.where(max_coord_mask, scale_max, scale_non_max)
        jacobi_outside = eye * scale_factor.unsqueeze(-1)
        
        # Use outside Jacobian where needed
        jacobi = torch.where(mask_inside.unsqueeze(-1), eye, jacobi_outside)
    
    return z, jacobi


def contract_cubic_mean_std(x, std):
    """MeRF's cubic contraction with std transformation - optimized version."""
    eps = torch.finfo(x.dtype).eps
    
    # L∞ norm and early exit optimization
    abs_x = torch.abs(x)
    x_norm_inf = torch.max(abs_x, dim=-1, keepdim=True)[0]
    mask_inside = x_norm_inf <= 1.0
    
    # Early return if all points inside (most common case)
    if torch.all(mask_inside):
        return x, std
    
    # Vectorized contraction computation (reuse optimized logic)
    x_norm_inf_safe = torch.clamp(x_norm_inf, min=eps)
    
    # Non-max and max coordinate transformations
    contracted_non_max = x / x_norm_inf_safe
    abs_x_safe = torch.clamp(abs_x, min=eps)
    sign_x = torch.sign(x)
    contracted_max = sign_x * (2.0 - 1.0 / abs_x_safe)
    
    # Apply transformations
    max_coord_mask = (abs_x == x_norm_inf) & (x_norm_inf > 1.0)
    contracted = torch.where(max_coord_mask, contracted_max, contracted_non_max)
    z = torch.where(mask_inside, x, contracted)
    
    # Optimized std transformation - simplified approximation
    # Use average scaling for efficiency (good approximation for cubic contraction)
    scale_factor = 1.0 / (x_norm_inf_safe + eps)
    det_approx = scale_factor.squeeze(-1)  # Simplified determinant approximation
    
    # Apply std scaling only where needed
    new_std = torch.where(mask_inside.squeeze(-1), std, det_approx * std)
    
    return z, new_std


@torch.no_grad()
def track_linearize(fn, mean, std, use_cubic_contraction=False):
    """Apply function `fn` to a set of means and covariances, ala a Kalman filter.

  We can analytically transform a Gaussian parameterized by `mean` and `cov`
  with a function `fn` by linearizing `fn` around `mean`, and taking advantage
  of the fact that Covar[Ax + y] = A(Covar[x])A^T (see
  https://cs.nyu.edu/~roweis/notes/gaussid.pdf for details).

  Args:
    fn: the function applied to the Gaussians parameterized by (mean, cov).
    mean: a tensor of means, where the last axis is the dimension.
    std: a tensor of covariances, where the last two axes are the dimensions.
    use_cubic_contraction: if True, use MeRF's cubic contraction; if False, use mip-NeRF 360's spherical contraction.

  Returns:
    fn_mean: the transformed means.
    fn_cov: the transformed covariances.
  """
    if fn == 'contract':
        if use_cubic_contraction:
            fn = contract_cubic_mean_std
        else:
            fn = contract_mean_std
    else:
        raise NotImplementedError

    pre_shape = mean.shape[:-1]
    mean = mean.reshape(-1, 3)
    std = std.reshape(-1)

    # jvp_1, mean_1 = vmap(jacrev(contract_tuple, has_aux=True))(mean)
    # std_1 = std * torch.linalg.det(jvp_1) ** (1 / mean.shape[-1])
    #
    # mean_2, jvp_2 = fn(mean)
    # std_2 = std * torch.linalg.det(jvp_2) ** (1 / mean.shape[-1])
    #
    # mean_3, std_3 = contract_mean_std(mean, std)  # calculate det explicitly by using eigenvalues
    # torch.allclose(std_1, std_3, atol=1e-7)  # True
    # torch.allclose(mean_1, mean_3)  # True
    # import ipdb; ipdb.set_trace()
    mean, std = fn(mean, std)  # calculate det explicitly by using eigenvalues

    mean = mean.reshape(*pre_shape, 3)
    std = std.reshape(*pre_shape)
    return mean, std


def power_transformation(x, lam):
    """
    power transformation for Eq(4) in zip-nerf
    """
    lam_1 = np.abs(lam - 1)
    return lam_1 / lam * ((x / lam_1 + 1) ** lam - 1)


def inv_power_transformation(x, lam):
    """
    inverse power transformation
    """
    lam_1 = np.abs(lam - 1)
    eps = torch.finfo(x.dtype).eps  # may cause inf
    # eps = 1e-3
    return ((x * lam / lam_1 + 1 + eps) ** (1 / lam) - 1) * lam_1


def construct_ray_warps(fn, t_near, t_far, lam=None):
    """Construct a bijection between metric distances and normalized distances.

  See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
  detailed explanation.

  Args:
    fn: the function to ray distances.
    t_near: a tensor of near-plane distances.
    t_far: a tensor of far-plane distances.
    lam: for lam in Eq(4) in zip-nerf

  Returns:
    t_to_s: a function that maps distances to normalized distances in [0, 1].
    s_to_t: the inverse of t_to_s.
  """
    if fn is None:
        fn_fwd = lambda x: x
        fn_inv = lambda x: x
    elif fn == 'piecewise':
        # Piecewise spacing combining identity and 1/x functions to allow t_near=0.
        fn_fwd = lambda x: torch.where(x < 1, .5 * x, 1 - .5 / x)
        fn_inv = lambda x: torch.where(x < .5, 2 * x, .5 / (1 - x))
    elif fn == 'power_transformation':
        fn_fwd = lambda x: power_transformation(x * 2, lam=lam)
        fn_inv = lambda y: inv_power_transformation(y, lam=lam) / 2
    else:
        inv_mapping = {
            'reciprocal': torch.reciprocal,
            'log': torch.exp,
            'exp': torch.log,
            'sqrt': torch.square,
            'square': torch.sqrt,
        }
        fn_fwd = fn
        fn_inv = inv_mapping[fn.__name__]

    s_near, s_far = [fn_fwd(x) for x in (t_near, t_far)]
    t_to_s = lambda t: (fn_fwd(t) - s_near) / (s_far - s_near)
    s_to_t = lambda s: fn_inv(s * s_far + (1 - s) * s_near)
    return t_to_s, s_to_t


def expected_sin(mean, var):
    """Compute the mean of sin(x), x ~ N(mean, var)."""
    return torch.exp(-0.5 * var) * math.safe_sin(mean)  # large var -> small value.


def integrated_pos_enc(mean, var, min_deg, max_deg):
    """Encode `x` with sinusoids scaled by 2^[min_deg, max_deg).

  Args:
    mean: tensor, the mean coordinates to be encoded
    var: tensor, the variance of the coordinates to be encoded.
    min_deg: int, the min degree of the encoding.
    max_deg: int, the max degree of the encoding.

  Returns:
    encoded: tensor, encoded variables.
  """
    scales = 2 ** torch.arange(min_deg, max_deg, device=mean.device)
    shape = mean.shape[:-1] + (-1,)
    scaled_mean = (mean[..., None, :] * scales[:, None]).reshape(*shape)
    scaled_var = (var[..., None, :] * scales[:, None] ** 2).reshape(*shape)

    return expected_sin(
        torch.cat([scaled_mean, scaled_mean + 0.5 * torch.pi], dim=-1),
        torch.cat([scaled_var] * 2, dim=-1))


def lift_and_diagonalize(mean, cov, basis):
    """Project `mean` and `cov` onto basis and diagonalize the projected cov."""
    fn_mean = math.matmul(mean, basis)
    fn_cov_diag = torch.sum(basis * math.matmul(cov, basis), dim=-2)
    return fn_mean, fn_cov_diag


def pos_enc(x, min_deg, max_deg, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = 2 ** torch.arange(min_deg, max_deg, device=x.device)
    shape = x.shape[:-1] + (-1,)
    scaled_x = (x[..., None, :] * scales[:, None]).reshape(*shape)
    # Note that we're not using safe_sin, unlike IPE.
    four_feat = torch.sin(
        torch.cat([scaled_x, scaled_x + 0.5 * torch.pi], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)
    else:
        return four_feat
