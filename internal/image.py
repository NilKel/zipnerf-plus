import torch
import numpy as np
from internal import math
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import cv2

# Import LPIPS for perceptual loss evaluation
try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    LPIPS_AVAILABLE = True
except ImportError:
    try:
        import lpips
        LPIPS_AVAILABLE = True
    except ImportError:
        LPIPS_AVAILABLE = False


def mse_to_psnr(mse):
    """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
    return -10. / np.log(10.) * np.log(mse)


def psnr_to_mse(psnr):
    """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
    return np.exp(-0.1 * np.log(10.) * psnr)


def ssim_to_dssim(ssim):
    """Compute DSSIM given an SSIM."""
    return (1 - ssim) / 2


def dssim_to_ssim(dssim):
    """Compute DSSIM given an SSIM."""
    return 1 - 2 * dssim


def linear_to_srgb(linear, eps=None):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = torch.finfo(linear.dtype).eps
        # eps = 1e-3

    srgb0 = 323 / 25 * linear
    srgb1 = (211 * linear.clamp_min(eps) ** (5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def linear_to_srgb_np(linear, eps=None):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = np.finfo(linear.dtype).eps
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
    return np.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb, eps=None):
    """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = np.finfo(srgb.dtype).eps
    linear0 = 25 / 323 * srgb
    linear1 = np.maximum(eps, ((200 * srgb + 11) / (211))) ** (12 / 5)
    return np.where(srgb <= 0.04045, linear0, linear1)


def downsample(img, factor):
    """Area downsample img (factor must evenly divide img height and width)."""
    sh = img.shape
    if not (sh[0] % factor == 0 and sh[1] % factor == 0):
        raise ValueError(f'Downsampling factor {factor} does not '
                         f'evenly divide image shape {sh[:2]}')
    img = img.reshape((sh[0] // factor, factor, sh[1] // factor, factor) + sh[2:])
    img = img.mean((1, 3))
    return img


def color_correct(img, ref, num_iters=5, eps=0.5 / 255):
    """Warp `img` to match the colors in `ref_img`."""
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f'img\'s {img.shape[-1]} and ref\'s {ref.shape[-1]} channels must match'
        )
    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])
    ref_mat = ref.reshape([-1, num_channels])
    is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))  # z \in [eps, 1-eps].
    mask0 = is_unclipped(img_mat)
    # Because the set of saturated pixels may change after solving for a
    # transformation, we repeatedly solve a system `num_iters` times and update
    # our estimate of which pixels are saturated.
    for _ in range(num_iters):
        # Construct the left hand side of a linear system that contains a quadratic
        # expansion of each pixel of `img`.
        a_mat = []
        for c in range(num_channels):
            a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])  # Quadratic term.
        a_mat.append(img_mat)  # Linear term.
        a_mat.append(torch.ones_like(img_mat[:, :1]))  # Bias term.
        a_mat = torch.cat(a_mat, dim=-1)
        warp = []
        for c in range(num_channels):
            # Construct the right hand side of a linear system containing each color
            # of `ref`.
            b = ref_mat[:, c]
            # Ignore rows of the linear system that were saturated in the input or are
            # saturated in the current corrected color estimate.
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = torch.where(mask[:, None], a_mat, torch.zeros_like(a_mat))
            mb = torch.where(mask, b, torch.zeros_like(b))
            w = torch.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
            assert torch.all(torch.isfinite(w))
            warp.append(w)
        warp = torch.stack(warp, dim=-1)
        # Apply the warp to update img_mat.
        img_mat = torch.clip(math.matmul(a_mat, warp), 0, 1)
    corrected_img = torch.reshape(img_mat, img.shape)
    return corrected_img


class MetricHarness:
    """A helper class for evaluating several error metrics."""

    def __init__(self):
        """Initialize LPIPS model if available."""
        if LPIPS_AVAILABLE:
            try:
                # Try torchmetrics version first
                self.lpips_model = LearnedPerceptualImagePatchSimilarity(normalize=True)
                self.lpips_model.eval()
                self._lpips_type = 'torchmetrics'
            except:
                try:
                    # Fallback to lpips package
                    self.lpips_model = lpips.LPIPS(net='alex')
                    self.lpips_model.eval()
                    self._lpips_type = 'lpips'
                except:
                    self.lpips_model = None
                    self._lpips_type = None
        else:
            self.lpips_model = None
            self._lpips_type = None

    def __call__(self, rgb_pred, rgb_gt, name_fn=lambda s: s):
        """Evaluate the error between a predicted rgb image and the true image."""
        rgb_pred = (rgb_pred * 255).astype(np.uint8)
        rgb_gt = (rgb_gt * 255).astype(np.uint8)
        rgb_pred_gray = cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2GRAY)
        rgb_gt_gray = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)
        psnr = float(peak_signal_noise_ratio(rgb_pred, rgb_gt, data_range=255))
        ssim = float(structural_similarity(rgb_pred_gray, rgb_gt_gray, data_range=255))

        metrics = {
            name_fn('psnr'): psnr,
            name_fn('ssim'): ssim,
        }
        
        # Add LPIPS if available
        if self.lpips_model is not None:
            try:
                # Convert back to float [0, 1] for LPIPS
                rgb_pred_float = rgb_pred.astype(np.float32) / 255.0
                rgb_gt_float = rgb_gt.astype(np.float32) / 255.0
                
                # Convert to torch tensors and add batch dimension
                rgb_pred_tensor = torch.from_numpy(rgb_pred_float).permute(2, 0, 1).unsqueeze(0)
                rgb_gt_tensor = torch.from_numpy(rgb_gt_float).permute(2, 0, 1).unsqueeze(0)
                
                with torch.no_grad():
                    if self._lpips_type == 'torchmetrics':
                        lpips_score = float(self.lpips_model(rgb_pred_tensor, rgb_gt_tensor).item())
                    elif self._lpips_type == 'lpips':
                        lpips_score = float(self.lpips_model(rgb_pred_tensor, rgb_gt_tensor).item())
                    else:
                        lpips_score = 0.0
                
                metrics[name_fn('lpips')] = lpips_score
            except Exception as e:
                print(f"Warning: LPIPS computation failed: {e}")
                metrics[name_fn('lpips')] = 0.0
        
        return metrics
