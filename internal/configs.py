import dataclasses
import os
from typing import Any, Callable, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from absl import flags
import gin
from internal import utils
from datetime import datetime

gin.add_config_file_search_path('configs/')

configurables = {
    'torch': [torch.reciprocal, torch.log, torch.log1p, torch.exp, torch.sqrt, torch.square],
}

for module, configurables in configurables.items():
    for configurable in configurables:
        gin.config.external_configurable(configurable, module=module)


@gin.configurable()
@dataclasses.dataclass
class Config:
    """Configuration flags for everything."""
    use_triplane: bool = False
    use_potential: bool = False
    confidence_grid_resolution: Tuple[int, int, int] = (128, 128, 128)
    confidence_reg_mult: float = 0.0  # Multiplier for confidence field regularization loss
    binary_occupancy: bool = False  # If True, use binary occupancy with STE instead of smooth sigmoid
    
    # Debug/sanity check options for confidence field
    debug_confidence_grid_path: Optional[str] = None  # Path to pretrained confidence grid for debugging
    freeze_debug_confidence: bool = True  # If True, freeze the debug confidence grid (no gradients)
    seed = 0
    dataset_loader: str = 'llff'  # The type of dataset loader to use.
    batching: str = 'all_images'  # Batch composition, [single_image, all_images].
    batch_size: int = 2 ** 16  # The number of rays/pixels in each batch.
    patch_size: int = 1  # Resolution of patches sampled for training batches.
    factor: int = 4  # The downsample factor of images, 0 for no downsampling.
    multiscale: bool = False  # use multiscale data for training.
    multiscale_levels: int = 4  # number of multiscale levels.
    # ordering (affects heldout test set).
    forward_facing: bool = False  # Set to True for forward-facing LLFF captures.
    render_path: bool = False  # If True, render a path. Used only by LLFF.
    llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
    # If true, use all input images for training.
    llff_use_all_images_for_training: bool = False
    llff_use_all_images_for_testing: bool = False
    use_tiffs: bool = False  # If True, use 32-bit TIFFs. Used only by Blender.
    compute_disp_metrics: bool = False  # If True, load and compute disparity MSE.
    compute_normal_metrics: bool = False  # If True, load and compute normal MAE.
    disable_multiscale_loss: bool = False  # If True, disable multiscale loss.
    randomized: bool = True  # Use randomized stratified sampling.
    near: float = 2.  # Near plane distance.
    far: float = 6.  # Far plane distance.
    exp_name: str = "test"  # experiment name
    comment: str = ""  # Optional comment to append to experiment name
    data_dir: Optional[str] = "/SSD_DISK/datasets/360_v2/bicycle"  # Input data directory.
    vocab_tree_path: Optional[str] = None  # Path to vocab tree for COLMAP.
    render_chunk_size: int = 65536  # Chunk size for whole-image renderings.
    num_showcase_images: int = 5  # The number of test-set images to showcase.
    deterministic_showcase: bool = True  # If True, showcase the same images.
    vis_num_rays: int = 16  # The number of rays to visualize.
    # Decimate images for tensorboard (ie, x[::d, ::d]) to conserve memory usage.
    vis_decimate: int = 0

    # Weights & Biases logging configuration
    use_wandb: bool = False  # If True, use Weights & Biases for logging
    wandb_project: str = "zipnerf"  # WandB project name
    wandb_entity: Optional[str] = None  # WandB entity (username/team)
    wandb_name: Optional[str] = None  # WandB run name (defaults to exp_name if None)
    wandb_tags: List[str] = None  # WandB run tags
    wandb_notes: Optional[str] = None  # WandB run notes
    disable_tensorboard: bool = False  # If True, disable tensorboard logging

    dpcpp_backend: bool = False  # If True, use DPC++ for the backend.
    importance_sampling: bool = False # If True, use CUDA or DPC++ implementation
    # of importance ray sampling from nerfacc

    # Only used by train.py:
    max_steps: int = 25000  # The number of optimization steps.
    early_exit_steps: Optional[int] = None  # Early stopping, for debugging.
    checkpoint_every: int = 5000  # The number of steps to save a checkpoint.
    resume_from_checkpoint: bool = True  # whether to resume from checkpoint.
    no_wandb: bool = False
    checkpoints_total_limit: int = 1
    gradient_scaling: bool = False  # If True, scale gradients as in https://gradient-scaling.github.io/.
    print_every: int = 100  # The number of steps between reports to tensorboard.
    train_render_every: int = 500  # Steps between test set renders when training
    data_loss_type: str = 'charb'  # What kind of loss to use ('mse' or 'charb').
    charb_padding: float = 0.001  # The padding used for Charbonnier loss.
    data_loss_mult: float = 1.0  # Mult for the finest data term in the loss.
    data_coarse_loss_mult: float = 0.  # Multiplier for the coarser data terms.
    interlevel_loss_mult: float = 0.0  # Mult. for the loss on the proposal MLP.
    anti_interlevel_loss_mult: float = 0.01  # Mult. for the loss on the proposal MLP.
    pulse_width = [0.03, 0.003]  # Mult. for the loss on the proposal MLP.
    orientation_loss_mult: float = 0.0  # Multiplier on the orientation loss.
    orientation_coarse_loss_mult: float = 0.0  # Coarser orientation loss weights.
    # What that loss is imposed on, options are 'normals' or 'normals_pred'.
    orientation_loss_target: str = 'normals_pred'
    predicted_normal_loss_mult: float = 0.0  # Mult. on the predicted normal loss.
    # Mult. on the coarser predicted normal loss.
    predicted_normal_coarse_loss_mult: float = 0.0
    hash_decay_mults: float = 0.1

    lr_init: float = 0.01  # The initial learning rate.
    lr_final: float = 0.001  # The final learning rate.
    lr_delay_steps: int = 5000  # The number of "warmup" learning steps.
    lr_delay_mult: float = 1e-8
    
    # Confidence field specific learning rates
    confidence_lr_init: Optional[float] = None  # Initial LR for confidence field (None = use lr_init)
    confidence_lr_final: Optional[float] = None  # Final LR for confidence field (None = use lr_final)
    confidence_lr_delay_steps: Optional[int] = None  # Warmup steps for confidence field (None = use lr_delay_steps)
    confidence_lr_delay_mult: Optional[float] = None  # Warmup mult for confidence field (None = use lr_delay_mult)
    confidence_lr_multiplier: float = 1.0  # Multiplier for confidence field LR (applied to main LR if specific rates not set)
    adam_beta1: float = 0.9  # Adam's beta2 hyperparameter.
    adam_beta2: float = 0.99  # Adam's beta2 hyperparameter.
    adam_eps: float = 1e-15  # Adam's epsilon hyperparameter.
    grad_max_norm: float = 0.  # Gradient clipping magnitude, disabled if == 0.
    grad_max_val: float = 0.  # Gradient clipping value, disabled if == 0.
    distortion_loss_mult: float = 0.005  # Multiplier on the distortion loss.
    opacity_loss_mult: float = 0.  # Multiplier on the distortion loss.

    # Only used by eval.py:
    eval_only_once: bool = True  # If True evaluate the model only once, ow loop.
    eval_save_output: bool = True  # If True save predicted images to disk.
    eval_save_ray_data: bool = False  # If True save individual ray traces.
    eval_render_interval: int = 1  # The interval between images saved to disk.
    eval_dataset_limit: int = np.iinfo(np.int32).max  # Num test images to eval.
    eval_quantize_metrics: bool = True  # If True, run metrics on 8-bit images.
    eval_crop_borders: int = 0  # Ignore c border pixels in eval (x[c:-c, c:-c]).

    # Only used by render.py
    render_video_fps: int = 60  # Framerate in frames-per-second.
    render_video_crf: int = 18  # Constant rate factor for ffmpeg video quality.
    render_path_frames: int = 120  # Number of frames in render path.
    z_variation: float = 0.  # How much height variation in render path.
    z_phase: float = 0.  # Phase offset for height variation in render path.
    render_dist_percentile: float = 0.5  # How much to trim from near/far planes.
    render_dist_curve_fn: Callable[..., Any] = np.log  # How depth is curved.
    render_path_file: Optional[str] = None  # Numpy render pose file to load.
    render_resolution: Optional[Tuple[int, int]] = None  # Render resolution, as
    # (width, height).
    render_focal: Optional[float] = None  # Render focal length.
    render_camtype: Optional[str] = None  # 'perspective', 'fisheye', or 'pano'.
    render_spherical: bool = False  # Render spherical 360 panoramas.
    render_save_async: bool = True  # Save to CNS using a separate thread.

    render_spline_keyframes: Optional[str] = None  # Text file containing names of
    # images to be used as spline
    # keyframes, OR directory
    # containing those images.
    render_spline_n_interp: int = 30  # Num. frames to interpolate per keyframe.
    render_spline_degree: int = 5  # Polynomial degree of B-spline interpolation.
    render_spline_smoothness: float = .03  # B-spline smoothing factor, 0 for
    # exact interpolation of keyframes.
    # Interpolate per-frame exposure value from spline keyframes.
    render_spline_interpolate_exposure: bool = False

    # Flags for raw datasets.
    rawnerf_mode: bool = False  # Load raw images and train in raw color space.
    exposure_percentile: float = 97.  # Image percentile to expose as white.
    num_border_pixels_to_mask: int = 0  # During training, discard N-pixel border
    # around each input image.
    apply_bayer_mask: bool = False  # During training, apply Bayer mosaic mask.
    autoexpose_renders: bool = False  # During rendering, autoexpose each image.
    # For raw test scenes, use affine raw-space color correction.
    eval_raw_affine_cc: bool = False

    zero_glo: bool = False

    # marching cubes
    valid_weight_thresh: float = 0.05
    isosurface_threshold: float = 20
    mesh_voxels: int = 512 ** 3
    visibility_resolution: int = 512
    mesh_radius: float = 1.0  # mesh radius * 2 = in contract space
    mesh_max_radius: float = 10.0  # in world space
    std_value: float = 0.0  # std of the sampled points
    compute_visibility: bool = False
    extract_visibility: bool = True
    decimate_target: int = -1
    vertex_color: bool = True
    vertex_projection: bool = True

    # tsdf
    tsdf_radius: float = 2.0
    tsdf_resolution: int = 512
    truncation_margin: float = 5.0
    tsdf_max_radius: float = 10.0  # in world space
    
    def __post_init__(self):
        """Auto-generate experiment names and adjust settings based on configuration."""
        # Extract scene name from data_dir
        scene_name = os.path.basename(self.data_dir.rstrip('/'))
        
        # Determine model type suffix
        if self.use_potential and self.use_triplane:
            model_suffix = "potential_triplane"
        elif self.use_triplane:
            model_suffix = "triplane"
        elif self.use_potential:
            model_suffix = "potential"
        else:
            model_suffix = "baseline"
            
        # Generate timestamp
        timestamp = datetime.now().strftime("%m%d_%H%M")
        
        # Set wandb project for synthetic datasets
        if 'nerf_synthetic' in self.data_dir:
            self.wandb_project = "my-nerf-experiments"
            
        # Auto-generate experiment name if not set or is default
        if self.exp_name == "test":
            self.exp_name = f"{scene_name}_{model_suffix}_{self.max_steps}_{timestamp}"
            
        # Append comment if provided
        if self.comment:
            self.exp_name = f"{self.exp_name}_{self.comment}"
            
        # Set wandb_name if not explicitly set
        if self.wandb_name is None:
            self.wandb_name = self.exp_name
            
        # Adjust batch size for potential encoder if memory constrained
        if self.use_potential and self.use_triplane and self.batch_size > 4096:
            print(f"Warning: Large batch size ({self.batch_size}) with potential+triplane may cause OOM. Consider reducing.")


def define_common_flags():
    # Define the flags used by both train.py and eval.py
    flags.DEFINE_string('mode', None, 'Required by GINXM, not used.')
    flags.DEFINE_string('base_folder', None, 'Required by GINXM, not used.')
    flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
    flags.DEFINE_multi_string('gin_configs', None, 'Gin config files.')


def load_config():
    """Load the config, and optionally checkpoint it."""
    gin.parse_config_files_and_bindings(
        flags.FLAGS.gin_configs, flags.FLAGS.gin_bindings, skip_unknown=True)
    config = Config()
    return config
