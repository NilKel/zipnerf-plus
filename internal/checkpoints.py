import os
import shutil

import accelerate
import torch
import glob


def restore_checkpoint(
        checkpoint_dir,
        accelerator: accelerate.Accelerator,
        logger=None
):
    dirs = glob.glob(os.path.join(checkpoint_dir, "*"))
    dirs.sort()
    path = dirs[-1] if len(dirs) > 0 else None
    if path is None:
        if logger is not None:
            logger.info("Checkpoint does not exist. Starting a new training run.")
        init_step = 0
    else:
        if logger is not None:
            logger.info(f"Resuming from checkpoint {path}")
        accelerator.load_state(path)
        init_step = int(os.path.basename(path))
    return init_step


def save_checkpoint(save_dir,
                    accelerator: accelerate.Accelerator,
                    step=0,
                    total_limit=3):
    # Save the new checkpoint first
    new_checkpoint_path = os.path.join(save_dir, f"{step:06d}")
    accelerator.save_state(new_checkpoint_path)
    
    # Then clean up old checkpoints, ensuring we don't delete the one we just saved
    if total_limit > 0:
        folders = glob.glob(os.path.join(save_dir, "*"))
        folders.sort()
        # Only delete if we have more than the limit
        if len(folders) > total_limit:
            for folder in folders[: len(folders) - total_limit]:
                # Double-check we're not deleting the checkpoint we just saved
                if folder != new_checkpoint_path:
                    shutil.rmtree(folder)
