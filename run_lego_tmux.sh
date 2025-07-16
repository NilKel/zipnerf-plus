#!/bin/bash

# Launch ZipNeRF training in a tmux session
# This script creates a tmux session and runs the training with proper environment setup

SESSION_NAME="train_lego"
DATA_DIR="/home/nilkel/Projects/data/nerf_synthetic/lego"
EXP_NAME="lego_triplane_25k"
WANDB_PROJECT="my-blender-experiments"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create new tmux session and run training
tmux new-session -d -s "$SESSION_NAME" -c "/home/nilkel/Projects/zipnerf-pytorch" bash -c "
    echo '🔧 Activating conda environment...';
    source ~/miniconda3/etc/profile.d/conda.sh;
    conda activate zipnerf2;
    echo 'Environment activated: '\$(conda info --show-active-env);
    echo 'Python path: '\$(which python);
    echo 'PyTorch CUDA: '\$(python -c 'import torch; print(torch.cuda.is_available())');
    
    echo '🚀 Starting ZipNeRF training for lego scene...';
    echo '📁 Data: $DATA_DIR';
    echo '🏷️  Experiment: $EXP_NAME';
    echo '📊 Wandb project: $WANDB_PROJECT';
    echo '';
    
    set -e;  # Exit on any error
    trap 'echo \"❌ Training failed at step: \$LINENO\"' ERR;
    
    accelerate launch train.py \\
        --gin_configs=configs/blender.gin \\
        --gin_bindings=\"Config.data_dir = '$DATA_DIR'\" \\
        --gin_bindings=\"Config.exp_name = '$EXP_NAME'\" \\
        --gin_bindings=\"Config.wandb_project = '$WANDB_PROJECT'\" \\
        --gin_bindings=\"Config.max_steps = 25000\" \\
        --gin_bindings=\"Config.batch_size = 65536\" \\
        --gin_bindings=\"Config.factor = 4\" \\
        --gin_bindings=\"Config.use_wandb = True\" \\
        --gin_bindings=\"Config.use_triplane = True\" 2>&1 | tee training_output.log;
    
    echo '✅ Training completed! Press any key to exit...';
    read -n 1;
"

echo "🎬 Training session '$SESSION_NAME' started!"
echo "📺 To attach to the session: tmux attach-session -t $SESSION_NAME"
echo "🔍 To check running sessions: tmux list-sessions"
echo "⏹️  To kill the session: tmux kill-session -t $SESSION_NAME" 