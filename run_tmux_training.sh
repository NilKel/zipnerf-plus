#!/bin/bash

# Robust ZipNeRF Training Script for tmux
# This script ensures proper environment setup and error handling

set -e  # Exit on error

# Configuration
SESSION_NAME="zipnerf_training"
DATA_DIR="/home/nilkel/Projects/data/nerf_synthetic/lego"
EXP_NAME="lego_triplane_robust_$(date +%m%d_%H%M)"
WANDB_PROJECT="my-blender-experiments"
MAX_STEPS=25000
BATCH_SIZE=65536
FACTOR=4

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting robust ZipNeRF training in tmux${NC}"
echo -e "${BLUE}📁 Data: $DATA_DIR${NC}"
echo -e "${BLUE}🏷️  Experiment: $EXP_NAME${NC}"
echo -e "${BLUE}📊 Wandb: $WANDB_PROJECT${NC}"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Function to check if conda environment exists
check_conda_env() {
    if conda info --envs | grep -q "zipnerf2"; then
        echo -e "${GREEN}✅ Found zipnerf2 environment${NC}"
        return 0
    else
        echo -e "${RED}❌ zipnerf2 environment not found${NC}"
        echo "Available environments:"
        conda info --envs
        return 1
    fi
}

# Check prerequisites
echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"
check_conda_env || exit 1

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Data directory not found: $DATA_DIR${NC}"
    exit 1
fi

# Create logging directory
LOG_DIR="./tmux_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_${EXP_NAME}.log"

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Create the tmux session with comprehensive setup
tmux new-session -d -s "$SESSION_NAME" -c "/home/nilkel/Projects/zipnerf-pytorch"

# Send commands to the tmux session
tmux send-keys -t "$SESSION_NAME" "
echo '🔧 Setting up environment...'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zipnerf2

# Verify environment
echo '📋 Environment verification:'
echo 'Active environment:' \$(conda info --show-active-env)
echo 'Python path:' \$(which python)
echo 'PyTorch version:' \$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'FAILED')
echo 'CUDA available:' \$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'FAILED')
echo 'GPU name:' \$(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'FAILED')

# Set error handling
set -e
trap 'echo \"❌ Training failed at line \$LINENO. Check logs for details.\"' ERR

# Set environment variables for stability
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1  # Better error reporting

echo ''
echo '🚀 Starting ZipNeRF training...'
echo '📁 Data: $DATA_DIR'
echo '🏷️  Experiment: $EXP_NAME'
echo '📊 Wandb: $WANDB_PROJECT'
echo '⏰ Started at:' \$(date)
echo ''

# Run training with full logging
accelerate launch train.py \\
    --gin_configs=configs/blender.gin \\
    --gin_bindings=\"Config.data_dir = '$DATA_DIR'\" \\
    --gin_bindings=\"Config.exp_name = '$EXP_NAME'\" \\
    --gin_bindings=\"Config.wandb_project = '$WANDB_PROJECT'\" \\
    --gin_bindings=\"Config.max_steps = $MAX_STEPS\" \\
    --gin_bindings=\"Config.batch_size = $BATCH_SIZE\" \\
    --gin_bindings=\"Config.factor = $FACTOR\" \\
    --gin_bindings=\"Config.use_wandb = True\" \\
    --gin_bindings=\"Config.use_triplane = True\" \\
    2>&1 | tee $LOG_FILE

echo ''
echo '✅ Training completed successfully!'
echo '⏰ Finished at:' \$(date)
echo '📄 Full log saved to: $LOG_FILE'
echo '📁 Experiment directory: exp/$EXP_NAME'
echo ''
echo 'Press any key to exit tmux session...'
read -n 1
" C-m

echo ""
echo -e "${GREEN}🎬 Training session '$SESSION_NAME' started!${NC}"
echo -e "${BLUE}📺 To attach: tmux attach-session -t $SESSION_NAME${NC}"
echo -e "${BLUE}🔍 To check sessions: tmux list-sessions${NC}"
echo -e "${BLUE}⏹️  To kill session: tmux kill-session -t $SESSION_NAME${NC}"
echo -e "${BLUE}📄 Log file: $LOG_FILE${NC}"
echo ""
echo -e "${YELLOW}💡 Tip: You can safely detach (Ctrl+B, then D) and the training will continue${NC}" 