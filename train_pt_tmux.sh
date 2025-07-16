#!/bin/bash

# Robust tmux wrapper for train_pt.sh
# Usage: ./train_pt_tmux.sh <scene> [comment] [binary_occupancy]
# Example: ./train_pt_tmux.sh lego binocc_unfixedoptconf_nogate_noreg_5xconf

set -e  # Exit on error

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <scene> [comment] [binary_occupancy]"
    echo "Example: $0 lego my_experiment"
    echo "Example: $0 lego binary_test True"
    echo ""
    echo "This creates a tmux session that runs train_pt.sh with robust logging and monitoring."
    exit 1
fi

SCENE="$1"
COMMENT="${2:-}"
BINARY_OCC="${3:-False}"

# Generate unique session and experiment names
TIMESTAMP=$(date +%m%d_%H%M%S)
SESSION_NAME="train_pt_${SCENE}_${TIMESTAMP}"
if [ -n "$COMMENT" ]; then
    LOG_NAME="training_pt_${SCENE}_${COMMENT}_${TIMESTAMP}"
else
    LOG_NAME="training_pt_${SCENE}_${TIMESTAMP}"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting ZipNeRF Potential+Triplane training in tmux${NC}"
echo -e "${BLUE}🎬 Scene: $SCENE${NC}"
if [ -n "$COMMENT" ]; then
    echo -e "${BLUE}💬 Comment: $COMMENT${NC}"
fi
echo -e "${BLUE}🔲 Binary Occupancy: $BINARY_OCC${NC}"
echo -e "${BLUE}📺 Session: $SESSION_NAME${NC}"

# Verify prerequisites
echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

# Check if conda environment exists
if ! conda info --envs | grep -q "zipnerf2"; then
    echo -e "${RED}❌ zipnerf2 environment not found${NC}"
    echo "Available environments:"
    conda info --envs
    exit 1
fi

# Check if data directory exists
DATA_DIR="/home/nilkel/Projects/data/nerf_synthetic/$SCENE"
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Data directory not found: $DATA_DIR${NC}"
    exit 1
fi

# Check if config file exists
CONFIG_FILE="configs/potential_triplane.gin"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Check if learned confidence grid exists
DEBUG_GRID="debug_grids/debug_confidence_grid_256.pt"
if [ ! -f "$DEBUG_GRID" ]; then
    echo -e "${RED}❌ Learned confidence grid not found: $DEBUG_GRID${NC}"
    echo -e "${RED}   Please run training without loading a grid first to generate it${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"
echo -e "${BLUE}🔧 Will load existing confidence grid: $DEBUG_GRID${NC}"

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create logging directory
LOG_DIR="./tmux_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${LOG_NAME}.log"
EXIT_LOG="$LOG_DIR/${LOG_NAME}_exit.log"

echo -e "${GREEN}📄 Logs will be saved to: $LOG_FILE${NC}"

# Create the tmux session
tmux new-session -d -s "$SESSION_NAME" -c "/home/nilkel/Projects/zipnerf-pytorch"

# Create a comprehensive training script within tmux
tmux send-keys -t "$SESSION_NAME" "
# Function to log tmux exit
log_exit() {
    echo \"⚠️  TMUX SESSION EXITED UNEXPECTEDLY AT \$(date)\" >> $EXIT_LOG
    echo \"Exit code: \$1\" >> $EXIT_LOG
    echo \"Last 50 lines of training log:\" >> $EXIT_LOG
    echo \"=====================================\" >> $EXIT_LOG
    tail -50 $LOG_FILE >> $EXIT_LOG 2>/dev/null || echo \"No training log found\" >> $EXIT_LOG
    echo \"=====================================\" >> $EXIT_LOG
    echo \"Process list at exit:\" >> $EXIT_LOG
    ps aux | grep -E \"python|accelerate|train\" | grep -v grep >> $EXIT_LOG || echo \"No training processes found\" >> $EXIT_LOG
}

# Set up exit trap
trap 'log_exit \$?' EXIT

echo '🔧 Setting up environment...' | tee -a $LOG_FILE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zipnerf2

echo '📋 Environment verification:' | tee -a $LOG_FILE
echo \"Active environment: \$(conda info --show-active-env)\" | tee -a $LOG_FILE
echo \"Python path: \$(which python)\" | tee -a $LOG_FILE
echo \"PyTorch version: \$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'FAILED')\" | tee -a $LOG_FILE
echo \"CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'FAILED')\" | tee -a $LOG_FILE
echo \"GPU name: \$(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'FAILED')\" | tee -a $LOG_FILE

# Set error handling
set -e
trap 'echo \"❌ Training failed at line \$LINENO. Check logs: $LOG_FILE\" | tee -a $LOG_FILE; log_exit \$?' ERR

# Set environment variables for stability
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

echo '' | tee -a $LOG_FILE
echo '🚀 Starting ZipNeRF Potential+Triplane training...' | tee -a $LOG_FILE
echo \"🎬 Scene: $SCENE\" | tee -a $LOG_FILE
echo \"💬 Comment: $COMMENT\" | tee -a $LOG_FILE
echo \"🔲 Binary Occupancy: $BINARY_OCC\" | tee -a $LOG_FILE
echo \"⏰ Started at: \$(date)\" | tee -a $LOG_FILE
echo \"📁 Data directory: $DATA_DIR\" | tee -a $LOG_FILE
echo \"📋 Config: $CONFIG_FILE\" | tee -a $LOG_FILE
echo '' | tee -a $LOG_FILE

# Build and execute the training command
echo '💻 Training command:' | tee -a $LOG_FILE
CMD=\"accelerate launch train.py --gin_configs=$CONFIG_FILE\"
CMD=\"\$CMD --gin_bindings='Config.data_dir = \\\"$DATA_DIR\\\"'\"
CMD=\"\$CMD --gin_bindings='Config.debug_confidence_grid_path = \\\"debug_grids/debug_confidence_grid_256.pt\\\"'\"
CMD=\"\$CMD --gin_bindings='Config.freeze_debug_confidence = False'\"
CMD=\"\$CMD --gin_bindings='Config.binary_occupancy = $BINARY_OCC'\"
CMD=\"\$CMD --gin_bindings='Config.confidence_grid_resolution = (256, 256, 256)'\"
" C-m

# Add comment binding if provided
if [ -n "$COMMENT" ]; then
    tmux send-keys -t "$SESSION_NAME" "CMD=\"\$CMD --gin_bindings='Config.comment = \\\"$COMMENT\\\"'\"" C-m
fi

tmux send-keys -t "$SESSION_NAME" "
echo \"\$CMD\" | tee -a $LOG_FILE
echo '' | tee -a $LOG_FILE

# Execute training with full logging
eval \$CMD 2>&1 | tee -a $LOG_FILE

# Check if training completed successfully
if [ \$? -eq 0 ]; then
    echo '' | tee -a $LOG_FILE
    echo '✅ Training completed successfully!' | tee -a $LOG_FILE
    echo \"⏰ Finished at: \$(date)\" | tee -a $LOG_FILE
    echo \"📄 Full log: $LOG_FILE\" | tee -a $LOG_FILE
    echo \"📁 Check results in: exp/\" | tee -a $LOG_FILE
    
    # Save confidence grid from final model
    echo '' | tee -a $LOG_FILE
    echo '💾 Saving learned confidence grid...' | tee -a $LOG_FILE
    
    # Create debug_grids directory
    mkdir -p debug_grids
    
    # Find the most recent experiment directory
    EXP_DIR=\$(find exp -name \"*$COMMENT*\" -type d | sort -V | tail -1)
    if [ -z \"\$EXP_DIR\" ]; then
        # Fallback to most recent directory
        EXP_DIR=\$(find exp -name \"lego_*\" -type d | sort -V | tail -1)
    fi
    
    echo \"🔍 Looking for experiment in: \$EXP_DIR\" | tee -a $LOG_FILE
    
    if [ -d \"\$EXP_DIR\" ]; then
        # Find the final checkpoint
        CHECKPOINT_DIR=\$(find \"\$EXP_DIR\" -name \"checkpoints\" -type d | head -1)
        if [ -d \"\$CHECKPOINT_DIR\" ]; then
            FINAL_CHECKPOINT=\$(find \"\$CHECKPOINT_DIR\" -name \"*\" -type d | sort -V | tail -1)
            if [ -d \"\$FINAL_CHECKPOINT\" ]; then
                echo \"📂 Found final checkpoint: \$FINAL_CHECKPOINT\" | tee -a $LOG_FILE
                
                # Extract confidence grid using Python script
                python3 -c \"
import torch
import accelerate
import gin
from pathlib import Path
from internal import configs, models, checkpoints

try:
    # Load experiment config
    exp_path = Path('\$EXP_DIR')
    config_gin_path = exp_path / 'config.gin'
    
    if config_gin_path.exists():
        gin.clear_config()
        gin.parse_config_file(str(config_gin_path))
        config = configs.Config()
        
        # Setup model
        accelerator = accelerate.Accelerator()
        model = models.Model(config=config)
        model = accelerator.prepare(model)
        
        # Load checkpoint
        checkpoint_dir = Path('\$FINAL_CHECKPOINT').parent
        step = checkpoints.restore_checkpoint(checkpoint_dir, accelerator)
        
        # Extract confidence field
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, 'confidence_field'):
            confidence_field = unwrapped_model.confidence_field
            learned_logits = confidence_field.c_grid.data.clone().cpu()
            
            # Save the confidence grid
            output_path = 'debug_grids/confidence_grid_256_learned.pt'
            torch.save(learned_logits, output_path)
            
            print(f'✅ Saved confidence grid to: {output_path}')
            print(f'   Shape: {learned_logits.shape}')
            print(f'   Range: [{learned_logits.min():.3f}, {learned_logits.max():.3f}]')
            
            # Also save as default debug grid
            debug_path = 'debug_grids/debug_confidence_grid_256.pt'
            torch.save(learned_logits, debug_path)
            print(f'✅ Also saved as: {debug_path}')
        else:
            print('❌ No confidence field found in model')
    else:
        print('❌ Config file not found')
        
except Exception as e:
    print(f'❌ Error saving confidence grid: {e}')
    import traceback
    traceback.print_exc()
\" 2>&1 | tee -a $LOG_FILE
            else
                echo \"❌ No final checkpoint found in: \$CHECKPOINT_DIR\" | tee -a $LOG_FILE
            fi
        else
            echo \"❌ No checkpoints directory found in: \$EXP_DIR\" | tee -a $LOG_FILE
        fi
    else
        echo \"❌ Experiment directory not found: \$EXP_DIR\" | tee -a $LOG_FILE
    fi
    
    echo '' | tee -a $LOG_FILE
else
    echo '' | tee -a $LOG_FILE
    echo '❌ Training failed!' | tee -a $LOG_FILE
fi

# Remove exit trap since we completed successfully
trap - EXIT

echo 'Training finished. Press any key to exit tmux session...'
read -n 1
" C-m

echo ""
echo -e "${GREEN}🎬 Training session '$SESSION_NAME' started!${NC}"
echo ""
echo -e "${BLUE}📺 To attach to session:${NC}"
echo "   tmux attach-session -t $SESSION_NAME"
echo ""
echo -e "${BLUE}🔍 To monitor training:${NC}"
echo "   tail -f $LOG_FILE"
echo ""
echo -e "${BLUE}⚠️  To check for unexpected exits:${NC}"
echo "   cat $EXIT_LOG"
echo ""
echo -e "${BLUE}📊 Other useful commands:${NC}"
echo "   tmux list-sessions                    # List all sessions"
echo "   tmux kill-session -t $SESSION_NAME   # Kill this session"
echo "   nvidia-smi                           # Check GPU usage"
echo ""
echo -e "${YELLOW}💡 Tip: You can safely detach with Ctrl+B, then D${NC}"
echo -e "${YELLOW}    The training will continue and logs will capture any issues${NC}"

# Monitor for session exit in background
(
    while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
        sleep 10
    done
    if [ ! -s "$EXIT_LOG" ]; then
        echo "✅ Session '$SESSION_NAME' ended normally at $(date)" >> "$EXIT_LOG"
    fi
) & 