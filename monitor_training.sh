#!/bin/bash

# Training Monitor Script
# Helps monitor active tmux training sessions and their logs

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 ZipNeRF Training Monitor${NC}"
echo "========================="

# Check for active tmux sessions
echo -e "\n${YELLOW}📺 Active tmux sessions:${NC}"
if tmux list-sessions 2>/dev/null; then
    echo ""
    echo -e "${BLUE}To attach to a session: tmux attach-session -t <session_name>${NC}"
else
    echo "No tmux sessions running"
fi

# Check for recent training logs
echo -e "\n${YELLOW}📄 Recent training logs:${NC}"
if [ -d "tmux_logs" ]; then
    echo "Latest 5 log files:"
    ls -lt tmux_logs/*.log 2>/dev/null | head -5 | while read line; do
        echo "  $line"
    done
    
    # Check for exit logs (indicates problems)
    echo -e "\n${YELLOW}⚠️  Exit logs (check for issues):${NC}"
    if ls tmux_logs/*_exit.log 2>/dev/null; then
        echo -e "${RED}Found exit logs - check these for issues:${NC}"
        ls -lt tmux_logs/*_exit.log | while read line; do
            echo "  $line"
        done
    else
        echo -e "${GREEN}No exit logs found (good!)${NC}"
    fi
else
    echo "No tmux_logs directory found"
fi

# Check GPU usage
echo -e "\n${YELLOW}🖥️  GPU Status:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r gpu name mem_used mem_total util; do
        echo "  GPU $gpu ($name): ${mem_used}MB/${mem_total}MB (${util}% util)"
    done
else
    echo "nvidia-smi not available"
fi

# Check for training processes
echo -e "\n${YELLOW}🔄 Training processes:${NC}"
if pgrep -f "train.py" > /dev/null; then
    echo -e "${GREEN}Found active training processes:${NC}"
    ps aux | grep -E "train\.py|accelerate.*train" | grep -v grep | while read line; do
        echo "  $line"
    done
else
    echo "No training processes found"
fi

echo -e "\n${BLUE}💡 Useful commands:${NC}"
echo "  tail -f tmux_logs/<latest>.log     # Follow latest log"
echo "  tmux attach-session -t <name>      # Attach to session"
echo "  tmux list-sessions                 # List all sessions"
echo "  watch nvidia-smi                   # Monitor GPU continuously"
echo "  ./train_pt_tmux.sh <scene> <comment>  # Start new training" 