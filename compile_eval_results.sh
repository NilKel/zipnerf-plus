#!/bin/bash

# Script to compile and average evaluation results from log_eval.txt files

# Default configuration
EXP_BASE_DIR="exp"
MODE="triplane" # Default mode, can be changed via --mode

# Help function
show_help() {
    echo "📊 ZipNeRF Evaluation Result Compiler"
    echo "====================================="
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --mode MODE          Specify model mode: triplane or baseline (default: $MODE)"
    echo "  --help               Show this help"
    echo ""
    echo "Example:"
    echo "  $0 --mode triplane     # Compile results for triplane models"
    echo "  $0 --mode baseline     # Compile results for baseline models"
}

# Parse arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            if [[ "$MODE" != "triplane" && "$MODE" != "baseline" ]]; then
                echo "❌ Error: --mode must be 'triplane' or 'baseline', got: $2"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Initialize arrays for metrics
declare -a SCENES
declare -a PSNR_VALUES
declare -a SSIM_VALUES
declare -a LPIPS_VALUES
declare -a PSNR_CC_VALUES
declare -a SSIM_CC_VALUES
declare -a LPIPS_CC_VALUES

echo "Compiling evaluation results for '$MODE' models..."
echo "=================================================="

# Find all experiment directories for the specified mode
# The pattern for experiment directories is {scene}_{mode}_{steps}_{timestamp}
# We sort to ensure consistent output order
find "$EXP_BASE_DIR" -maxdepth 1 -type d -name "*_${MODE}_*_*_*" | sort | while read -r exp_path; do
    exp_name=$(basename "$exp_path")
    # Extract scene name by taking everything before the first underscore
    scene_name=$(echo "$exp_name" | cut -d'_' -f1) 

    log_file="$exp_path/log_eval.txt"

    if [ -f "$log_file" ]; then
        echo "Processing: $exp_name"

        # Extract the final metrics (non-color corrected)
        # Use sed to get the block starting from "metrics:" up to (but not including) "metrics_cc:"
        # Then grep for specific metrics within that block
        metrics_block=$(sed -n '/metrics:/,/metrics_cc:/p' "$log_file" | head -n -1)
        psnr_val=$(echo "$metrics_block" | grep "psnr:" | awk '{print $2}')
        ssim_val=$(echo "$metrics_block" | grep "ssim:" | awk '{print $2}')
        lpips_val=$(echo "$metrics_block" | grep "lpips:" | awk '{print $2}')

        # Extract the final color-corrected metrics
        # Use sed to get the block starting from "metrics_cc:" to the end of the file
        metrics_cc_block=$(sed -n '/metrics_cc:/,$p' "$log_file" | tail -n +1) # tail -n +1 to skip the "metrics_cc:" line itself
        psnr_cc_val=$(echo "$metrics_cc_block" | grep "psnr:" | awk '{print $2}')
        ssim_cc_val=$(echo "$metrics_cc_block" | grep "ssim:" | awk '{print $2}')
        lpips_cc_val=$(echo "$metrics_cc_block" | grep "lpips:" | awk '{print $2}')

        # Check if all values were successfully extracted
        if [ -n "$psnr_val" ] && [ -n "$ssim_val" ] && [ -n "$lpips_val" ] && \
           [ -n "$psnr_cc_val" ] && [ -n "$ssim_cc_val" ] && [ -n "$lpips_cc_val" ]; then
            
            SCENES+=("$scene_name")
            PSNR_VALUES+=("$psnr_val")
            SSIM_VALUES+=("$ssim_val")
            LPIPS_VALUES+=("$lpips_val")
            PSNR_CC_VALUES+=("$psnr_cc_val")
            SSIM_CC_VALUES+=("$ssim_cc_val")
            LPIPS_CC_VALUES+=("$lpips_cc_val")
            
            echo "  Extracted: PSNR=$psnr_val, SSIM=$ssim_val, LPIPS=$lpips_val (Non-CC)"
            echo "             PSNR_CC=$psnr_cc_val, SSIM_CC=$ssim_cc_val, LPIPS_CC=$lpips_cc_val (CC)"
        else
            echo "  Warning: Could not extract all required metrics from $log_file. Skipping."
        fi
    else
        echo "Skipping: $exp_name (log_eval.txt not found)"
    fi
done

echo ""
echo "Compilation complete. Calculating averages..."
echo "=================================================="

# Function to calculate average of an array of numbers
# Requires bash 4.3+ for nameref (local -n arr=$1)
calculate_average() {
    local -n arr=$1 
    local sum=0.0
    local count=0
    for val in "${arr[@]}"; do
        sum=$(echo "scale=10; $sum + $val" | bc -l) # Use higher scale for intermediate sums
        count=$((count + 1))
    done

    if [ "$count" -eq 0 ]; then
        echo "N/A"
    else
        echo "scale=4; $sum / $count" | bc -l # Final result formatted to 4 decimal places
    fi
}

# Check if any data was collected
if [ ${#SCENES[@]} -eq 0 ]; then
    echo "No evaluation results found for '$MODE' models matching the pattern."
    exit 0
fi

# Print individual results
echo "Individual Scene Results (Mode: $MODE)"
echo "-----------------------------------"
# Print header
printf "%-15s %-10s %-10s %-10s | %-12s %-12s %-12s\n" "Scene" "PSNR" "SSIM" "LPIPS" "PSNR_CC" "SSIM_CC" "LPIPS_CC"
printf "%-15s %-10s %-10s %-10s + %-12s %-12s %-12s\n" "---------------" "----------" "----------" "----------" "------------" "------------" "------------"

# Print data rows
for i in "${!SCENES[@]}"; do
    printf "%-15s %-10.4f %-10.4f %-10.4f | %-12.4f %-12.4f %-12.4f\n" \\
        "${SCENES[$i]}" \\
        "$(echo "${PSNR_VALUES[$i]}" | bc -l)" \\
        "$(echo "${SSIM_VALUES[$i]}" | bc -l)" \\
        "$(echo "${LPIPS_VALUES[$i]}" | bc -l)" \\
        "$(echo "${PSNR_CC_VALUES[$i]}" | bc -l)" \\
        "$(echo "${SSIM_CC_VALUES[$i]}" | bc -l)" \\
        "$(echo "${LPIPS_CC_VALUES[$i]}" | bc -l)"
done

echo ""
echo "Average Results Across ${#SCENES[@]} Scene(s) (Mode: $MODE)"
echo "--------------------------------------------------"

AVG_PSNR=$(calculate_average PSNR_VALUES)
AVG_SSIM=$(calculate_average SSIM_VALUES)
AVG_LPIPS=$(calculate_average LPIPS_VALUES)
AVG_PSNR_CC=$(calculate_average PSNR_CC_VALUES)
AVG_SSIM_CC=$(calculate_average SSIM_CC_VALUES)
AVG_LPIPS_CC=$(calculate_average LPIPS_CC_VALUES)

printf "%-15s %-10s %-10s %-10s | %-12s %-12s %-12s\n" "Average" "$AVG_PSNR" "$AVG_SSIM" "$AVG_LPIPS" "$AVG_PSNR_CC" "$AVG_SSIM_CC" "$AVG_LPIPS_CC"

echo ""
echo "Done." 