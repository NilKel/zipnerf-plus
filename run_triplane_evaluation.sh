#!/bin/bash
# ZipNeRF Triplane Models Evaluation Script
# Usage: ./run_triplane_evaluation.sh [dry_run|organize_only|test_only]

set -e  # Exit on error

echo "🚀 ZipNeRF Triplane Models Evaluation"
echo "====================================="

# Make the script executable
chmod +x organize_and_test_triplane_models.py

# Determine mode
MODE=""
if [ "$1" = "dry_run" ]; then
    MODE="--dry_run"
    echo "🔍 Running in DRY RUN mode"
elif [ "$1" = "organize_only" ]; then
    MODE="--organize_only"
    echo "🗂️  Only organizing models"
elif [ "$1" = "test_only" ]; then
    MODE="--test_only"
    echo "🧪 Only running evaluation"
else
    echo "🔄 Running full organize + evaluation"
fi

echo ""

# Run the Python script
python organize_and_test_triplane_models.py $MODE

echo ""
echo "🎉 Evaluation completed!"
echo "📊 Check the 'triplane_results' directory for reports" 