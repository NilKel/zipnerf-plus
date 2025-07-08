#!/bin/bash

# Quick test script for all triplane models
# Usage: ./quick_test_all.sh [dry_run]

echo "🚀 ZipNeRF Triplane Models - Quick Test Script"
echo "=============================================="

if [ "$1" = "dry_run" ]; then
    echo "🔍 Running in DRY RUN mode..."
    python test_all_triplane_models.py --dry_run
else
    echo "🧪 Running full evaluation..."
    echo "🏃 Starting evaluation automatically (no confirmations)..."
    python test_all_triplane_models.py
fi 