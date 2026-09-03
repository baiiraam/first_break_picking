#!/bin/bash
# scripts/cleanup_old_checkpoints.sh

cd models/registry

echo "🧹 Cleaning up old checkpoints..."

# Keep only the latest best model (without timestamp)
# Delete all timestamped best models
echo "  Deleting timestamped best models..."
rm -f *_best_*.pt

# For each model-dataset combination, keep only the latest epoch
echo "  Keeping only latest epoch checkpoints..."

# Pattern: {Model}_{Dataset}_epoch_{N}_{timestamp}.pt
# Keep the one with the highest N

# PicoUNet Halfmile
ls -t PicoUNet_Halfmile_epoch_*.pt 2>/dev/null | tail -n +2 | xargs rm -f 2>/dev/null

# PicoUNet Sudbury
ls -t PicoUNet_Sudbury_epoch_*.pt 2>/dev/null | tail -n +2 | xargs rm -f 2>/dev/null

# MPSLightUNet Halfmile
ls -t MPSLightUNet_Halfmile_epoch_*.pt 2>/dev/null | tail -n +2 | xargs rm -f 2>/dev/null

# NanoUNet Brunswick
ls -t NanoUNet_Brunswick_epoch_*.pt 2>/dev/null | tail -n +2 | xargs rm -f 2>/dev/null

# TinyUNet Brunswick
ls -t TinyUNet_Brunswick_epoch_*.pt 2>/dev/null | tail -n +2 | xargs rm -f 2>/dev/null

echo "✅ Cleanup complete!"
echo ""
echo "📁 Files remaining:"
ls -la *.pt