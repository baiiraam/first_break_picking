#!/usr/bin/env python3
"""
Run model pairs across all datasets WITH graceful fallback.
Uses batch_train.py internally for memory recovery.
"""

import os
import sys
import subprocess
import click
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logger

logger = setup_logger(task_name="model_pairs_fallback")


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PAIRS = [
    ["pico", "nano"],           # Pair 1: ~2K + ~10K params
    ["tiny", "mpslight"],       # Pair 2: ~50K + ~1.7M params
    ["light", "mobile"],        # Pair 3: ~2.5M + ~3.5M params
    ["efficient", "unet"],      # Pair 4: ~5M + ~31M params
]

DATASETS = ["Brunswick", "Halfmile", "Lalor", "Sudbury"]

# Models to skip per dataset (if they don't fit)
SKIP_PER_DATASET = {
    "Lalor": ["unet", "efficient"],
}


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_model_pairs_with_fallback(
    model_pairs: list,
    datasets: list,
    epochs: int = 2,
    device: str = "mps",
    verbose: bool = True,
    log_memory: bool = True,
    dry_run: bool = False,
):
    """
    Run model pairs using batch_train.py for graceful fallback.
    
    For each pair, runs batch_train.py with specific models filtered.
    """
    
    total_pairs = len(model_pairs)
    pair_count = 0
    
    logger.info("=" * 80)
    logger.info("🚀 MODEL PAIRS WITH GRACEFUL FALLBACK")
    logger.info("=" * 80)
    logger.info(f"Model pairs: {total_pairs}")
    logger.info(f"Datasets: {len(datasets)}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)
    
    for pair_idx, model_pair in enumerate(model_pairs, 1):
        pair_count += 1
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 PAIR {pair_idx}/{total_pairs}: {model_pair}")
        logger.info(f"{'='*80}")
        
        # Filter models for this pair
        available_models = []
        for model in model_pair:
            # Check if model is skipped for ANY dataset
            is_skipped = False
            for dataset in datasets:
                if model in SKIP_PER_DATASET.get(dataset, []):
                    is_skipped = True
                    logger.info(f"⚠️  Model '{model}' will be skipped for {dataset}")
                    break
            if not is_skipped:
                available_models.append(model)
        
        if not available_models:
            logger.warning(f"⚠️  No models available for pair {pair_idx}")
            continue
        
        # ============================================================
        # KEY: Use batch_train.py with --models filter
        # ============================================================
        
        # Build command for batch_train.py
        cmd = [
            "python3.12",
            "scripts/batch_train.py",
            "--auto-config",           # ← Enables smart config + fallback
            "--epochs", str(epochs),
            "--device", device,
        ]
        
        # Add models (one per --models flag)
        for model in available_models:
            cmd.extend(["--models", model])
        
        # Add datasets (one per --datasets flag)
        for dataset in datasets:
            cmd.extend(["--datasets", dataset])
        
        # Add logging flags
        if verbose:
            cmd.append("--verbose")
        if log_memory:
            cmd.append("--log-memory")
        
        if dry_run:
            logger.info(f"\n🏃 DRY RUN: {' '.join(cmd)}")
            continue
        
        # Run batch_train.py for this pair
        logger.info(f"\n🚀 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                logger.info(f"✅ SUCCESS! Pair {pair_idx} completed")
            else:
                logger.error(f"❌ FAILED! Pair {pair_idx} had errors")
                if result.stderr:
                    logger.error(f"   Error: {result.stderr[:500]}")
                    
        except Exception as e:
            logger.error(f"❌ ERROR running pair: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 MODEL PAIRS WITH FALLBACK - COMPLETE")
    logger.info("=" * 80)


@click.command()
@click.option("--epochs", "-e", default=2, help="Number of epochs")
@click.option("--device", "-d", default="mps", help="Device to use")
@click.option("--dry-run", is_flag=True, help="Print commands without running")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--no-log-memory", is_flag=True, help="Disable memory logging")
def main(epochs, device, dry_run, verbose, no_log_memory):
    """Run model pairs with graceful fallback."""
    
    run_model_pairs_with_fallback(
        model_pairs=MODEL_PAIRS,
        datasets=DATASETS,
        epochs=epochs,
        device=device,
        verbose=verbose,
        log_memory=not no_log_memory,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
