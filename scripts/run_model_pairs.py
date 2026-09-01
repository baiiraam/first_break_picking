#!/usr/bin/env python3
"""
Run model pairs across all datasets.
Trains models in pairs: [m1, m2] on dataset1, then dataset2, etc.
"""

import os
import sys
import subprocess
import click
from pathlib import Path
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logger

logger = setup_logger(task_name="model_pairs")


# ============================================================
# CONFIGURATION
# ============================================================

# Define model pairs (in order of increasing size)
MODEL_PAIRS = [
    ["pico", "nano"],           # Pair 1: ~2K + ~10K params
    ["tiny", "mpslight"],       # Pair 2: ~50K + ~1.7M params
    ["light", "mobile"],        # Pair 3: ~2.5M + ~3.5M params
    ["efficient", "unet"],      # Pair 4: ~5M + ~31M params
]

# All datasets
DATASETS = ["Brunswick", "Halfmile", "Lalor", "Sudbury"]

# Models to skip per dataset (if they don't fit)
SKIP_PER_DATASET = {
    "Lalor": ["unet"],  # UNet may not fit on Lalor (large dataset)
    # "Halfmile": [],  # All models fit
    # "Brunswick": [],
    # "Sudbury": [],
}


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_model_pairs(
    model_pairs: list,
    datasets: list,
    epochs: int = 2,
    device: str = "mps",
    verbose: bool = True,
    log_memory: bool = True,
    dry_run: bool = False,
):
    """
    Run model pairs across all datasets.
    
    Order: For each pair, train both models on dataset1, then dataset2, etc.
    """
    
    total_combinations = sum(
        len([m for m in pair if m not in SKIP_PER_DATASET.get(dataset, [])])
        for pair in model_pairs
        for dataset in datasets
    )
    
    logger.info("=" * 80)
    logger.info("🚀 MODEL PAIRS TRAINING")
    logger.info("=" * 80)
    logger.info(f"Model pairs: {len(model_pairs)}")
    logger.info(f"Datasets: {len(datasets)}")
    logger.info(f"Total combinations: {total_combinations}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)
    
    pair_count = 0
    combo_count = 0
    
    # ============================================================
    # OUTER LOOP: Model Pairs
    # ============================================================
    for pair_idx, model_pair in enumerate(model_pairs, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 PAIR {pair_idx}/{len(model_pairs)}: {model_pair}")
        logger.info(f"{'='*80}")
        
        # Filter models for this pair (skip those that don't fit)
        available_models = []
        for model in model_pair:
            # Check if model is skipped for any dataset
            model_skipped = False
            for dataset in datasets:
                if model in SKIP_PER_DATASET.get(dataset, []):
                    model_skipped = True
                    logger.info(f"⚠️  Model '{model}' skipped for {dataset} (won't fit)")
                    break
            if not model_skipped:
                available_models.append(model)
        
        if not available_models:
            logger.warning(f"⚠️  No models available for pair {pair_idx}")
            continue
        
        # ============================================================
        # INNER LOOP: Datasets
        # ============================================================
        for dataset_idx, dataset in enumerate(datasets, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📁 DATASET {dataset_idx}/{len(datasets)}: {dataset}")
            logger.info(f"{'='*60}")
            
            # Get models for this dataset (skip those that don't fit)
            dataset_models = [
                m for m in available_models 
                if m not in SKIP_PER_DATASET.get(dataset, [])
            ]
            
            if not dataset_models:
                logger.warning(f"⚠️  No models available for {dataset}")
                continue
            
            # Train each model in the pair on this dataset
            for model in dataset_models:
                combo_count += 1
                logger.info(f"\n  🔬 [{combo_count}/{total_combinations}] Model: {model}")
                
                # Build command
                cmd = [
                    "python3.12",
                    "scripts/train.py",
                    "--config", f"configs/{dataset.lower()}.yaml",
                    "--model", model,
                    "--epochs", str(epochs),
                    "--device", device,
                    "--loss", "combo",
                ]
                
                if verbose:
                    cmd.append("--verbose")
                if log_memory:
                    cmd.append("--log-memory")
                
                if dry_run:
                    logger.info(f"  🏃 DRY RUN: {' '.join(cmd)}")
                    continue
                
                # Run training
                logger.info(f"  🚀 Running: {' '.join(cmd)}")
                
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"  ✅ SUCCESS! {model} on {dataset}")
                    else:
                        logger.error(f"  ❌ FAILED! {model} on {dataset}")
                        if result.stderr:
                            logger.error(f"     Error: {result.stderr[:200]}")
                
                except Exception as e:
                    logger.error(f"  ❌ ERROR: {e}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    logger.info("\n" + "=" * 80)
    logger.info("📊 MODEL PAIRS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total combinations: {combo_count}")
    logger.info("=" * 80)


@click.command()
@click.option("--epochs", "-e", default=2, help="Number of epochs")
@click.option("--device", "-d", default="mps", help="Device to use")
@click.option("--dry-run", is_flag=True, help="Print commands without running")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--no-log-memory", is_flag=True, help="Disable memory logging")
def main(epochs, device, dry_run, verbose, no_log_memory):
    """Run model pairs across all datasets."""
    
    run_model_pairs(
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
