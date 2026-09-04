#!/usr/bin/env python3
"""
Run model pairs across all datasets WITH graceful fallback.
Uses batch_train.py internally for memory recovery.
"""

import os
import subprocess
import sys

import click

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.logger import setup_logger

logger = setup_logger(task_name="model_pairs_fallback")


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PAIRS = [
    ["pico", "nano"],  # Pair 1: ~2K + ~10K params
    ["tiny", "mpslight"],  # Pair 2: ~50K + ~1.7M params
    ["light", "mobile"],  # Pair 3: ~2.5M + ~3.5M params
    ["efficient", "unet"],  # Pair 4: ~5M + ~31M params
]

DATASETS = ["Halfmile", "Sudbury", "Brunswick", "Lalor"]

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
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📊 PAIR {pair_idx}/{total_pairs}: {model_pair}")
        logger.info(f"{'=' * 80}")

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
            "--auto-config",  # ← Enables smart config + fallback
            "--epochs",
            str(epochs),
            "--device",
            device,
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
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                logger.info(f"✅ SUCCESS! Pair {pair_idx} completed")
            else:
                logger.error(f"❌ FAILED! Pair {pair_idx} had errors")
                if result.stderr:
                    logger.error(f"   Error: {result.stderr[:500]}")

        except Exception as e:  # noqa BLE001
            logger.error(f"❌ ERROR running pair: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("📊 MODEL PAIRS WITH FALLBACK - COMPLETE")
    logger.info("=" * 80)


def run_dry_run(
    model_pairs: list,
    datasets: list,
    epochs: int,
    device: str,
    show_mlflow: bool = False,
    show_functions: bool = False,
    show_output: bool = False,
    show_metrics: bool = False,
    verbose: bool = False,
):
    """Run an enhanced dry run with detailed information."""

    logger.info("=" * 80)
    logger.info("🏃 ENHANCED DRY RUN")
    logger.info("=" * 80)
    logger.info(f"Model pairs: {len(model_pairs)}")
    logger.info(f"Datasets: {len(datasets)}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)

    # Generate all combinations
    all_combinations = []
    for model_pair in model_pairs:
        for dataset in datasets:
            for model in model_pair:
                # Check if model is skipped
                if model in SKIP_PER_DATASET.get(dataset, []):
                    continue
                all_combinations.append((dataset, model, model_pair))

    # ============================================================
    # 1. SHOW FUNCTION CALL STACK
    # ============================================================
    if show_functions:
        logger.info("\n" + "=" * 80)
        logger.info("📚 FUNCTION CALL STACK")
        logger.info("=" * 80)

        call_stack = [
            ("1. run_model_pairs_with_fallback()", "Entry point"),
            ("   └── 2. build_command()", "Builds the command string"),
            ("       └── 3. subprocess.run()", "Runs the command"),
            ("           └── 4. scripts/batch_train.py", "Main orchestrator"),
            ("               └── 5. run_auto_batch_training()", "Auto-config training"),
            ("                   └── 6. train_dataset()", "Trains a single model"),
            ("                       └── 7. subprocess.run()", "Runs train.py"),
            (
                "                           └── 8. scripts/train.py",
                "Single model training",
            ),
            (
                "                               └── 9. SeismicTrainer.fit()",
                "Main training loop",
            ),
            (
                "                                   └── 10. train_epoch()",
                "Runs one epoch",
            ),
            (
                "                                       └── 11. model.forward()",
                "Forward pass",
            ),
        ]

        for call in call_stack:
            logger.info(f"  {call[0]}  # {call[1]}")

        logger.info("\n  📌 Total combinations: {len(all_combinations)}")
        logger.info("  📌 Each combination follows this stack")

    # ============================================================
    # 2. SHOW COMMANDS
    # ============================================================
    logger.info("\n" + "=" * 80)
    logger.info("📝 COMMANDS THAT WILL RUN")
    logger.info("=" * 80)

    for idx, (dataset, model, model_pair) in enumerate(all_combinations, 1):
        pair_name = f"{model_pair[0]}+{model_pair[1]}"
        cmd = (
            f"python3.12 scripts/batch_train.py "
            f"--auto-config --epochs {epochs} --device {device} "
            f"--models {model} --datasets {dataset} "
            f"--verbose --log-memory"
        )
        logger.info(
            f"\n  [{idx}/{len(all_combinations)}] {dataset} | {model} (Pair: {pair_name})"
        )
        logger.info(f"    🏃 {cmd}")

    # ============================================================
    # 3. SHOW MLFLOW RUN DETAILS
    # ============================================================
    if show_mlflow:
        logger.info("\n" + "=" * 80)
        logger.info("📊 MLFLOW RUN PREVIEW (First 3 combinations)")
        logger.info("=" * 80)

        for idx, (dataset, model, model_pair) in enumerate(all_combinations[:3], 1):
            logger.info(f"\n  Run {idx}: {dataset}_{model}_epoch_{epochs}")
            logger.info("  ├── Parameters:")
            logger.info(f'  │   ├── dataset: "{dataset}"')
            logger.info(f'  │   ├── model: "{model}"')
            logger.info('  │   ├── loss: "combo"')
            logger.info(f"  │   ├── epochs: {epochs}")
            logger.info("  │   └── batch_size: auto-detected")
            logger.info("  ├── Tags:")
            logger.info(f'  │   ├── dataset: "{dataset}"')
            logger.info(f'  │   └── model_type: "{model}"')
            logger.info("  └── Artifacts:")
            logger.info("      ├── model/ (saved model)")
            logger.info("      ├── checkpoints/ (checkpoints)")
            logger.info("      └── config.yaml")

        if len(all_combinations) > 3:
            logger.info(f"\n  ... and {len(all_combinations) - 3} more runs")

    # ============================================================
    # 4. SHOW EXPECTED METRICS
    # ============================================================
    if show_metrics:
        logger.info("\n" + "=" * 80)
        logger.info("📈 EXPECTED METRICS")
        logger.info("=" * 80)

        # Define expected metrics based on model size
        model_metrics = {
            "pico": {"iou": "~0.15-0.20", "loss": "~1.3-1.5"},
            "nano": {"iou": "~0.18-0.25", "loss": "~1.2-1.4"},
            "tiny": {"iou": "~0.25-0.35", "loss": "~1.0-1.2"},
            "mpslight": {"iou": "~0.35-0.50", "loss": "~0.8-1.0"},
            "light": {"iou": "~0.40-0.55", "loss": "~0.6-0.8"},
            "mobile": {"iou": "~0.38-0.52", "loss": "~0.7-0.9"},
            "efficient": {"iou": "~0.42-0.58", "loss": "~0.5-0.7"},
            "unet": {"iou": "~0.45-0.62", "loss": "~0.4-0.6"},
        }

        for idx, (dataset, model, model_pair) in enumerate(all_combinations[:3], 1):
            metrics = model_metrics.get(
                model, {"iou": "~0.20-0.40", "loss": "~0.8-1.2"}
            )
            logger.info(f"\n  {idx}. {dataset} | {model}")
            logger.info("     Expected:")
            logger.info(f"     ├── train_loss: {metrics['loss']}")
            logger.info(f"     ├── val_loss: {metrics['loss']} (slightly higher)")
            logger.info(f"     ├── train_iou: {metrics['iou']}")
            logger.info(f"     ├── val_iou: {metrics['iou']} (slightly lower)")
            logger.info("     ├── train_ce_loss: ~0.4-0.6")
            logger.info("     ├── train_focal_loss: ~0.3-0.5")
            logger.info("     ├── train_dice_loss: ~0.2-0.4")
            logger.info("     └── train_loss_class_2: ~0.5-0.8 (most important)")

        if len(all_combinations) > 3:
            logger.info(f"\n  ... and {len(all_combinations) - 3} more combinations")

    # ============================================================
    # 5. SHOW EXPECTED OUTPUT
    # ============================================================
    if show_output:
        logger.info("\n" + "=" * 80)
        logger.info("📄 EXPECTED CONSOLE OUTPUT")
        logger.info("=" * 80)

        logger.info("""
  For each combination, you'll see:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  🔥 Warming up MPS shaders (first pass can take 2-10 minutes)...  │
  │  ✅ MPS warmup complete!                                           │
  │  Epoch 1/1 - Train Loss: 1.3534                                    │
  │    Train IoU: 0.1723, Train Acc: 0.5169                            │
  │  Epoch 1/1 - Val Loss: 1.3203                                      │
  │    Val IoU: 0.1700, Val Acc: 0.5100                                │
  │  📊 Memory: allocated=1.52 GB                                      │
  │  ⏱ Epoch duration: 284.1s                                         │
  │  ✅ Model checkpoint logged to MLflow                             │
  │  🏆 New champion model! Version 1 with val_loss 1.3203            │
  │  ✅ TRAINING COMPLETE!                                            │
  └─────────────────────────────────────────────────────────────────────┘

  Each run will be saved to:
  - MLflow: model_loss_sweep experiment
  - TensorBoard: runs/{dataset}/{model}/
  - Models: models/registry/{model}_{dataset}_best.pt
  - Logs: logs/YYYY-MM-DD/HH-MM-SS_*.log
        """)

    # ============================================================
    # 6. SUMMARY
    # ============================================================
    logger.info("\n" + "=" * 80)
    logger.info("📊 DRY RUN SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Total combinations: {len(all_combinations)}")
    logger.info(f"  Total model pairs: {len(model_pairs)}")
    logger.info(f"  Total datasets: {len(datasets)}")
    logger.info(f"  Models to train: {len({m for _, m, _ in all_combinations})}")
    logger.info(f"  Datasets to train: {len({d for d, _, _ in all_combinations})}")

    # Show which models are skipped
    skipped = []
    for model in {m for pair in model_pairs for m in pair}:
        for dataset in datasets:
            if model in SKIP_PER_DATASET.get(dataset, []):
                skipped.append((model, dataset))

    if skipped:
        logger.info("\n  ⚠️  Skipped combinations:")
        for model, dataset in skipped:
            logger.info(f"     - {model} on {dataset} (too large)")

    logger.info("\n  💡 To run this, remove --dry-run flag")
    logger.info("=" * 80)


@click.command()
@click.option("--epochs", "-e", default=2, help="Number of epochs")
@click.option("--device", "-d", default="mps", help="Device to use")
@click.option("--dry-run", is_flag=True, help="Print commands without running")
# ============================================================
# NEW DRY RUN OPTIONS
# ============================================================
@click.option("--show-mlflow", is_flag=True, help="Show MLflow run details")
@click.option("--show-functions", is_flag=True, help="Show function call stack")
@click.option("--show-output", is_flag=True, help="Show expected output")
@click.option("--show-metrics", is_flag=True, help="Show expected metrics")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--no-log-memory", is_flag=True, help="Disable memory logging")
def main(
    epochs,
    device,
    dry_run,
    show_mlflow,
    show_functions,
    show_output,
    show_metrics,
    verbose,
    no_log_memory,
):
    """Run model pairs with graceful fallback."""

    if dry_run:
        # Enhanced dry run
        run_dry_run(
            model_pairs=MODEL_PAIRS,
            datasets=DATASETS,
            epochs=epochs,
            device=device,
            show_mlflow=show_mlflow,
            show_functions=show_functions,
            show_output=show_output,
            show_metrics=show_metrics,
            verbose=verbose,
        )
        return

    # Normal run
    run_model_pairs_with_fallback(
        model_pairs=MODEL_PAIRS,
        datasets=DATASETS,
        epochs=epochs,
        device=device,
        verbose=verbose,
        log_memory=not no_log_memory,
        dry_run=False,
    )


if __name__ == "__main__":
    main()
