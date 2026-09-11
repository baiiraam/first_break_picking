#!/usr/bin/env python3
# file location: scripts/compare_baselines.py

"""
Compare ML model picks and STA/LTA picks on the same shots.

Loads N shots from a dataset, runs both methods, produces 6 comparison
figures, a per-trace CSV, and logs everything to the
seismic-fbp-comparison MLflow experiment.

Usage:
    python scripts/compare_baselines.py \
        --config configs/halfmile.yaml \
        --ml-model models/registry/MPSLightUNet_Halfmile_best.pt \
        --sta-window 15 --lta-window 150 --threshold 3.0 \
        --shots 5 \
        --phase baseline-comparison-v1.0
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import STALTAPicker
from src.config import SeismicConfig
from src.evaluation import BaselineComparison, ComparisonImageGenerator
from src.models.loader import load_evaluation_model
from src.utils.hdf5_utils import load_shot_data, load_shot_indices
from src.utils.logger import create_task_name, setup_logger
from src.utils.mlflow_utils import get_mlflow_manager
from src.utils.tracking_conventions import (
    EXPERIMENT_COMPARISON,
    comparison_tags,
)


# ============================================================
# HELPERS
# ============================================================

def select_shot_indices(
    n_total: int,
    n_shots: int,
    seed: int | None,
    shots_from: int | None,
    shots_to: int | None,
) -> list[int]:
    if shots_from is not None and shots_to is not None:
        return list(range(shots_from, min(shots_to + 1, n_total)))
    indices = list(range(n_total))
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    return indices[:n_shots]


def load_shots(
    cfg: SeismicConfig,
    shot_indices: list[int],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    unique_shots, start_indices, end_indices = load_shot_indices(cfg.hdf5_path)
    shots = []
    for idx in shot_indices:
        if idx >= len(unique_shots):
            continue
        shot_id = int(unique_shots[idx])
        start = int(start_indices[idx])
        end = int(end_indices[idx])
        shot_data, gt_picks_ms = load_shot_data(
            cfg.hdf5_path, start, end,
            cfg.target_traces, cfg.n_samples,
        )
        gt_picks_samples = np.round(
            gt_picks_ms / cfg.sampling_interval_ms
        ).astype(np.int64)
        shots.append((shot_id, shot_data, gt_picks_samples))
    return shots


def build_run_name(dataset: str, phase: str) -> str:
    hhmmss = datetime.now(timezone.utc).strftime("%H%M%S")
    return f"COMPARE_{dataset}_{phase}_{hhmmss}"


def make_sta_config_label(sta_window: int, lta_window: int, threshold: float) -> str:
    return f"sta{sta_window}_lta{lta_window}_thr{threshold}"


# ============================================================
# MAIN
# ============================================================

@click.command()
@click.option("--config", "-c", required=True, help="Config YAML path")
@click.option("--ml-model", "-m", required=True,
              help="Path to ML model checkpoint")
@click.option("--dataset", "-ds", default=None,
              help="Override dataset name")
@click.option("--sta-window", type=int, default=15)
@click.option("--lta-window", type=int, default=150)
@click.option("--threshold", type=float, default=3.0)
@click.option("--shots", type=int, default=5)
@click.option("--seed", type=int, default=None)
@click.option("--shots-from", type=int, default=None)
@click.option("--shots-to", type=int, default=None)
@click.option("--save-images", is_flag=True, default=True,
              help="Generate and log the 6 comparison images (default: on)")
@click.option("--no-images", is_flag=True, default=False,
              help="Disable image generation")
@click.option("--dry-run", is_flag=True, default=False,
              help="Compute and print, do not log to MLflow")
@click.option("--phase", type=str, default="baseline-comparison-v1.0")
def main(
    config: str,
    ml_model: str,
    dataset: str | None,
    sta_window: int,
    lta_window: int,
    threshold: float,
    shots: int,
    seed: int | None,
    shots_from: int | None,
    shots_to: int | None,
    save_images: bool,
    no_images: bool,
    dry_run: bool,
    phase: str,
):
    """Compare ML and STA/LTA picks on the same shots."""
    if no_images:
        save_images = False

    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = SeismicConfig(**config_dict)
    if dataset:
        cfg.dataset_name = dataset

    task_name = create_task_name(cfg, "compare")
    logger = setup_logger(task_name=task_name)

    logger.info("=" * 70)
    logger.info("BASELINE COMPARISON — ML vs STA/LTA")
    logger.info("=" * 70)
    logger.info(f"  Dataset:        {cfg.dataset_name}")
    logger.info(f"  ML model:       {ml_model}")
    logger.info(f"  STA/LTA:        sta={sta_window}, lta={lta_window}, thr={threshold}")
    logger.info(f"  Shots:          {shots}")
    logger.info(f"  Seed:           {seed}")
    logger.info(f"  Shots range:    {shots_from}..{shots_to}")
    logger.info(f"  Save images:    {save_images}")
    logger.info(f"  Dry run:        {dry_run}")

    # --- Device ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"  Device:         {device}")

    # --- Load ML model ---
    logger.info("")
    logger.info("Loading ML model...")
    ml_model_obj = load_evaluation_model(ml_model, cfg, device, logger)

    # --- Build STA/LTA picker ---
    sta_picker = STALTAPicker(
        sta_window=sta_window,
        lta_window=lta_window,
        threshold=threshold,
    )

    # --- Select and load shots ---
    unique_shots, _, _ = load_shot_indices(cfg.hdf5_path)
    selected = select_shot_indices(
        n_total=len(unique_shots),
        n_shots=shots,
        seed=seed,
        shots_from=shots_from,
        shots_to=shots_to,
    )
    logger.info("")
    logger.info(f"Loading {len(selected)} shots...")
    shots_data = load_shots(cfg, selected)
    logger.info(f"Loaded {len(shots_data)} shots")

    # --- Run comparison ---
    comparison = BaselineComparison(
        ml_model=ml_model_obj,
        sta_picker=sta_picker,
        logger=logger,
        device=device,
    )
    result = comparison.compare_shots(shots_data)

    # --- Print metrics ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    m = result["ml_metrics"]
    s = result["sta_lta_metrics"]
    logger.info(f"  Traces evaluated:       {m['total_traces']}")
    logger.info(f"  Valid comparisons (ML): {m['total_traces']}")
    logger.info("")
    logger.info("  ┌─────────────────┬───────────┬───────────┐")
    logger.info("  │ Metric          │ STA/LTA   │ ML        │")
    logger.info("  ├─────────────────┼───────────┼───────────┤")
    logger.info(
        f"  │ MAE (samples)   │ {s['mean_absolute_error']:>9.2f} │ "
        f"{m['mean_absolute_error']:>9.2f} │"
    )
    logger.info(
        f"  │ Median          │ {s['median_absolute_error']:>9.2f} │ "
        f"{m['median_absolute_error']:>9.2f} │"
    )
    logger.info(
        f"  │ ±3 accuracy     │ {s['accuracy_within_tolerance']*100:>8.1f}% │ "
        f"{m['accuracy_within_tolerance']*100:>8.1f}% │"
    )
    logger.info("  └─────────────────┴───────────┴───────────┘")

    # --- Save per-trace CSV ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"comparison_{cfg.dataset_name}_{ts}.csv"
    result["per_trace"].to_csv(csv_path, index=False)
    logger.info(f"  Per-trace CSV:          {csv_path}")

    # --- Images ---
    images_dir = None
    if save_images:
        images_dir = output_dir / f"comparison_images_{cfg.dataset_name}_{ts}"
        gen = ComparisonImageGenerator(output_dir=images_dir, logger=logger)
        image_paths = gen.generate_all(
            result=result,
            shots=shots_data,
            dataset_name=cfg.dataset_name,
        )
        logger.info(f"  Images written:         {len(image_paths)} → {images_dir}")

    if dry_run:
        logger.info("")
        logger.info("DRY RUN — not logging to MLflow")
        return

    # --- Log to MLflow ---
    mlflow_manager = get_mlflow_manager(
        experiment_name=EXPERIMENT_COMPARISON,
        enable_system_metrics=False,
        enable_autolog=False,
    )

    run_name = build_run_name(cfg.dataset_name, phase)
    sta_label = make_sta_config_label(sta_window, lta_window, threshold)
    ml_model_label = Path(ml_model).stem

    tags = comparison_tags(
        dataset=cfg.dataset_name,
        ml_model=ml_model_label,
        sta_config=sta_label,
        phase=phase,
        env="research",
    )

    params: dict[str, Any] = {
        "sta_window": sta_window,
        "lta_window": lta_window,
        "threshold": threshold,
        "n_shots": len(shots_data),
        "ml_model_path": ml_model,
    }

    ml_mae = float(m["mean_absolute_error"])
    sta_mae = float(s["mean_absolute_error"])
    ml_within_3 = float(m["accuracy_within_tolerance"])
    sta_within_3 = float(s["accuracy_within_tolerance"])

    metrics = {
        # ML
        "ml_mae_samples": ml_mae,
        "ml_median_samples": float(m["median_absolute_error"]),
        "ml_std_samples": float(m["std_absolute_error"]),
        "ml_within_3_accuracy": ml_within_3,
        # STA/LTA
        "sta_lta_mae_samples": sta_mae,
        "sta_lta_median_samples": float(s["median_absolute_error"]),
        "sta_lta_std_samples": float(s["std_absolute_error"]),
        "sta_lta_within_3_accuracy": sta_within_3,
        # Comparative (ml - sta_lta)
        "ml_minus_sta_lta_mae": ml_mae - sta_mae,
        "ml_minus_sta_lta_within_3": ml_within_3 - sta_within_3,
    }

    mlflow_manager.start_run(
        config_dict=params,
        run_name=run_name,
        tags=tags,
    )
    mlflow_manager.log_metrics(metrics, step=0)
    mlflow_manager.log_artifact(str(csv_path), artifact_path="predictions")
    if images_dir is not None:
        for img in sorted(images_dir.glob("*.png")):
            mlflow_manager.log_artifact(str(img), artifact_path="comparison")
    mlflow_manager.end_run()

    logger.info("")
    logger.info(f"✅ Logged to MLflow: {run_name}")
    logger.info(f"   Experiment: {EXPERIMENT_COMPARISON}")
    logger.info("")


if __name__ == "__main__":
    main()