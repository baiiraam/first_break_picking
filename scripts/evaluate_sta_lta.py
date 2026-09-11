#!/usr/bin/env python3
# file location: scripts/evaluate_sta_lta.py

"""
Evaluate an STA/LTA configuration on real data and log to MLflow.

Loads N shots from a dataset, runs the picker, computes metrics,
saves per-trace predictions, and optionally generates 5 diagnostic
images. Logs everything to the seismic-fbp-baselines experiment.

Default behavior: first 5 shots (matches the sweep used to pick the
winner config), no images, no MLflow logging unless configured.

Usage:
    python scripts/evaluate_sta_lta.py \
        --config configs/halfmile.yaml \
        --sta-window 15 --lta-window 150 --threshold 3.0 \
        --phase baseline-v1.0-detailed \
        --save-images
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import BaselineEvaluator, BaselineImageGenerator, STALTAPicker
from src.config import SeismicConfig
from src.utils.hdf5_utils import load_shot_data, load_shot_indices
from src.utils.logger import create_task_name, setup_logger
from src.utils.mlflow_utils import get_mlflow_manager
from src.utils.tracking_conventions import EXPERIMENT_BASELINES


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
    """
    Decide which shot indices to use.

    Priority:
        1. --shots-from / --shots-to range (overrides everything)
        2. --seed: shuffle, take first n_shots
        3. default: first n_shots
    """
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
    sampling_interval_ms: float,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Load shots from HDF5. Returns a list of
    (shot_id, shot_data, gt_picks_in_samples).

    gt_picks are converted from ms to samples here.
    """
    unique_shots, start_indices, end_indices = load_shot_indices(cfg.hdf5_path)

    shots: list[tuple[int, np.ndarray, np.ndarray]] = []
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
            gt_picks_ms / sampling_interval_ms
        ).astype(np.int64)

        shots.append((shot_id, shot_data, gt_picks_samples))

    return shots


def build_run_name(dataset: str, phase: str) -> str:
    hhmmss = datetime.now(timezone.utc).strftime("%H%M%S")
    return f"STALTA_{dataset}_{phase}_{hhmmss}"


# ============================================================
# MAIN
# ============================================================

@click.command()
@click.option("--config", "-c", required=True, help="Path to config YAML")
@click.option("--dataset", "-ds", default=None,
              help="Override dataset name (for logging)")
@click.option("--sta-window", type=int, default=15,
              help="STA window in samples (default 15)")
@click.option("--lta-window", type=int, default=150,
              help="LTA window in samples (default 150)")
@click.option("--threshold", type=float, default=3.0,
              help="STA/LTA ratio threshold (default 3.0)")
@click.option("--shots", type=int, default=5,
              help="Number of shots to evaluate (default 5)")
@click.option("--seed", type=int, default=None,
              help="Shuffle shot order with this seed before selecting")
@click.option("--shots-from", type=int, default=None,
              help="Explicit start index for a shot range")
@click.option("--shots-to", type=int, default=None,
              help="Explicit end index for a shot range")
@click.option("--save-images", is_flag=True, default=False,
              help="Generate and log 5 diagnostic images")
@click.option("--dry-run", is_flag=True, default=False,
              help="Compute and print, do not log to MLflow")
@click.option("--phase", type=str, default="baseline-v1.0-detailed",
              help="MLflow phase tag")
def main(
    config: str,
    dataset: str | None,
    sta_window: int,
    lta_window: int,
    threshold: float,
    shots: int,
    seed: int | None,
    shots_from: int | None,
    shots_to: int | None,
    save_images: bool,
    dry_run: bool,
    phase: str,
):
    """Evaluate an STA/LTA configuration on real data."""

    # Load config
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = SeismicConfig(**config_dict)
    if dataset:
        cfg.dataset_name = dataset

    task_name = create_task_name(cfg, "eval_sta_lta")
    logger = setup_logger(task_name=task_name)

    logger.info("=" * 70)
    logger.info("STA/LTA EVALUATION")
    logger.info("=" * 70)
    logger.info(f"  Dataset:        {cfg.dataset_name}")
    logger.info(f"  HDF5:           {cfg.hdf5_path}")
    logger.info(f"  sta_window:     {sta_window}")
    logger.info(f"  lta_window:     {lta_window}")
    logger.info(f"  threshold:      {threshold}")
    logger.info(f"  shots:          {shots}")
    logger.info(f"  seed:           {seed}")
    logger.info(f"  shots range:    {shots_from}..{shots_to}")
    logger.info(f"  save_images:    {save_images}")
    logger.info(f"  dry_run:        {dry_run}")

    # Determine shot indices
    unique_shots, _, _ = load_shot_indices(cfg.hdf5_path)
    n_total = len(unique_shots)

    selected = select_shot_indices(
        n_total=n_total,
        n_shots=shots,
        seed=seed,
        shots_from=shots_from,
        shots_to=shots_to,
    )
    logger.info(f"  Selected shot indices: {selected[:10]}"
                f"{'...' if len(selected) > 10 else ''}")

    # Load shots
    shots_data = load_shots(cfg, selected, cfg.sampling_interval_ms)
    logger.info(f"  Loaded {len(shots_data)} shots")

    # Build picker and evaluator
    picker = STALTAPicker(
        sta_window=sta_window,
        lta_window=lta_window,
        threshold=threshold,
    )
    evaluator = BaselineEvaluator(picker, logger=logger)

    result = evaluator.evaluate_shots(shots_data)

    # Print metrics
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    m = result["metrics"]
    logger.info(f"  Traces evaluated:       {m['total_traces']}")
    logger.info(f"  MAE:                    {m['mean_absolute_error']:.2f} samples")
    logger.info(f"  Median error:           {m['median_absolute_error']:.2f} samples")
    logger.info(f"  Std error:              {m['std_absolute_error']:.2f} samples")
    logger.info(f"  ±3 accuracy:            {m['accuracy_within_tolerance']*100:.2f}%")
    logger.info(f"  Predicted picks:        {int((result['per_trace']['pred_pick_sample'] > 0).sum())}")
    logger.info(f"  Valid comparisons:      {len(result['per_trace'])}")

    # Save per-trace CSV
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"sta_lta_eval_{cfg.dataset_name}_{ts}.csv"
    result["per_trace"].to_csv(csv_path, index=False)
    logger.info(f"  Per-trace CSV:          {csv_path}")

    # Images
    images_dir = None
    if save_images:
        images_dir = output_dir / f"sta_lta_images_{cfg.dataset_name}_{ts}"
        gen = BaselineImageGenerator(output_dir=images_dir, logger=logger)
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
        experiment_name=EXPERIMENT_BASELINES,
        enable_system_metrics=False,
        enable_autolog=False,
    )

    run_name = build_run_name(cfg.dataset_name, phase)
    tags = {
        "dataset": cfg.dataset_name,
        "model_type": "STALTA",
        "phase": phase,
        "env": "research",
        "sta_window": str(sta_window),
        "lta_window": str(lta_window),
        "threshold": str(threshold),
    }
    params: dict[str, Any] = {
        "sta_window": sta_window,
        "lta_window": lta_window,
        "threshold": threshold,
        "n_shots": len(shots_data),
    }
    metrics = {
        "mae_samples": float(m["mean_absolute_error"]),
        "median_samples": float(m["median_absolute_error"]),
        "std_samples": float(m["std_absolute_error"]),
        "within_3_accuracy": float(m["accuracy_within_tolerance"]),
        "predicted_picks": int((result["per_trace"]["pred_pick_sample"] > 0).sum()),
        "matched_traces": int(len(result["per_trace"])),
        "total_traces": int(m["total_traces"]),
        "n_shots": len(shots_data),
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
            mlflow_manager.log_artifact(str(img), artifact_path="images")

    mlflow_manager.end_run()

    logger.info("")
    logger.info(f"✅ Logged to MLflow: {run_name}")
    logger.info(f"   Experiment: {EXPERIMENT_BASELINES}")
    logger.info("")


if __name__ == "__main__":
    main()