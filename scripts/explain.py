#!/usr/bin/env python3
# file location: scripts/explain.py

"""
Grad-CAM explainability for ML models.

Selects best/median/worst shots by mean per-trace error, runs Grad-CAM,
generates 4-panel figures, logs them to the seismic-fbp-explainability
MLflow experiment.

Usage:
    python scripts/explain.py \
        --config configs/halfmile.yaml \
        --model models/registry/MPSLightUNet_Halfmile_best.pt \
        --split test \
        --phase explain-v1.0
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

from src.config import SeismicConfig
from src.data.chunked_dataset import ChunkedDataManager
from src.explainability import (
    ExplainabilityImageGenerator,
    ExplainabilityRunner,
    GradCAM,
)
from src.models.loader import load_evaluation_model
from src.preprocessing.manifest import load_manifest
from src.training.metrics import extract_picks_from_mask
from src.utils.logger import create_task_name, setup_logger
from src.utils.mlflow_utils import get_mlflow_manager
from src.utils.tracking_conventions import (
    EXPERIMENT_EXPLAINABILITY,
    explainability_tags,
)

# ============================================================
# HELPERS
# ============================================================

def compute_per_shot_errors(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    logger,
) -> dict[int, dict]:
    """
    Run the model over the dataset and compute per-shot mean errors.

    Returns:
        dict: shot_id -> {"mean_error": float, "n_valid": int}
    """
    per_shot: dict[int, dict[str, Any]] = {}
    model.eval()

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]  # x: (1, H, W), y: (H, W)
            xb = x.unsqueeze(0).to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
            gt = y.cpu().numpy()

            pred_picks = extract_picks_from_mask(pred)
            gt_picks = extract_picks_from_mask(gt)

            mask = (pred_picks > 0) & (gt_picks > 0)
            n_valid = int(mask.sum())
            if n_valid > 0:
                mean_err = float(
                    np.abs(pred_picks[mask] - gt_picks[mask]).mean()
                )
            else:
                mean_err = float("inf")

            shot_id = int(dataset.get_shot_id(i))
            per_shot[shot_id] = {
                "mean_error": mean_err,
                "n_valid": n_valid,
                "dataset_index": i,
            }

    return per_shot


def select_best_median_worst(
    per_shot: dict[int, dict],
) -> list[tuple[int, str, float]]:
    """
    Given per-shot stats, return [(shot_id, label, mean_error), ...]
    for best, median, worst (in that order).
    """
    # Filter to shots with at least one valid comparison
    valid = {
        sid: stats for sid, stats in per_shot.items()
        if stats["n_valid"] > 0
    }
    if len(valid) < 3:
        raise ValueError(
            f"Need at least 3 valid shots to pick best/median/worst, "
            f"got {len(valid)}"
        )

    sorted_items = sorted(valid.items(), key=lambda kv: kv[1]["mean_error"])
    n = len(sorted_items)

    best_id, best_stats = sorted_items[0]
    median_id, median_stats = sorted_items[n // 2]
    worst_id, worst_stats = sorted_items[-1]

    return [
        (best_id, "best", best_stats["mean_error"]),
        (median_id, "median", median_stats["mean_error"]),
        (worst_id, "worst", worst_stats["mean_error"]),
    ]


def build_run_name(dataset: str, phase: str) -> str:
    hhmmss = datetime.now(timezone.utc).strftime("%H%M%S")
    return f"EXPLAIN_{dataset}_{phase}_{hhmmss}"


# ============================================================
# MAIN
# ============================================================

@click.command()
@click.option("--config", "-c", required=True, help="Config YAML path")
@click.option("--model", "-m", required=True,
              help="Path to ML model checkpoint")
@click.option("--split", "-s", default="test",
              type=click.Choice(["train", "val", "test"]),
              help="Which split to draw shots from (default: test)")
@click.option("--method", default="gradcam",
              type=click.Choice(["gradcam", "hirescam"]),
              help="Explainability method")
@click.option("--target-class", type=int, default=2,
              help="Class to explain (default: 2 = strip)")
@click.option("--no-mlflow", is_flag=True, default=False,
              help="Skip MLflow logging")
@click.option("--dry-run", is_flag=True, default=False,
              help="Generate figures, don't log to MLflow")
@click.option("--phase", type=str, default="explain-v1.0",
              help="MLflow phase tag")
def main(
    config: str,
    model: str,
    split: str,
    method: str,
    target_class: int,
    no_mlflow: bool,
    dry_run: bool,
    phase: str,
):
    """Run Grad-CAM explainability on best/median/worst shots."""

    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = SeismicConfig(**config_dict)

    task_name = create_task_name(cfg, "explain")
    logger = setup_logger(task_name=task_name)

    logger.info("=" * 70)
    logger.info("EXPLAINABILITY — Grad-CAM")
    logger.info("=" * 70)
    logger.info(f"  Dataset:        {cfg.dataset_name}")
    logger.info(f"  Model:          {model}")
    logger.info(f"  Split:          {split}")
    logger.info(f"  Method:         {method}")
    logger.info(f"  Target class:   {target_class}")

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"  Device:         {device}")

    # Load model
    logger.info("")
    logger.info("Loading model...")
    model_obj = load_evaluation_model(model, cfg, device, logger)

    # Load dataset
    logger.info("")
    logger.info(f"Loading {split} split...")
    chunk_dir = Path(cfg.chunk_dir) / cfg.dataset_name
    manifest_path = chunk_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = load_manifest(manifest_path)
    data_manager = ChunkedDataManager(
        chunk_dir=str(chunk_dir),
        manifest=manifest,
        cache_size=2,
        shuffle_chunks=False,
    )
    dataset = data_manager.get_dataset(split)
    logger.info(f"  Dataset size: {len(dataset)} shots")

    # Compute per-shot mean errors
    logger.info("")
    logger.info("Computing per-shot errors...")
    per_shot = compute_per_shot_errors(model_obj, dataset, device, logger)

    # Select best/median/worst
    selections = select_best_median_worst(per_shot)
    logger.info("")
    logger.info("Selected shots:")
    for sid, label, err in selections:
        logger.info(f"  {label:>8}: shot_id={sid}, mean_error={err:.2f}")

    # Load the selected shots
    shots_for_explain = []
    shot_ids = []
    labels = []
    mean_errors = []
    for sid, label, err in selections:
        idx = per_shot[sid]["dataset_index"]
        x, y = dataset[idx]
        shot_data = x.squeeze(0).cpu().numpy()  # (H, W)
        gt_mask = y.cpu().numpy()               # (H, W)
        shots_for_explain.append((shot_data, gt_mask))
        shot_ids.append(sid)
        labels.append(label)
        mean_errors.append(err)

    # Set up explainer and generator
    explainer = GradCAM(method=method)
    output_dir = Path("evaluation_results") / (
        f"explainability_images_{cfg.dataset_name}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    image_generator = ExplainabilityImageGenerator(
        output_dir=output_dir, logger=logger
    )

    runner = ExplainabilityRunner(
        model=model_obj,
        explainer=explainer,
        device=device,
        logger=logger,
        target_class=target_class,
    )

    logger.info("")
    logger.info("Running Grad-CAM...")
    paths = runner.explain_shots(
        shot_ids=shot_ids,
        shots=shots_for_explain,
        labels=labels,
        mean_errors=mean_errors,
        image_generator=image_generator,
        dataset_name=cfg.dataset_name,
    )

    logger.info("")
    logger.info(f"✅ Generated {len(paths)} figures in {output_dir}")

    if dry_run or no_mlflow:
        logger.info("Skipping MLflow logging.")
        return

    # MLflow logging
    mlflow_manager = get_mlflow_manager(
        experiment_name=EXPERIMENT_EXPLAINABILITY,
        enable_system_metrics=False,
        enable_autolog=False,
    )

    run_name = build_run_name(cfg.dataset_name, phase)
    tags = explainability_tags(
        dataset=cfg.dataset_name,
        model_type="MPSLightUNet",  # will be inferred in a later version
        split=split,
        phase=phase,
        env="research",
    )
    tags["method"] = method
    tags["target_class"] = str(target_class)

    params = {
        "model_path": model,
        "split": split,
        "method": method,
        "target_class": target_class,
    }

    mlflow_manager.start_run(
        config_dict=params,
        run_name=run_name,
        tags=tags,
    )

    for path in paths:
        mlflow_manager.log_artifact(str(path), artifact_path="explainability")

    mlflow_manager.end_run()

    logger.info("")
    logger.info(f"✅ Logged to MLflow: {run_name}")
    logger.info(f"   Experiment: {EXPERIMENT_EXPLAINABILITY}")
    logger.info("")


if __name__ == "__main__":
    main()