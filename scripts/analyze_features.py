#!/usr/bin/env python3
# file location: scripts/analyze_features.py

"""
Feature analysis for a trained conv model.

Produces:
    - weight_histograms.png
    - activation_statistics.png
    - first_layer_filters.png
    - kernel_similarity_<layer>.png (one per analyzed layer)
    - feature_analysis_summary.csv

Logs to seismic-fbp-explainability.

Usage:
    python scripts/analyze_features.py \
        --checkpoint models/registry/MPSLightUNet_Halfmile_best.pt \
        --config configs/halfmile.yaml \
        --n-shots 5 \
        --phase feature-analysis-v1.0
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SeismicConfig
from src.data.chunked_dataset import ChunkedDataManager
from src.explainability import (
    ActivationStatisticsAnalyzer,
    FirstLayerFilterAnalyzer,
    KernelSimilarityAnalyzer,
    WeightHistogramAnalyzer,
)
from src.models.loader import load_evaluation_model
from src.preprocessing.manifest import load_manifest
from src.utils.logger import create_task_name, setup_logger
from src.utils.mlflow_utils import get_mlflow_manager
from src.utils.tracking_conventions import (
    EXPERIMENT_EXPLAINABILITY,
    explainability_tags,
)


def build_run_name(dataset: str, phase: str) -> str:
    hhmmss = datetime.now(timezone.utc).strftime("%H%M%S")
    return f"FEATANALYZE_{dataset}_{phase}_{hhmmss}"


@click.command()
@click.option("--checkpoint", "-c", required=True, help="Model checkpoint")
@click.option("--config", "-cfg", required=True, help="Config YAML")
@click.option("--split", "-s", default="test",
              type=click.Choice(["train", "val", "test"]))
@click.option("--n-shots", "-n", type=int, default=5,
              help="Number of shots for activation statistics")
@click.option("--no-mlflow", is_flag=True, default=False)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--phase", default="feature-analysis-v1.0")
def main(
    checkpoint: str,
    config: str,
    split: str,
    n_shots: int,
    no_mlflow: bool,
    dry_run: bool,
    phase: str,
):
    """Run feature analysis on a trained model."""
    with open(config) as f:
        cfg = SeismicConfig(**yaml.safe_load(f))

    logger = setup_logger(task_name=create_task_name(cfg, "featanalyze"))

    logger.info("=" * 70)
    logger.info("FEATURE ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"  Checkpoint: {checkpoint}")
    logger.info(f"  Dataset:    {cfg.dataset_name}")
    logger.info(f"  Split:      {split}")
    logger.info(f"  N shots:    {n_shots}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"  Device:     {device}")

    # Load model
    logger.info("")
    logger.info("Loading model...")
    model = load_evaluation_model(checkpoint, cfg, device, logger)

    # Output directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("evaluation_results") / (
        f"feature_analysis_{cfg.dataset_name}_{ts}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Weight histograms ---
    logger.info("")
    logger.info("Computing weight histograms...")
    wha = WeightHistogramAnalyzer(logger=logger)
    weight_stats = wha.analyze(model)
    wha.plot(model, output_dir)

    # --- First-layer filters ---
    logger.info("")
    logger.info("Rendering first-layer filters...")
    fla = FirstLayerFilterAnalyzer(logger=logger)
    fla.plot(model, output_dir)

    # --- Kernel similarity ---
    logger.info("")
    logger.info("Computing kernel similarity...")
    ksa = KernelSimilarityAnalyzer(logger=logger)
    sim_stats = ksa.analyze(model)
    ksa.plot(model, output_dir, max_layers=4)

    # --- Activation statistics (requires shots) ---
    logger.info("")
    logger.info(f"Loading {n_shots} shots for activation statistics...")
    chunk_dir = Path(cfg.chunk_dir) / cfg.dataset_name
    manifest_path = chunk_dir / "manifest.json"
    if not manifest_path.exists():
        logger.warning(f"Manifest not found: {manifest_path}; skipping activations")
        act_stats = {}
    else:
        manifest = load_manifest(manifest_path)
        dm = ChunkedDataManager(
            chunk_dir=str(chunk_dir),
            manifest=manifest,
            cache_size=2,
            shuffle_chunks=False,
        )
        dataset = dm.get_dataset(split)
        shots = []
        for i in range(min(n_shots, len(dataset))):
            x, _ = dataset[i]
            shots.append(x.squeeze(0).cpu().numpy())  # (H, W)
        asa = ActivationStatisticsAnalyzer(logger=logger, max_shots=n_shots)
        act_stats = asa.analyze(model, shots, device)
        asa.plot(act_stats, output_dir)

    # --- Summary CSV ---
    import pandas as pd
    rows = []
    for name in weight_stats:
        rows.append({
            "layer": name,
            "n_params": weight_stats[name]["n_params"],
            "weight_std": weight_stats[name]["std"],
            "weight_frac_near_zero": weight_stats[name]["frac_near_zero"],
            "n_channels": act_stats.get(name, {}).get("n_channels", None),
            "activation_dead_frac": act_stats.get(name, {}).get("overall_dead_frac", None),
            "high_similarity_pairs": sim_stats.get(name, {}).get("high_similarity_pairs", None),
            "mean_similarity": sim_stats.get(name, {}).get("mean_similarity", None),
        })
    summary_df = pd.DataFrame(rows)
    csv_path = output_dir / "feature_analysis_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")

    logger.info("")
    logger.info(f"✅ All outputs in {output_dir}")

    if dry_run or no_mlflow:
        logger.info("Skipping MLflow.")
        return

    # --- MLflow ---
    mlflow_manager = get_mlflow_manager(
        experiment_name=EXPERIMENT_EXPLAINABILITY,
        enable_system_metrics=False,
        enable_autolog=False,
    )

    run_name = build_run_name(cfg.dataset_name, phase)
    tags = explainability_tags(
        dataset=cfg.dataset_name,
        model_type="MPSLightUNet",
        split=split,
        phase=phase,
        env="research",
    )
    tags["analysis_type"] = "feature_analysis"

    # Overall metrics
    metrics: dict[str, float] = {}
    if act_stats:
        dead_fracs = [
            s["overall_dead_frac"] for s in act_stats.values()
        ]
        metrics["mean_dead_channel_frac"] = float(np.mean(dead_fracs))
        metrics["max_dead_channel_frac"] = float(np.max(dead_fracs))

    if sim_stats:
        sims = [s["mean_similarity"] for s in sim_stats.values()]
        metrics["mean_kernel_similarity"] = float(np.mean(sims))
        high_pairs = sum(s["high_similarity_pairs"] for s in sim_stats.values())
        metrics["total_high_similarity_pairs"] = float(high_pairs)

    params = {
        "checkpoint": checkpoint,
        "split": split,
        "n_shots": n_shots,
    }

    mlflow_manager.start_run(
        config_dict=params,
        run_name=run_name,
        tags=tags,
    )
    if metrics:
        mlflow_manager.log_metrics(metrics, step=0)
    mlflow_manager.log_artifact(str(csv_path), artifact_path="feature_analysis")
    for png in sorted(output_dir.glob("*.png")):
        mlflow_manager.log_artifact(str(png), artifact_path="feature_analysis")
    mlflow_manager.end_run()

    logger.info("")
    logger.info(f"✅ Logged to MLflow: {run_name}")
    logger.info(f"   Experiment: {EXPERIMENT_EXPLAINABILITY}")
    logger.info("")


if __name__ == "__main__":
    main()