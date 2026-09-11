#!/usr/bin/env python3
# file location: scripts/diagnose.py

"""
Diagnostics: wavefront coherence and worst-N error galleries.

Runs the ML model and STA/LTA on every shot in a split, then produces:
    - coherence_trajectories.png
    - coherence_distribution.png
    - worst_gallery.png       (worst ML traces)
    - worst_gallery_sta.png   (worst STA/LTA traces)
    - coherence_metrics.csv

Logs to seismic-fbp-explainability.

Usage:
    python scripts/diagnose.py \
        --config configs/halfmile.yaml \
        --model models/registry/MPSLightUNet_Halfmile_best.pt \
        --split test \
        --phase diagnose-v1.0
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import STALTAPicker
from src.config import SeismicConfig
from src.data.chunked_dataset import ChunkedDataManager
from src.explainability import CoherenceAnalyzer, ErrorGalleryGenerator
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

def build_per_trace_df(
    model: torch.nn.Module,
    sta_picker: STALTAPicker,
    dataset,
    device: torch.device,
    logger,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """
    Run ML and STA/LTA on every shot in the dataset. Build a
    per-trace DataFrame and a dict of shot_id -> seismogram.
    """
    rows: list[dict] = []
    shots: dict[int, np.ndarray] = {}

    model.eval()

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]  # x: (1, H, W), y: (H, W)
            shot_id = int(dataset.get_shot_id(i))

            shot_data = x.squeeze(0).cpu().numpy()   # (H, W)
            gt_mask = y.cpu().numpy()                # (H, W)
            shots[shot_id] = shot_data

            # ML prediction
            xb = x.unsqueeze(0).to(device)  # (1, 1, H, W)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()

            ml_picks = extract_picks_from_mask(pred)
            gt_picks = extract_picks_from_mask(gt_mask)
            sta_picks = sta_picker.pick(shot_data)

            n_traces = shot_data.shape[0]
            for t in range(n_traces):
                gt = int(gt_picks[t])
                if gt <= 0:
                    continue
                ml = int(ml_picks[t])
                sta = int(sta_picks[t])
                rows.append({
                    "shot_id": shot_id,
                    "trace_index": t,
                    "gt_pick_sample": gt,
                    "sta_lta_pick": sta,
                    "sta_lta_error_samples": abs(sta - gt) if sta > 0 else -1,
                    "ml_pick": ml,
                    "ml_error_samples": abs(ml - gt) if ml > 0 else -1,
                })

    df = pd.DataFrame(rows)
    logger.info(
        f"[Diagnose] Built per-trace DataFrame: {len(df)} rows "
        f"across {len(shots)} shots"
    )
    return df, shots


def select_representative_shots(
    metrics: pd.DataFrame,
) -> list[tuple[int, str]]:
    """
    Pick 3 shots by coherence: most coherent, median, least coherent.
    """
    df = metrics.copy()
    df = df[np.isfinite(df["ml_coherence"])]
    if len(df) < 3:
        raise ValueError(
            f"Need at least 3 shots with finite coherence, got {len(df)}"
        )
    df = df.sort_values("ml_coherence")
    n = len(df)
    return [
        (int(df.iloc[0]["shot_id"]), "most coherent ML"),
        (int(df.iloc[n // 2]["shot_id"]), "median coherence ML"),
        (int(df.iloc[-1]["shot_id"]), "least coherent ML"),
    ]


def build_run_name(dataset: str, phase: str) -> str:
    hhmmss = datetime.now(timezone.utc).strftime("%H%M%S")
    return f"DIAGNOSE_{dataset}_{phase}_{hhmmss}"


# ============================================================
# MAIN
# ============================================================

@click.command()
@click.option("--config", "-c", required=True, help="Config YAML path")
@click.option("--model", "-m", required=True, help="ML model checkpoint")
@click.option("--split", "-s", default="test",
              type=click.Choice(["train", "val", "test"]))
@click.option("--sta-window", type=int, default=15)
@click.option("--lta-window", type=int, default=150)
@click.option("--threshold", type=float, default=3.0)
@click.option("--n-worst", type=int, default=20,
              help="How many traces in each worst-N gallery")
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--phase", type=str, default="diagnose-v1.0")
def main(
    config: str,
    model: str,
    split: str,
    sta_window: int,
    lta_window: int,
    threshold: float,
    n_worst: int,
    dry_run: bool,
    phase: str,
):
    """Run diagnostics: wavefront coherence and worst-N galleries."""

    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = SeismicConfig(**config_dict)

    logger = setup_logger(task_name=create_task_name(cfg, "diagnose"))

    logger.info("=" * 70)
    logger.info("DIAGNOSTICS — coherence + worst-N galleries")
    logger.info("=" * 70)
    logger.info(f"  Dataset:        {cfg.dataset_name}")
    logger.info(f"  Model:          {model}")
    logger.info(f"  Split:          {split}")
    logger.info(f"  STA/LTA:        sta={sta_window}, lta={lta_window}, thr={threshold}")
    logger.info(f"  N worst:        {n_worst}")

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

    # Build per-trace DataFrame
    logger.info("")
    logger.info("Running ML and STA/LTA on all shots...")
    sta_picker = STALTAPicker(
        sta_window=sta_window,
        lta_window=lta_window,
        threshold=threshold,
    )
    per_trace, shots = build_per_trace_df(
        model=model_obj,
        sta_picker=sta_picker,
        dataset=dataset,
        device=device,
        logger=logger,
    )

    # Output directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("evaluation_results") / (
        f"diagnostics_{cfg.dataset_name}_{ts}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Coherence
    logger.info("")
    logger.info("Computing coherence metrics...")
    coherence = CoherenceAnalyzer(logger=logger)
    metrics_df = coherence.compute_metrics(per_trace)

    csv_path = output_dir / "coherence_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")

    # Trajectories
    logger.info("")
    logger.info("Generating figures...")
    selections = select_representative_shots(metrics_df)
    coherence.plot_trajectories(
        per_trace=per_trace,
        selections=selections,
        output_dir=output_dir,
        n_samples=cfg.n_samples,
    )
    coherence.plot_distribution(metrics_df, output_dir)

    # Galleries
    gallery = ErrorGalleryGenerator(output_dir=output_dir, logger=logger)
    gallery.generate_gallery(
        per_trace=per_trace,
        shots=shots,
        method="ml",
        n_worst=n_worst,
        output_filename="worst_gallery.png",
    )
    gallery.generate_gallery(
        per_trace=per_trace,
        shots=shots,
        method="sta_lta",
        n_worst=n_worst,
        output_filename="worst_gallery_sta.png",
    )

    logger.info("")
    logger.info(f"✅ All outputs in {output_dir}")

    if dry_run:
        logger.info("Dry run — not logging to MLflow.")
        return

    # MLflow
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
    tags["diagnostic_type"] = "coherence_and_gallery"

    params = {
        "model_path": model,
        "split": split,
        "n_worst": n_worst,
        "sta_window": sta_window,
        "lta_window": lta_window,
        "threshold": threshold,
    }

    # Summary metrics
    ml_coh = metrics_df["ml_coherence"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    gt_coh = metrics_df["gt_coherence"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    sta_coh = metrics_df["sta_coherence"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    ml_metrics = {
        "coherence_ml_median": float(ml_coh.median()) if len(ml_coh) else 0.0,
        "coherence_gt_median": float(gt_coh.median()) if len(gt_coh) else 0.0,
        "coherence_sta_median": float(sta_coh.median()) if len(sta_coh) else 0.0,
        "n_shots": int(len(metrics_df)),
        "n_traces": int(len(per_trace)),
    }

    mlflow_manager.start_run(
        config_dict=params,
        run_name=run_name,
        tags=tags,
    )
    mlflow_manager.log_metrics(ml_metrics, step=0)
    mlflow_manager.log_artifact(str(csv_path), artifact_path="diagnostics")
    for png in sorted(output_dir.glob("*.png")):
        mlflow_manager.log_artifact(str(png), artifact_path="diagnostics")
    mlflow_manager.end_run()

    logger.info("")
    logger.info(f"✅ Logged to MLflow: {run_name}")
    logger.info(f"   Experiment: {EXPERIMENT_EXPLAINABILITY}")
    logger.info("")


if __name__ == "__main__":
    main()