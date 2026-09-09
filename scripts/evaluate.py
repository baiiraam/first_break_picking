#!/usr/bin/env python3
"""
Evaluation script for trained seismic FBP model.
Thin CLI wrapper for the evaluation pipeline.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import torch
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SeismicConfig
from src.data.chunked_dataset import ChunkedDataManager
from src.evaluation import EvaluationRunner, ResultExporter
from src.models.loader import load_evaluation_model
from src.preprocessing.manifest import load_manifest
from src.utils.logger import create_task_name, setup_logger


@click.command()
@click.option("--config", "-c", required=True, help="Path to config YAML file")
@click.option(
    "--model",
    "-m",
    required=True,
    help="Path to model checkpoint (.pt) or 'best' for MLflow champion",
)
@click.option(
    "--output", "-o", default="evaluation_results", help="Output directory for results"
)
@click.option("--device", "-d", default="mps", help="Device to use (cpu/cuda/mps)")
@click.option("--batch_size", "-b", default=4, help="Batch size for evaluation")
@click.option("--dataset", "-ds", help="Override dataset name (for logging)")
@click.option(
    "--split",
    "-s",
    default="test",
    type=click.Choice(["train", "val", "test", "all"]),
    help="Which split to evaluate",
)
@click.option("--detailed", is_flag=True, help="Generate detailed per-shot metrics")
def main(
    config: str,
    model: str,
    output: str,
    device: str,
    batch_size: int,
    dataset: str,
    split: str,
    detailed: bool,
):
    """Evaluate the trained model on specified set."""

    # Load config
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = SeismicConfig(**config_dict)
    cfg.device = device
    cfg.batch_size = batch_size

    if dataset:
        cfg.dataset_name = dataset

    task_name = create_task_name(cfg, "evaluate")
    logger = setup_logger(task_name=task_name)

    logger.info("=" * 60)
    logger.info("SEISMIC FBP - EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Dataset: {cfg.dataset_name}")
    logger.info(f"Model: {model}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Split: {split}")

    # Load manifest
    chunk_dir = Path(cfg.chunk_dir) / cfg.dataset_name
    manifest_path = chunk_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = load_manifest(manifest_path)

    # Create data manager
    data_manager = ChunkedDataManager(
        chunk_dir=str(chunk_dir),
        manifest=manifest,
        cache_size=2,
        shuffle_chunks=False,
    )

    splits = ["train", "val", "test"] if split == "all" else [split]
    device_obj = torch.device(cfg.device)

    # Load model
    try:
        model_obj = load_evaluation_model(model, cfg, device_obj, logger)
    except ValueError as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Run evaluation
    runner = EvaluationRunner(
        model=model_obj,
        device_obj=device_obj,
        cfg=cfg,
        logger=logger,
        detailed=detailed,
    )

    all_results: dict[str, dict[str, Any]] = {}
    all_detailed_results: list[dict[str, Any]] = []

    for split_name in splits:
        dataset_obj = data_manager.get_dataset(split_name)
        loader = torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=cfg.device == "cuda",
        )

        metrics, detailed_results = runner.evaluate_split(
            loader=loader,
            dataset_obj=dataset_obj,
            split_name=split_name,
        )

        all_results.update(metrics)
        if detailed_results:
            all_detailed_results.append(detailed_results)

    # Export results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exporter = ResultExporter(output_dir=Path(output), logger=logger)
    exporter.print_summary_table(all_results)

    exporter.export_all(
        dataset_name=cfg.dataset_name,
        model_path=model,
        device_str=str(device_obj),
        timestamp=timestamp,
        all_results=all_results,
        all_detailed_results=all_detailed_results,
        detailed=detailed,
    )

    logger.info("\n" + "=" * 60)
    logger.info("✅ EVALUATION COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
