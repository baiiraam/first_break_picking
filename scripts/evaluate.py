#!/usr/bin/env python3

# file location: scripts/evaluate.py

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
from src.utils.mlflow_utils import get_mlflow_manager
from src.utils.tracking_conventions import evaluation_tags


def _log_evaluation_to_mlflow(
    cfg,
    model_path: str,
    split_results: dict,
    device_str: str,
    output_dir,
    detailed: bool,
    logger,
    images_dir=None,
) -> None:
    """
    Log evaluation results to MLflow.

    Creates a run in the 'seismic-fbp-evaluation' experiment.
    Logs all segmentation and first-break metrics with 'eval_{split}_' prefix.
    Uploads JSON and CSV artifacts.

    Gracefully degrades if MLflow is unavailable.
    """
    try:
        mlflow_manager = get_mlflow_manager(
            experiment_name="seismic-fbp-evaluation",
            enable_system_metrics=False,
            enable_autolog=False,
        )

        # Start a new run
        config_dict = {
            "dataset": cfg.dataset_name,
            "model_path": model_path,
            "device": device_str,
            "splits_evaluated": list(split_results.keys()),
        }

        # Determine model type from the model path for tagging.
        # Match longest name first (e.g., "MPSLightUNet" before "UNet")
        # so a path containing "mpslightunet" doesn't get tagged "UNet".
        model_type_tag = "unknown"
        try:
            from src.models.loader import MODEL_NAME_TO_KEY

            for display_name in sorted(MODEL_NAME_TO_KEY.keys(), key=len, reverse=True):
                if display_name.lower() in model_path.lower():
                    model_type_tag = display_name
                    break
        except (ImportError, AttributeError):
            pass  # keep "unknown"

        tags = evaluation_tags(
            dataset=cfg.dataset_name,
            model_type=model_type_tag,
            phase=cfg.phase,
            env="research",
        )
        tags["model_path"] = model_path
        tags["device"] = device_str

        mlflow_manager.start_run(
            config_dict=config_dict,
            tags=tags,
        )

        # Log metrics for each split
        all_metrics = {}
        for split_name, results in split_results.items():
            seg = results["segmentation"]
            fb = results["first_break"]

            prefix = f"eval_{split_name}_"

            # Segmentation metrics
            all_metrics[f"{prefix}accuracy"] = seg["accuracy"]
            all_metrics[f"{prefix}mean_iou"] = seg["mean_iou"]
            all_metrics[f"{prefix}mean_f1"] = seg["mean_f1"]
            all_metrics[f"{prefix}iou_before"] = seg["iou_per_class"][0]
            all_metrics[f"{prefix}iou_after"] = seg["iou_per_class"][1]
            all_metrics[f"{prefix}iou_strip"] = seg["iou_per_class"][2]

            # First-break metrics
            all_metrics[f"{prefix}mae_samples"] = fb["mean_absolute_error"]
            all_metrics[f"{prefix}median_error"] = fb["median_absolute_error"]
            all_metrics[f"{prefix}std_error"] = fb["std_absolute_error"]
            all_metrics[f"{prefix}max_error"] = fb["max_absolute_error"]
            all_metrics[f"{prefix}min_error"] = fb["min_absolute_error"]
            all_metrics[f"{prefix}accuracy_within_3"] = fb["accuracy_within_tolerance"]
            all_metrics[f"{prefix}total_traces"] = fb["total_traces"]

        mlflow_manager.log_metrics(all_metrics, step=0)
        logger.info(f"✅ Logged {len(all_metrics)} metrics to MLflow")

        # Log artifacts

        output_path = Path(output_dir)
        for artifact_path in output_path.glob(f"*{cfg.dataset_name}*.json"):
            mlflow_manager.log_artifact(str(artifact_path), artifact_path="evaluation")
        for artifact_path in output_path.glob(f"*{cfg.dataset_name}*.csv"):
            mlflow_manager.log_artifact(str(artifact_path), artifact_path="evaluation")

        # Log evaluation images if provided
        if images_dir is not None:
            images_dir = Path(images_dir)
            if images_dir.exists():
                n_images = 0
                for image_path in images_dir.glob("*.png"):
                    mlflow_manager.log_artifact(
                        str(image_path), artifact_path="evaluation/images"
                    )
                    n_images += 1
                logger.info(f"✅ Logged {n_images} image(s) to MLflow")
            else:
                logger.warning(
                    f"⚠️  images_dir provided but does not exist: {images_dir}"
                )

        logger.info("✅ Logged artifacts to MLflow")

        mlflow_manager.end_run()
        logger.info("✅ MLflow run ended")

    except Exception as e:
        logger.warning(f"⚠️  MLflow logging failed (non-critical): {e}")
        logger.warning("Evaluation completed successfully; only MLflow logging failed.")


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
@click.option(
    "--save-images",
    is_flag=True,
    default=False,
    help="Generate and log evaluation images (shot comparisons, "
    "error histogram, error vs. position). Adds ~15-30s.",
)
@click.option(
    "--phase",
    type=str,
    default=None,
    help="MLflow phase tag (e.g., 'baseline-v1.0'). Default: 'unset'.",
)
def main(
    config: str,
    model: str,
    output: str,
    device: str,
    batch_size: int,
    dataset: str,
    split: str,
    detailed: bool,
    phase: str | None,
    save_images: bool,
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
    if phase:
        cfg.phase = phase

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

    # === OPTIONAL: image generation ===
    images_dir = None
    if save_images:
        from src.evaluation.image_generator import EvaluationImageGenerator

        images_dir = Path(output) / f"images_{cfg.dataset_name}_{timestamp}"
        generator = EvaluationImageGenerator(output_dir=images_dir, logger=logger)

        # Build a per-split DataFrame of shot_errors from detailed results.
        # The detailed_results dicts contain a 'dataframe' with shot_id and
        # error_samples columns.
        for i, split_name in enumerate(splits):
            if i < len(all_detailed_results) and all_detailed_results[i]:
                det = all_detailed_results[i]
                if isinstance(det, dict) and "dataframe" in det:
                    generator.generate_for_split(
                        split_name=split_name,
                        shot_errors=det["dataframe"],
                        model=model_obj,
                        device=device_obj,
                        data_manager=data_manager,
                        cfg=cfg,
                    )

        logger.info(f"✅ Images saved to: {images_dir}")

    # ✅ NEW: Log evaluation results to MLflow
    _log_evaluation_to_mlflow(
        cfg=cfg,
        model_path=model,
        split_results=all_results,
        device_str=str(device_obj),
        output_dir=output,
        detailed=detailed,
        logger=logger,
        images_dir=images_dir if save_images else None,  # ← NEW
    )

    logger.info("\n" + "=" * 60)
    logger.info("✅ EVALUATION COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
