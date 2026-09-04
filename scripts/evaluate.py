"""
Evaluation script for trained seismic FBP model.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SeismicConfig
from src.data.chunked_dataset import ChunkedDataManager
from src.models.loader import load_model_from_checkpoint  # ✅ NEW
from src.preprocessing.manifest import load_manifest
from src.training.metrics import (
    FirstBreakMetrics,
    SegmentationMetrics,
    extract_picks_from_mask,
)
from src.utils.logger import create_task_name, setup_logger
from src.utils.mlflow_utils import format_registered_model_name, get_mlflow_manager


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
    "--model-type",
    "-t",
    type=click.Choice(["unet", "mpslight", "light", "nano", "tiny", "pico", "mobile", "efficient"]),
    default=None,
    help="Model architecture type override",
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
    model_type: str
):
    """Evaluate the trained model on test set."""

    # Load config
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = SeismicConfig(**config_dict)
    cfg.device = device
    cfg.batch_size = batch_size

    # Override dataset name if provided
    if dataset:
        cfg.dataset_name = dataset

    # Setup logger with dynamic task name
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
        chunk_dir=chunk_dir,
        manifest=manifest,
        cache_size=2,
        shuffle_chunks=False,
    )

    # Get requested split(s)
    if split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [split]

    device_obj = torch.device(cfg.device)

    # ============================================================
    # LOAD MODEL - Using unified loader
    # ============================================================
    if model == "best":
        logger.info("🔍 Searching for best model...")
        mlflow_manager = get_mlflow_manager()

        # Try to get champion alias
        registered_name = format_registered_model_name(cfg.dataset_name)
        champion = mlflow_manager.get_model_by_alias(
            registered_model_name=registered_name,
            alias="champion",
        )

        if champion:
            model_uri = f"models:/{registered_name}@champion"
            logger.info(f"Found champion model: {model_uri}")
            model_obj = mlflow.pytorch.load_model(model_uri)
            model_obj = model_obj.to(device_obj)
        else:
            # Search by metrics
            best_models = mlflow_manager.search_models(
                filter_string=f"tags.dataset = '{cfg.dataset_name}'",
                order_by=[{"field_name": "metrics.val_iou", "ascending": False}],
                max_results=1,
            )
            if best_models:
                model_uri = f"models:/{best_models[0].model_id}"
                logger.info(f"Found best model by IoU: {model_uri}")
                model_obj = mlflow.pytorch.load_model(model_uri)
                model_obj = model_obj.to(device_obj)
            else:
                logger.error(f"No model found for dataset '{cfg.dataset_name}'")
                sys.exit(1)

        logger.info("✅ Model loaded successfully from MLflow")

    else:
        # ✅ Use unified loader for local and MLflow checkpoints
        # Determine model type
        if hasattr(cfg, "model_name"):
            model_type = cfg.model_name
        else:
            # Try to infer from file name or default
            model_type = "mpslight"

        logger.info(f"Loading model with type: {model_type}")

        if model.startswith("models:/"):
            # Load from MLflow URI
            try:
                logger.info(f"Loading model from MLflow: {model}")
                model_obj = mlflow.pytorch.load_model(model)
                model_obj = model_obj.to(device_obj)
                logger.info("✅ Model loaded successfully from MLflow")
            except Exception as e:
                logger.error(f"Failed to load from MLflow: {e}")
                sys.exit(1)
        else:
            # Load from file using unified loader
            # Regular file path
            try:
                logger.info(f"Loading model from file: {model}")
                
                # ✅ Use model_type if provided, else infer from config
                if model_type:
                    arch_type = model_type
                elif hasattr(cfg, "model_name"):
                    arch_type = cfg.model_name
                else:
                    arch_type = "mpslight"
                
                model_obj = load_model_from_checkpoint(
                    model_path=model,
                    model_type=arch_type,
                    device=device_obj,
                )
                logger.info("✅ Model loaded successfully from file")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                sys.exit(1)

    model_obj.eval()
    logger.info("\n✅ Model loaded")

    # ============================================================
    # EVALUATE EACH SPLIT
    # ============================================================
    all_results = {}
    all_detailed_results = []

    for split_name in splits:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 Evaluating {split_name.upper()} SET")
        logger.info(f"{'=' * 60}")

        dataset_obj = data_manager.get_dataset(split_name)
        loader = torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=cfg.device == "cuda",
        )

        logger.info(f"  {split_name}: {len(dataset_obj)} shots, {len(loader)} batches")

        # Initialize metrics for this split
        seg_metrics = SegmentationMetrics(num_classes=3)
        fb_metrics = FirstBreakMetrics(tolerance_samples=3)

        # For detailed per-shot metrics
        shot_errors = []
        shot_ids = []

        logger.info("\nRunning evaluation...")

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(
                tqdm(loader, desc=f"Evaluating {split_name}")
            ):
                x, y = x.to(device_obj), y.to(device_obj)

                outputs = model_obj(x)
                preds = torch.argmax(outputs, dim=1)

                # Update segmentation metrics
                seg_metrics.update(preds, y)

                # Extract picks for first-break metrics
                pred_picks = extract_picks_from_mask(preds.cpu().numpy())
                true_picks = extract_picks_from_mask(y.cpu().numpy())
                fb_metrics.update(pred_picks, true_picks)

                # Per-shot detailed metrics
                if detailed:
                    for i in range(len(pred_picks)):
                        if true_picks[i] > 0 and pred_picks[i] > 0:
                            error = abs(pred_picks[i] - true_picks[i])
                            shot_errors.append(error)

                            # Get shot ID if available
                            try:
                                shot_id = dataset_obj.get_shot_id(
                                    batch_idx * cfg.batch_size + i
                                )
                                shot_ids.append(shot_id)
                            except (AttributeError, IndexError, KeyError) as e:
                                logger.debug(f"Could not get shot_id: {e}")
                                shot_ids.append(batch_idx * cfg.batch_size + i)

        # Compute metrics
        seg_results = seg_metrics.compute()
        fb_results = fb_metrics.compute()

        # Store results
        all_results[split_name] = {
            "segmentation": seg_results,
            "first_break": fb_results,
            "n_shots": len(dataset_obj),
        }

        # Log results
        logger.info(f"\n📊 SEGMENTATION METRICS ({split_name.upper()})")
        logger.info("-" * 40)
        logger.info(f"  Accuracy: {seg_results['accuracy']:.4f}")
        logger.info(f"  Mean IoU: {seg_results['mean_iou']:.4f}")
        logger.info(f"  Mean F1: {seg_results['mean_f1']:.4f}")

        logger.info("\n  Class-wise IoU:")
        class_names = ["Before", "After", "Strip"]
        for i, (name, iou) in enumerate(zip(class_names, seg_results["iou_per_class"])):
            logger.info(f"    {name}: {iou:.4f}")

        logger.info(f"\n📊 FIRST-BREAK METRICS ({split_name.upper()})")
        logger.info("-" * 40)
        logger.info(
            f"  Mean Absolute Error (MAE): {fb_results['mean_absolute_error']:.2f} samples"
        )
        logger.info(
            f"  Std Absolute Error: {fb_results['std_absolute_error']:.2f} samples"
        )
        logger.info(
            f"  Median Absolute Error: {fb_results['median_absolute_error']:.2f} samples"
        )
        logger.info(
            f"  Accuracy within ±3 samples: {fb_results['accuracy_within_tolerance']:.2%}"
        )
        logger.info(f"  Total traces evaluated: {fb_results['total_traces']}")

        # Detailed error distribution
        if detailed and shot_errors:
            logger.info("\n📊 ERROR DISTRIBUTION")
            logger.info("-" * 40)
            errors_array = np.array(shot_errors)
            percentiles = [50, 75, 90, 95, 99]
            logger.info(f"  Min: {errors_array.min():.2f}")
            logger.info(f"  Max: {errors_array.max():.2f}")
            for p in percentiles:
                logger.info(f"  {p}th percentile: {np.percentile(errors_array, p):.2f}")
            logger.info(f"  >5 samples: {(errors_array > 5).mean() * 100:.1f}%")
            logger.info(f"  >10 samples: {(errors_array > 10).mean() * 100:.1f}%")

            # Save detailed results
            detailed_df = pd.DataFrame(
                {
                    "shot_id": shot_ids,
                    "error_samples": shot_errors,
                    "error_ms": np.array(shot_errors) * 2,
                }
            )
            all_detailed_results.append(
                {
                    "split": split_name,
                    "dataframe": detailed_df,
                }
            )

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("=" * 60)

    logger.info(
        f"\n{'Split':<10} {'IoU':<10} {'F1':<10} {'MAE (samples)':<15} {'Acc ±3':<10}"
    )
    logger.info("-" * 60)

    for split_name, results in all_results.items():
        seg = results["segmentation"]
        fb = results["first_break"]
        logger.info(
            f"{split_name:<10} {seg['mean_iou']:.4f}   {seg['mean_f1']:.4f}   {fb['mean_absolute_error']:>10.2f}   {fb['accuracy_within_tolerance']:>8.1%}"
        )

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    results_to_save = {
        "timestamp": timestamp,
        "dataset": cfg.dataset_name,
        "model_path": model,
        "device": str(device_obj),
        "split_results": all_results,
    }

    # Convert numpy arrays to lists for JSON serialization
    for split_name in results_to_save["split_results"]:
        for metric_type in ["segmentation", "first_break"]:
            for key, value in results_to_save["split_results"][split_name][
                metric_type
            ].items():
                if isinstance(value, np.ndarray):
                    results_to_save["split_results"][split_name][metric_type][key] = (
                        value.tolist()
                    )
                elif isinstance(value, (np.float32, np.float64)):
                    results_to_save["split_results"][split_name][metric_type][key] = (
                        float(value)
                    )
                elif isinstance(value, (np.int64, np.int32)):
                    results_to_save["split_results"][split_name][metric_type][key] = (
                        int(value)
                    )

    json_path = output_dir / f"evaluation_results_{cfg.dataset_name}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    logger.info(f"\n✅ JSON results saved to: {json_path}")

    # Save detailed CSV
    if detailed and all_detailed_results:
        for det in all_detailed_results:
            csv_path = (
                output_dir
                / f"detailed_errors_{cfg.dataset_name}_{det['split']}_{timestamp}.csv"
            )
            det["dataframe"].to_csv(csv_path, index=False)
            logger.info(f"✅ Detailed errors saved to: {csv_path}")

    # Save summary CSV
    summary_data = []
    for split_name, results in all_results.items():
        seg = results["segmentation"]
        fb = results["first_break"]
        summary_data.append(
            {
                "split": split_name,
                "n_shots": results["n_shots"],
                "accuracy": seg["accuracy"],
                "mean_iou": seg["mean_iou"],
                "mean_f1": seg["mean_f1"],
                "iou_before": seg["iou_per_class"][0],
                "iou_after": seg["iou_per_class"][1],
                "iou_strip": seg["iou_per_class"][2],
                "mae_samples": fb["mean_absolute_error"],
                "std_error": fb["std_absolute_error"],
                "median_error": fb["median_absolute_error"],
                "accuracy_within_tolerance": fb["accuracy_within_tolerance"],
                "total_traces": fb["total_traces"],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = (
        output_dir / f"evaluation_summary_{cfg.dataset_name}_{timestamp}.csv"
    )
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"✅ Summary saved to: {summary_csv_path}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ EVALUATION COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
