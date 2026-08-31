#!/usr/bin/env python3
"""
Search and compare MLflow models.
"""

import sys
import os
import click
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.mlflow_utils import get_mlflow_manager
from src.utils.logger import setup_logger

logger = setup_logger(task_name="search_models")


@click.command()
@click.option("--dataset", "-d", help="Filter by dataset name")
@click.option("--model-type", "-m", help="Filter by model type (UNet, MPSLightUNet, etc.)")
@click.option("--min-iou", type=float, help="Minimum IoU threshold")
@click.option("--top", "-n", default=10, help="Number of results to show")
@click.option("--compare", "-c", is_flag=True, help="Compare models side by side")
def main(dataset, model_type, min_iou, top, compare):
    """Search and compare MLflow models."""

    logger.info("=" * 60)
    logger.info("🔍 MLflow Model Search")
    logger.info("=" * 60)

    mlflow_manager = get_mlflow_manager()

    # Build filter
    filters = []
    if dataset:
        filters.append(f"tags.dataset = '{dataset}'")
    if model_type:
        filters.append(f"tags.model_type = '{model_type}'")
    if min_iou is not None:
        filters.append(f"metrics.val_iou > {min_iou}")

    filter_string = " AND ".join(filters) if filters else None

    # Search
    models = mlflow_manager.search_models(
        filter_string=filter_string,
        order_by=[{"field_name": "metrics.val_iou", "ascending": False}],
        max_results=top,
    )

    logger.info(f"\nFound {len(models)} models")

    if not models:
        logger.info("No models found matching criteria")
        return

    # Display results
    logger.info("\n" + "-" * 80)
    logger.info(f"{'#':<4} {'Model':<30} {'Dataset':<12} {'IoU':<8} {'F1':<8} {'Class2 IoU':<10}")
    logger.info("-" * 80)

    for i, model in enumerate(models[:top]):
        metrics = {m.key: m.value for m in model.metrics}
        tags = {t.key: t.value for t in model.tags}

        name = model.name[:28] if model.name else "Unknown"
        dataset_name = tags.get("dataset", "Unknown")
        iou = metrics.get("val_iou", 0)
        f1 = metrics.get("val_f1", 0)
        class2_iou = metrics.get("class_2_iou", 0)

        logger.info(f"{i+1:<4} {name:<30} {dataset_name:<12} {iou:.4f}  {f1:.4f}  {class2_iou:.4f}")

    logger.info("-" * 80)

    # Comparison view
    if compare and len(models) >= 2:
        logger.info("\n📊 Model Comparison:")
        logger.info("-" * 80)

        for i, model in enumerate(models[:2]):
            metrics = {m.key: m.value for m in model.metrics}
            tags = {t.key: t.value for t in model.tags}

            logger.info(f"\nModel {i+1}: {model.name}")
            logger.info(f"  Dataset: {tags.get('dataset', 'Unknown')}")
            logger.info(f"  Type: {tags.get('model_type', 'Unknown')}")
            logger.info(f"  IoU: {metrics.get('val_iou', 0):.4f}")
            logger.info(f"  F1: {metrics.get('val_f1', 0):.4f}")
            logger.info(f"  Accuracy: {metrics.get('val_accuracy', 0):.4f}")
            logger.info(f"  Class 2 IoU: {metrics.get('class_2_iou', 0):.4f}")

            # Get model URI
            logger.info(f"  URI: models:/{model.model_id}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Search complete!")


if __name__ == "__main__":
    main()