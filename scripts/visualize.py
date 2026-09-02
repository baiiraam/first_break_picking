#!/usr/bin/env python3
"""
Visualization script for seismic FBP results.
"""

import os
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SeismicConfig
from src.data.chunked_dataset import ChunkedDataManager
from src.models.factory import create_model
from src.preprocessing.manifest import load_manifest
from src.utils.logger import create_task_name, setup_logger


@click.command()
@click.option("--config", "-c", required=True, help="Path to config YAML file")
@click.option("--model", "-m", required=True, help="Path to model checkpoint (.pt)")
@click.option(
    "--output", "-o", default="visualization_results", help="Output directory for plots"
)
@click.option("--n_samples", "-n", default=10, help="Number of samples to visualize")
@click.option("--device", "-d", default="mps", help="Device to use (cpu/cuda/mps)")
def main(config: str, model: str, output: str, n_samples: int, device: str):
    """Visualize model predictions on test samples."""

    # Load config
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = SeismicConfig(**config_dict)
    cfg.device = device

    # Setup logger with task name
    task_name = create_task_name(cfg, "visualize")
    logger = setup_logger(task_name=task_name)

    logger.info("=" * 60)
    logger.info("SEISMIC FBP - VISUALIZATION")
    logger.info("=" * 60)
    logger.info(f"Dataset: {cfg.dataset_name}")
    logger.info(f"Model: {model}")
    logger.info(f"Samples: {n_samples}")

    # Load manifest
    chunk_dir = Path(cfg.chunk_dir) / cfg.dataset_name
    manifest_path = chunk_dir / "manifest.json"

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = load_manifest(manifest_path)

    # Create data manager and test dataset
    data_manager = ChunkedDataManager(
        chunk_dir=chunk_dir, manifest=manifest, cache_size=2, shuffle_chunks=False
    )

    test_dataset = data_manager.get_dataset("test")

    logger.info(f"\nTest set: {len(test_dataset)} shots")

    # Load model
    device_obj = torch.device(cfg.device)

    # Determine model type from checkpoint or config
    model_type = getattr(cfg, 'model_name', 'mpslight')
    model_obj = create_model(model_type, in_channels=1, out_channels=3)

    checkpoint = torch.load(model, map_location=device_obj)
    if "model_state_dict" in checkpoint:
        model_obj.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_obj.load_state_dict(checkpoint)

    model_obj = model_obj.to(device_obj)
    model_obj.eval()

    # Visualize samples
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nGenerating visualizations...")

    n_to_plot = min(n_samples, len(test_dataset))

    with torch.no_grad():
        for idx in range(n_to_plot):
            shot_id = test_dataset.get_shot_id(idx)
            data, mask = test_dataset[idx]

            # Add batch dimension
            data_batch = data.unsqueeze(0).to(device_obj)

            # Predict
            output = model_obj(data_batch)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]

            # Convert to numpy
            data_np = data.numpy()[0]  # (1578, 751)
            mask_np = mask.numpy()  # (1578, 751)

            # Create visualization
            _, axes = plt.subplots(1, 3, figsize=(18, 8))

            # Original seismogram
            ax1 = axes[0]
            ax1.imshow(
                data_np.T,
                cmap="seismic",
                aspect="auto",
                vmin=-np.percentile(np.abs(data_np), 95),
                vmax=np.percentile(np.abs(data_np), 95),
            )
            ax1.set_title(f"Seismogram (Shot {shot_id})")
            ax1.set_xlabel("Trace")
            ax1.set_ylabel("Sample")

            # Ground truth mask
            ax2 = axes[1]
            im2 = ax2.imshow(mask_np.T, cmap="tab10", aspect="auto", vmin=0, vmax=2)
            ax2.set_title("Ground Truth Mask")
            ax2.set_xlabel("Trace")
            ax2.set_ylabel("Sample")
            plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2], label="Class")

            # Prediction
            ax3 = axes[2]
            im3 = ax3.imshow(pred.T, cmap="tab10", aspect="auto", vmin=0, vmax=2)
            ax3.set_title("Prediction")
            ax3.set_xlabel("Trace")
            ax3.set_ylabel("Sample")
            plt.colorbar(im3, ax=ax3, ticks=[0, 1, 2], label="Class")

            plt.tight_layout()
            plt.savefig(output_dir / f"shot_{shot_id}_comparison.png", dpi=150)
            plt.close()

            logger.info(f"  Saved: shot_{shot_id}_comparison.png")

    logger.info(f"\n✅ Visualizations saved to: {output_dir}")
    logger.info("\n" + "=" * 60)
    logger.info("✅ VISUALIZATION COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
