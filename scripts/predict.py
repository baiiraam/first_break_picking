#!/usr/bin/env python3
"""
Inference script for trained models on new seismic data.
"""

from pathlib import Path

import click
import h5py
import numpy as np
import torch
import yaml

from src.config import SeismicConfig
from src.models.loader import load_model_from_checkpoint  # ✅ NEW
from src.training.metrics import extract_picks_from_mask
from src.utils.logger import setup_logger

logger = setup_logger(task_name="predict")


@click.command()
@click.option("--model", "-m", required=True, help="Model path or MLflow URI")
@click.option("--input", "-i", required=True, help="Input HDF5 file path")
@click.option("--output", "-o", required=True, help="Output picks file (.npy or .csv)")
@click.option("--config", "-c", required=True, help="Config file for data shape")
@click.option("--device", "-d", default="mps")
@click.option("--batch-size", "-b", default=4)
@click.option("--chunk-size", default=100, help="Process in chunks for memory efficiency")
@click.option(
    "--model-type",
    "-t",
    type=click.Choice(["unet", "mpslight", "light", "nano", "tiny", "pico", "mobile", "efficient"]),
    default="mpslight",
    help="Model architecture type",
)
def main(model, input, output, config, device, batch_size, chunk_size, model_type):
    """Run inference on new seismic data."""

    # 1. Load config
    with open(config, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = SeismicConfig(**config_dict)

    device_obj = torch.device(device)

    # 2. Load model using unified loader
    if model.startswith("models:/"):
        import mlflow
        model_obj = mlflow.pytorch.load_model(model)
        model_obj = model_obj.to(device_obj)
    else:
        model_obj = load_model_from_checkpoint(
            model_path=model,
            model_type=model_type,
            device=device_obj,
        )

    model_obj.eval()
    logger.info("✅ Model loaded successfully")

    # 3. Load data
    with h5py.File(input, "r") as f:
        data = f["TRACE_DATA"]["DEFAULT"]["data_array"][:]

    logger.info(f"📊 Loaded {data.shape[0]} traces, {data.shape[1]} samples")

    # 4. Process in chunks
    n_traces = data.shape[0]
    all_picks = []

    for i in range(0, n_traces, chunk_size):
        chunk = data[i : i + chunk_size]
        chunk = torch.from_numpy(chunk).float().unsqueeze(1).to(device_obj)

        with torch.no_grad():
            output_tensor = model_obj(chunk)
            pred = torch.argmax(output_tensor, dim=1).cpu().numpy()

        picks = extract_picks_from_mask(pred)
        all_picks.extend(picks)

        if (i + chunk_size) % (chunk_size * 10) == 0:
            logger.info(
                f"⏳ Processed {min(i + chunk_size, n_traces)}/{n_traces} traces"
            )

    # 5. Save results
    all_picks = np.array(all_picks)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output.endswith(".npy"):
        np.save(output_path, all_picks)
    else:
        import pandas as pd

        df = pd.DataFrame(
            {"trace_idx": range(len(all_picks)), "pick_sample": all_picks}
        )
        df.to_csv(output_path, index=False)

    logger.info(f"✅ Saved {len(all_picks)} picks to {output}")


if __name__ == "__main__":
    main()
