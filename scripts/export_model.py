#!/usr/bin/env python3
# file location: scripts/export_model.py

"""
Export trained model to TorchScript and ONNX.

Uses src.deployment.exporters for the actual export logic, and
src.deployment.validators to verify numeric equivalence.

Usage:
    python scripts/export_model.py \
        --model models/registry/MPSLightUNet_Halfmile_best.pt \
        --model-type mpslight \
        --config configs/halfmile.yaml \
        --output exported_models \
        --torchscript --onnx
"""

import os
import sys
from pathlib import Path

import click
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SeismicConfig
from src.deployment import read_contract_from_checkpoint, verify_numeric_equivalence
from src.deployment.exporters import export_onnx, export_torchscript
from src.models.factory import create_model, get_available_models
from src.utils.logger import create_task_name, setup_logger


@click.command()
@click.option("--model", "-m", required=True, help="Path to model checkpoint (.pt)")
@click.option("--output", "-o", default="exported_models", help="Output directory")
@click.option("--onnx", is_flag=True, help="Export to ONNX format")
@click.option("--torchscript", is_flag=True, help="Export to TorchScript format")
@click.option("--device", "-d", default="cpu", help="Device for export")
@click.option(
    "--model-type",
    "-t",
    type=click.Choice(get_available_models()),
    default=None,
    help="Model architecture (auto-detected from checkpoint if omitted)",
)
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to config YAML (optional)",
)
@click.option(
    "--verify/--no-verify",
    default=True,
    help="Verify numeric equivalence after export",
)
def main(
    model: str,
    output: str,
    onnx: bool,
    torchscript: bool,
    device: str,
    model_type: str | None,
    config: str | None,
    verify: bool,
):
    """Export a trained model to production formats."""

    if not onnx and not torchscript:
        print("ERROR: please specify at least one of --onnx or --torchscript")
        sys.exit(1)

    # Setup logger
    cfg = None
    if config:
        with open(config, "r") as f:
            cfg = SeismicConfig(**yaml.safe_load(f))
        task_name = create_task_name(cfg, "export")
    else:
        task_name = "export_model"

    logger = setup_logger(task_name=task_name)

    logger.info("=" * 70)
    logger.info("SEISMIC FBP — MODEL EXPORT")
    logger.info("=" * 70)
    logger.info(f"  Model:       {model}")
    logger.info(f"  Model type:  {model_type or 'auto'}")
    logger.info(f"  Output:      {output}")
    logger.info(f"  Device:      {device}")
    logger.info(f"  TorchScript: {torchscript}")
    logger.info(f"  ONNX:        {onnx}")
    logger.info(f"  Verify:      {verify}")

    # --- Read contract from checkpoint ---
    logger.info("")
    logger.info("Reading contract from checkpoint...")
    contract = read_contract_from_checkpoint(model)
    logger.info(f"  {contract.describe()}")

    # --- Instantiate the model architecture ---
    logger.info("")
    logger.info("Instantiating model architecture...")

    # Try to infer model_type from checkpoint if not provided
    if model_type is None:
        checkpoint = torch.load(model, map_location="cpu", weights_only=False)
        model_type = checkpoint.get("model_key") or checkpoint.get("model_name")
        if isinstance(model_type, str):
            model_type = model_type.lower().replace("unet", "").strip("_") or "unet"
        logger.info(f"  Auto-detected: {model_type}")

    model_obj, model_name = create_model(model_type)
    logger.info(f"  Created: {model_name}")

    # --- Load weights ---
    logger.info("")
    logger.info("Loading checkpoint weights...")
    checkpoint = torch.load(model, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model_obj.load_state_dict(state_dict)
    model_obj = model_obj.to(torch.device(device))
    model_obj.eval()
    logger.info("  ✅ Weights loaded")

    # --- Output directory ---
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_path = None
    onnx_path = None

    # --- TorchScript ---
    if torchscript:
        logger.info("")
        logger.info("📦 Exporting to TorchScript...")
        ts_path = output_dir / f"{model_type}_model_scripted.pt"
        try:
            export_torchscript(
                model=model_obj,
                contract=contract,
                output_path=ts_path,
                logger=logger,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"  ❌ TorchScript export failed: {e}")
            ts_path = None

    # --- ONNX ---
    if onnx:
        logger.info("")
        logger.info("📦 Exporting to ONNX...")
        onnx_path = output_dir / f"{model_type}_model.onnx"
        try:
            export_onnx(
                model=model_obj,
                contract=contract,
                output_path=onnx_path,
                logger=logger,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"  ❌ ONNX export failed: {e}")
            onnx_path = None

    # --- Verify ---
    if verify and (ts_path or onnx_path):
        logger.info("")
        logger.info("🔬 Verifying numeric equivalence...")
        x = torch.randn(*contract.expected_shape)
        results = verify_numeric_equivalence(
            pytorch_model=model_obj,
            input_tensor=x,
            torchscript_path=ts_path,
            onnx_path=onnx_path,
        )
        for backend, res in results.items():
            status = "✅" if res["match"] else "❌"
            if res.get("error"):
                logger.info(f"  {status} {backend}: {res['error']}")
            else:
                logger.info(
                    f"  {status} {backend}: max_diff={res['max_diff']:.2e}"
                )

    # --- Summary ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ EXPORT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Output directory: {output_dir}")

    if ts_path:
        logger.info(f"  TorchScript: {ts_path}")
    if onnx_path:
        logger.info(f"  ONNX:        {onnx_path}")

    logger.info("")
    logger.info("To use the exported model:")
    if ts_path:
        logger.info(
            f"  TorchScript: torch.jit.load('{ts_path}')"
        )
    if onnx_path:
        logger.info(
            f"  ONNX: import onnxruntime as ort; "
            f"session = ort.InferenceSession('{onnx_path}')"
        )


if __name__ == "__main__":
    main()