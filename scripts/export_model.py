#!/usr/bin/env python3
"""
Export trained model to ONNX and TorchScript formats.
"""

import os
import sys
import yaml
import torch
from pathlib import Path
import click

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import SeismicConfig
from src.utils.logger import setup_logger
from src.models.unet import UNet

logger = setup_logger()


@click.command()
@click.option('--model', '-m', required=True, help='Path to model checkpoint (.pt)')
@click.option('--output', '-o', default='exported_models', help='Output directory')
@click.option('--onnx', is_flag=True, help='Export to ONNX format')
@click.option('--torchscript', is_flag=True, help='Export to TorchScript format')
@click.option('--device', '-d', default='cpu', help='Device for export (cpu/cuda/mps)')
def main(model: str, output: str, onnx: bool, torchscript: bool, device: str):
    """Export trained model to production formats."""
    
    if not onnx and not torchscript:
        logger.error("Please specify at least one export format: --onnx or --torchscript")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("SEISMIC FBP - MODEL EXPORT")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Output: {output}")
    
    # Load model
    device_obj = torch.device(device)
    model_obj = UNet(in_channels=1, out_channels=3)
    
    checkpoint = torch.load(model, map_location=device_obj)
    if 'model_state_dict' in checkpoint:
        model_obj.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_obj.load_state_dict(checkpoint)
    
    model_obj = model_obj.to(device_obj)
    model_obj.eval()
    
    logger.info(f"✅ Model loaded")
    
    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create example input
    example_input = torch.randn(1, 1, 1578, 751).to(device_obj)
    
    # Export to TorchScript
    if torchscript:
        logger.info("\nExporting to TorchScript...")
        scripted_model = torch.jit.trace(model_obj, example_input)
        scripted_path = output_dir / "model_scripted.pt"
        torch.jit.save(scripted_model, scripted_path)
        logger.info(f"  ✅ Saved: {scripted_path}")
    
    # Export to ONNX
    if onnx:
        logger.info("\nExporting to ONNX...")
        onnx_path = output_dir / "model.onnx"
        torch.onnx.export(
            model_obj,
            example_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=11
        )
        logger.info(f"  ✅ Saved: {onnx_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ EXPORT COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()