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
from src.utils.logger import setup_logger, create_task_name
from src.models.unet import UNet
from src.models.mps_light_unet import MPSLightUNet


@click.command()
@click.option('--model', '-m', required=True, help='Path to model checkpoint (.pt)')
@click.option('--output', '-o', default='exported_models', help='Output directory')
@click.option('--onnx', is_flag=True, help='Export to ONNX format')
@click.option('--torchscript', is_flag=True, help='Export to TorchScript format')
@click.option('--device', '-d', default='cpu', help='Device for export (cpu/cuda/mps)')
@click.option('--model-type', '-t', type=click.Choice(['unet', 'mpslight']), 
              default='unet', help='Model architecture type')
@click.option('--config', '-c', help='Path to config YAML file (for dataset name in logging)')
def main(model: str, output: str, onnx: bool, torchscript: bool, 
         device: str, model_type: str, config: str):
    """Export trained model to production formats."""
    
    if not onnx and not torchscript:
        print("ERROR: Please specify at least one export format: --onnx or --torchscript")
        sys.exit(1)
    
    # Setup logger
    if config:
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)
        cfg = SeismicConfig(**config_dict)
        task_name = create_task_name(cfg, "export")
    else:
        task_name = "export_model"
    
    logger = setup_logger(task_name=task_name)
    
    logger.info("=" * 60)
    logger.info("SEISMIC FBP - MODEL EXPORT")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Output: {output}")
    logger.info(f"Device: {device}")
    
    # Load model based on type
    device_obj = torch.device(device)
    
    logger.info(f"\nInitializing {model_type} model...")
    
    if model_type == "unet":
        model_obj = UNet(in_channels=1, out_channels=3)
    elif model_type == "mpslight":
        model_obj = MPSLightUNet(in_channels=1, out_channels=3)
    else:
        logger.error(f"Unknown model type: {model_type}")
        sys.exit(1)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model, map_location=device_obj)
        if 'model_state_dict' in checkpoint:
            model_obj.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
        else:
            model_obj.load_state_dict(checkpoint)
            logger.info("Loaded model state dict")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    model_obj = model_obj.to(device_obj)
    model_obj.eval()
    
    logger.info(f"✅ Model loaded successfully")
    
    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create example input
    example_input = torch.randn(1, 1, 1578, 751).to(device_obj)
    
    # Export to TorchScript
    if torchscript:
        logger.info("\n📦 Exporting to TorchScript...")
        try:
            scripted_model = torch.jit.trace(model_obj, example_input)
            scripted_path = output_dir / f"{model_type}_model_scripted.pt"
            torch.jit.save(scripted_model, scripted_path)
            logger.info(f"  ✅ Saved: {scripted_path}")
            logger.info(f"  File size: {scripted_path.stat().st_size / (1024*1024):.2f} MB")
        except Exception as e:
            logger.error(f"  ❌ TorchScript export failed: {e}")
    
    # Export to ONNX
    if onnx:
        logger.info("\n📦 Exporting to ONNX...")
        try:
            onnx_path = output_dir / f"{model_type}_model.onnx"
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
                opset_version=11,
                do_constant_folding=True,
                verbose=False
            )
            logger.info(f"  ✅ Saved: {onnx_path}")
            logger.info(f"  File size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Optional: Verify ONNX model
            try:
                import onnx
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                logger.info("  ✅ ONNX model verified")
            except ImportError:
                logger.info("  ⚠️  ONNX library not installed, skipping verification")
            except Exception as e:
                logger.warning(f"  ⚠️  ONNX verification failed: {e}")
                
        except Exception as e:
            logger.error(f"  ❌ ONNX export failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ EXPORT COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info("\nTo use the exported model:")
    if torchscript:
        logger.info(f"  TorchScript: torch.jit.load('{output_dir}/{model_type}_model_scripted.pt')")
    if onnx:
        logger.info(f"  ONNX: import onnx; model = onnx.load('{output_dir}/{model_type}_model.onnx')")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()