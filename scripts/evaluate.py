#!/usr/bin/env python3
"""
Evaluation script for trained seismic FBP model.
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import click
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import SeismicConfig
from src.utils.logger import setup_logger, create_task_name
from src.data.chunked_dataset import ChunkedDataManager
from src.preprocessing.manifest import load_manifest
from src.models.mps_light_unet import MPSLightUNet
from src.training.metrics import (
    SegmentationMetrics,
    FirstBreakMetrics,
    extract_picks_from_mask
)


@click.command()
@click.option('--config', '-c', required=True, help='Path to config YAML file')
@click.option('--model', '-m', required=True, help='Path to model checkpoint (.pt)')
@click.option('--output', '-o', default='evaluation_results', help='Output directory for results')
@click.option('--device', '-d', default='mps', help='Device to use (cpu/cuda/mps)')
@click.option('--batch_size', '-b', default=4, help='Batch size for evaluation')
@click.option('--dataset', '-ds', help='Override dataset name (for logging)')
def main(config: str, model: str, output: str, device: str, batch_size: int, dataset: str):
    """Evaluate the trained model on test set."""
    
    # Load config
    with open(config, 'r') as f:
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
    
    # Load manifest
    chunk_dir = Path(cfg.chunk_dir) / cfg.dataset_name
    manifest_path = chunk_dir / "manifest.json"
    
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)
    
    manifest = load_manifest(manifest_path)
    
    # Create data manager and test dataset
    data_manager = ChunkedDataManager(
        chunk_dir=chunk_dir,
        manifest=manifest,
        cache_size=2,
        shuffle_chunks=False
    )
    
    test_dataset = data_manager.get_dataset('test')
    
    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.device == "cuda"
    )
    
    logger.info(f"\nTest set: {len(test_dataset)} shots, {len(test_loader)} batches")
    
    # Load model
    device_obj = torch.device(cfg.device)
    model_obj = MPSLightUNet(in_channels=1, out_channels=3)
    
    checkpoint = torch.load(model, map_location=device_obj)
    if 'model_state_dict' in checkpoint:
        model_obj.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_obj.load_state_dict(checkpoint)
    
    model_obj = model_obj.to(device_obj)
    model_obj.eval()
    
    logger.info(f"\n✅ Model loaded")
    
    # Initialize metrics
    seg_metrics = SegmentationMetrics(num_classes=3)
    fb_metrics = FirstBreakMetrics(tolerance_samples=3)
    
    all_predictions = []
    all_labels = []
    
    # Evaluate
    logger.info("\nRunning evaluation...")
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(device_obj), y.to(device_obj)
            
            outputs = model_obj(x)
            preds = torch.argmax(outputs, dim=1)
            
            # Update segmentation metrics
            seg_metrics.update(preds, y)
            
            # Extract picks for first-break metrics
            pred_picks = extract_picks_from_mask(preds.cpu().numpy())
            true_picks = extract_picks_from_mask(y.cpu().numpy())
            fb_metrics.update(pred_picks, true_picks)
            
            # Store for later
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    
    # Compute all metrics
    seg_results = seg_metrics.compute()
    fb_results = fb_metrics.compute()
    
    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("📊 SEGMENTATION METRICS")
    logger.info("=" * 60)
    logger.info(f"  Accuracy: {seg_results['accuracy']:.4f}")
    logger.info(f"  Mean IoU: {seg_results['mean_iou']:.4f}")
    logger.info(f"  Mean F1: {seg_results['mean_f1']:.4f}")
    
    logger.info("\n  Class-wise IoU:")
    class_names = ["Class 0 (Before)", "Class 1 (After)", "Class 2 (Strip)"]
    for i, (name, iou) in enumerate(zip(class_names, seg_results['iou_per_class'])):
        logger.info(f"    {name}: {iou:.4f}")
    
    logger.info("\n  Class-wise F1:")
    for i, (name, f1) in enumerate(zip(class_names, seg_results['f1_per_class'])):
        logger.info(f"    {name}: {f1:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 FIRST-BREAK METRICS")
    logger.info("=" * 60)
    logger.info(f"  Mean Absolute Error (MAE): {fb_results['mean_absolute_error']:.2f} samples")
    logger.info(f"  Std Absolute Error: {fb_results['std_absolute_error']:.2f} samples")
    logger.info(f"  Median Absolute Error: {fb_results['median_absolute_error']:.2f} samples")
    logger.info(f"  Max Absolute Error: {fb_results['max_absolute_error']:.2f} samples")
    logger.info(f"  Min Absolute Error: {fb_results['min_absolute_error']:.2f} samples")
    logger.info(f"  Accuracy within ±3 samples: {fb_results['accuracy_within_tolerance']:.2%}")
    logger.info(f"  Total traces evaluated: {fb_results['total_traces']}")
    
    # Save results
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'segmentation_metrics': seg_results,
        'first_break_metrics': fb_results,
        'config': cfg.to_dict(),
        'model_path': model,
        'dataset': cfg.dataset_name,
        'test_samples': len(test_dataset)
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_dir}")
    logger.info("\n" + "=" * 60)
    logger.info("✅ EVALUATION COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()