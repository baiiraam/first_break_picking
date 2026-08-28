#!/usr/bin/env python3
"""
Training script for seismic FBP with U-Net.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import click

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import SeismicConfig
from src.utils.logger import get_logger
from src.data.chunked_dataset import ChunkedDataManager
from src.preprocessing.manifest import load_manifest, validate_manifest

from src.models.unet import UNet
from src.models.efficient_unet import EfficientUNet
from src.models.mobilenet import MobileUNet

from src.training.trainer import SeismicTrainer

logger = get_logger()


@click.command()
@click.option('--config', '-c', required=True, help='Path to config YAML file')
@click.option('--resume', '-r', help='Path to checkpoint to resume from')
@click.option('--device', '-d', help='Override device (cpu/cuda/mps)')
@click.option('--epochs', '-e', type=int, help='Override number of epochs')
@click.option('--model', '-m', type=click.Choice(['unet', 'efficient', 'mobile']), 
              default='unet', help='Model architecture to use')
def main(config: str, resume: str, device: str, epochs: int, model: str):
    """Run the training pipeline."""
    
    # Load config
    with open(config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    cfg = SeismicConfig(**config_dict)
    
    # Override options
    if device:
        cfg.device = device
    if epochs:
        cfg.n_epochs = epochs
    
    logger.info("=" * 60)
    logger.info("SEISMIC FBP - TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Dataset: {cfg.dataset_name}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Epochs: {cfg.n_epochs}")
    logger.info(f"Learning rate: {cfg.learning_rate}")
    logger.info(f"LR scheduler: {cfg.lr_scheduler}")
    
    # Load manifest
    chunk_dir = Path(cfg.chunk_dir) / cfg.dataset_name
    manifest_path = chunk_dir / "manifest.json"
    
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        logger.error("Please run preprocessing first: python scripts/preprocess.py --config configs/halfmile.yaml")
        sys.exit(1)
    
    manifest = load_manifest(manifest_path)
    if not validate_manifest(manifest):
        logger.error("Invalid manifest")
        sys.exit(1)
    
    logger.info(f"\nManifest loaded: {manifest['dataset']}")
    logger.info(f"  Total shots: {manifest['total_shots']}")
    logger.info(f"  Total chunks: {len(manifest['chunks'])}")
    
    # Create data manager and datasets
    data_manager = ChunkedDataManager(
        chunk_dir=chunk_dir,
        manifest=manifest,
        cache_size=3,
        shuffle_chunks=True
    )
    
    train_dataset = data_manager.get_dataset('train')
    val_dataset = data_manager.get_dataset('val')
    test_dataset = data_manager.get_dataset('test')
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers // 2,
        pin_memory=cfg.device == "cuda",
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers // 2,
        pin_memory=cfg.device == "cuda",
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    logger.info(f"\nData loaded:")
    logger.info(f"  Training: {len(train_dataset)} shots, {len(train_loader)} batches")
    logger.info(f"  Validation: {len(val_dataset)} shots, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} shots, {len(test_loader)} batches")
    
    # --- MODEL INITIALIZATION ---
    logger.info(f"\nInitializing model: {model}")
    
    if model == "unet":
        model_obj = UNet(in_channels=1, out_channels=3)
        model_name = "UNet"
    elif model == "efficient":
        model_obj = EfficientUNet(in_channels=1, out_channels=3, pretrained=True)
        model_name = "EfficientUNet"
    elif model == "mobile":
        model_obj = MobileUNet(in_channels=1, out_channels=3, pretrained=True)
        model_name = "MobileUNet"
    else:
        raise ValueError(f"Unknown model: {model}")
    
    total_params = sum(p.numel() for p in model_obj.parameters())
    logger.info(f"\nModel: {model_name}")
    logger.info(f"  Parameters: {total_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model_obj.parameters(), lr=cfg.learning_rate)
    
    # Weighted loss for class imbalance
    device = torch.device(cfg.device)
    class_weights = torch.tensor([0.2, 0.2, 0.6], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = criterion.to(device)
    
    logger.info(f"\nClass weights: {class_weights.tolist()}")
    
    # Trainer
    trainer = SeismicTrainer(
        model=model_obj,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        config=cfg
    )
    
    # Train
    trainer.fit(resume_from=resume)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()