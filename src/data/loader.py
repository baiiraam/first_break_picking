# file location: src/data/loader.py

"""
Data loader setup for training pipeline.
"""

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.config import SeismicConfig
from src.data.chunked_dataset import ChunkedDataManager
from src.preprocessing.manifest import load_manifest, validate_manifest
from src.types import LoggerType  # ✅ Import from shared types


def create_dataloaders(
    cfg: SeismicConfig,
    logger: LoggerType,
    manifest_path: Path,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    """
    Create dataloaders for train, val, and test sets.

    Args:
        cfg: Configuration object
        logger: Logger instance
        manifest_path: Path to manifest file

    Returns:
        Tuple of (dataloaders_dict, dataset_info_dict)
    """
    # Load manifest
    manifest = load_manifest(manifest_path)
    if not validate_manifest(manifest):
        raise RuntimeError("Invalid manifest")

    logger.info(f"\nManifest loaded: {manifest['dataset']}")
    logger.info(f"  Total shots: {manifest['total_shots']}")
    logger.info(f"  Total chunks: {len(manifest['chunks'])}")

    # Create data manager and datasets
    chunk_dir = manifest_path.parent
    data_manager = ChunkedDataManager(
        chunk_dir=str(chunk_dir),
        manifest=manifest,
        cache_size=cfg.cache_size,
        shuffle_chunks=True,
    )

    train_dataset = data_manager.get_dataset("train")
    val_dataset = data_manager.get_dataset("val")
    test_dataset = data_manager.get_dataset("test")

    # Create dataloaders
    # Decide sampling strategy
    if cfg.chunk_aware_sampling:
        from src.data.samplers import ChunkAwareSampler

        sampler = ChunkAwareSampler(
            dataset=train_dataset,
            seed=cfg.random_seed,
            shuffle_chunks=True,
            shuffle_within=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,  # ← Must be False when using sampler
            sampler=sampler,  # ← NEW
            num_workers=cfg.num_workers,
            pin_memory=cfg.device == "cuda",
            prefetch_factor=2 if cfg.num_workers > 0 else None,
            persistent_workers=cfg.num_workers > 0,
            worker_init_fn=sampler.worker_init_fn if cfg.num_workers > 0 else None,
        )

        logger.info(
            f"🔀 Using ChunkAwareSampler for training "
            f"(chunks={sampler._num_chunks}, seed={cfg.random_seed})"
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,  # ← Fallback to full shuffle
            num_workers=cfg.num_workers,
            pin_memory=cfg.device == "cuda",
            prefetch_factor=2 if cfg.num_workers > 0 else None,
            persistent_workers=cfg.num_workers > 0,
        )

        logger.info("🔀 Using full shuffle for training (chunk_aware_sampling=False)")

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers // 2,
        pin_memory=cfg.device == "cuda",
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers // 2,
        pin_memory=cfg.device == "cuda",
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    dataset_info = {
        "train_shots": len(train_dataset),
        "train_batches": len(train_loader),
        "val_shots": len(val_dataset),
        "val_batches": len(val_loader),
        "test_shots": len(test_dataset),
        "test_batches": len(test_loader),
    }

    logger.info("\nData loaded:")
    logger.info(
        f"  Training: {dataset_info['train_shots']} shots, {dataset_info['train_batches']} batches"
    )
    logger.info(
        f"  Validation: {dataset_info['val_shots']} shots, {dataset_info['val_batches']} batches"
    )
    logger.info(
        f"  Test: {dataset_info['test_shots']} shots, {dataset_info['test_batches']} batches"
    )

    return dataloaders, dataset_info
