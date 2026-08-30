#!/usr/bin/env python3
"""
Preprocessing pipeline for seismic data.
Converts HDF5 to chunked PyTorch tensors.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
import click
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import SeismicConfig
from src.utils.logger import setup_logger, create_task_name
from src.utils.hdf5_utils import load_shot_indices, load_shot_data, validate_hdf5
from src.preprocessing.processor import ShotProcessor
from src.preprocessing.chunker import Chunker
from src.preprocessing.manifest import generate_manifest, save_manifest


@click.command()
@click.option('--config', '-c', required=True, help='Path to config YAML file')
@click.option('--force', '-f', is_flag=True, help='Force reprocessing even if chunks exist')
@click.option('--dataset', '-d', help='Override dataset name (for logging)')
def main(config: str, force: bool, dataset: str):
    """Run the preprocessing pipeline."""
    
    # Load config
    with open(config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    cfg = SeismicConfig(**config_dict)
    
    # Override dataset name if provided
    if dataset:
        cfg.dataset_name = dataset
    
    # Setup logger with dynamic task name
    task_name = create_task_name(cfg, "preprocess")
    logger = setup_logger(task_name=task_name)
    
    if force:
        cfg.force_reprocess = True
    
    logger.info("=" * 60)
    logger.info("SEISMIC FBP - PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Dataset: {cfg.dataset_name}")
    logger.info(f"HDF5 path: {cfg.hdf5_path}")
    logger.info(f"Chunk dir: {cfg.chunk_dir}")
    logger.info(f"Target traces: {cfg.target_traces}")
    logger.info(f"Samples: {cfg.n_samples}")
    logger.info(f"Strip width: {cfg.strip_width}")
    logger.info(f"Chunk size: {cfg.chunk_size}")
    logger.info(f"Random seed: {cfg.random_seed}")
    
    # Validate HDF5
    if not validate_hdf5(cfg.hdf5_path):
        logger.error("HDF5 validation failed. Exiting.")
        sys.exit(1)
    
    # Check if preprocessing already exists
    chunk_dir = Path(cfg.chunk_dir) / cfg.dataset_name
    manifest_path = chunk_dir / "manifest.json"
    
    if manifest_path.exists() and not cfg.force_reprocess:
        logger.info("✅ Preprocessing already exists. Skipping.")
        logger.info(f"   Manifest: {manifest_path}")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        logger.info(f"   Total shots: {manifest['total_shots']}")
        logger.info(f"   Total chunks: {len(manifest['chunks'])}")
        logger.info("✅ Preprocessing complete! (already existed)")
        return
    
    # Create chunk directory
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Data Discovery
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Data Discovery")
    logger.info("=" * 60)
    
    unique_shots, start_indices, end_indices = load_shot_indices(cfg.hdf5_path)
    total_shots = len(unique_shots)
    trace_counts = end_indices - start_indices
    
    logger.info(f"Total shots: {total_shots}")
    logger.info(f"Trace counts: min={trace_counts.min()}, max={trace_counts.max()}, mean={trace_counts.mean():.1f}")
    
    # Filter valid shots (at least 10 traces)
    valid_mask = trace_counts >= 10
    valid_shots = unique_shots[valid_mask]
    valid_indices = start_indices[valid_mask]
    valid_end_indices = end_indices[valid_mask]
    
    logger.info(f"Valid shots: {len(valid_shots)} (filtered {total_shots - len(valid_shots)} with <10 traces)")
    
    if len(valid_shots) == 0:
        logger.error("No valid shots found. Exiting.")
        sys.exit(1)
    
    # Phase 2: Chunk Assignment
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Chunk Assignment")
    logger.info("=" * 60)
    
    chunker = Chunker(
        chunk_size=cfg.chunk_size,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        random_seed=cfg.random_seed
    )
    
    splits = chunker.assign_splits(valid_shots)
    
    # Map shot IDs to their indices
    shot_to_idx = {shot: idx for idx, shot in enumerate(valid_shots)}
    shot_to_start = {shot: start for shot, start in zip(valid_shots, valid_indices)}
    shot_to_end = {shot: end for shot, end in zip(valid_shots, valid_end_indices)}
    
    chunks = {}
    for split_name, shot_list in splits.items():
        chunks[split_name] = chunker.create_chunks(shot_list)
        logger.info(f"  {split_name}: {len(shot_list)} shots, {len(chunks[split_name])} chunks")
    
    # Phase 3: Processing and Writing
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Processing and Writing Chunks")
    logger.info("=" * 60)
    
    processor = ShotProcessor(
        target_traces=cfg.target_traces,
        n_samples=cfg.n_samples,
        strip_width=cfg.strip_width
    )
    
    total_chunks = sum(len(c) for c in chunks.values())
    processed_chunks = 0
    
    for split_name, chunk_list in chunks.items():
        for chunk in tqdm(chunk_list, desc=f"Processing {split_name}"):
            chunk_id = chunk['id']
            shot_ids = chunk['shot_ids']
            n_shots = chunk['n_shots']
            
            # Pre-allocate tensors
            data_batch = np.zeros((n_shots, cfg.target_traces, cfg.n_samples), dtype=np.float32)
            mask_batch = np.zeros((n_shots, cfg.target_traces, cfg.n_samples), dtype=np.int64)
            
            for i, shot_id in enumerate(shot_ids):
                start_idx = shot_to_start[shot_id]
                end_idx = shot_to_end[shot_id]
                
                shot_data, shot_picks = load_shot_data(
                    cfg.hdf5_path, start_idx, end_idx, 
                    cfg.target_traces, cfg.n_samples
                )
                
                processed_data, processed_mask, stats = processor.process_shot(shot_data, shot_picks)
                data_batch[i] = processed_data
                mask_batch[i] = processed_mask
            
            # Save chunk
            chunk_filename = f"chunk_{chunk_id:03d}_{split_name}.pt"
            chunk_path = chunk_dir / chunk_filename
            
            torch.save({
                'data': torch.tensor(data_batch, dtype=torch.float32),
                'mask': torch.tensor(mask_batch, dtype=torch.long),
                'shot_ids': shot_ids,
                'split': split_name,
                'chunk_id': chunk_id,
                'n_shots': n_shots,
                'target_traces': cfg.target_traces,
                'n_samples': cfg.n_samples
            }, chunk_path)
            
            processed_chunks += 1
            chunk['filename'] = chunk_filename
            chunk['data_shape'] = list(data_batch.shape)
            chunk['mask_shape'] = list(mask_batch.shape)
            chunk['file_size_mb'] = chunk_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"\n✅ Processed {processed_chunks} chunks")
    
    # Phase 4: Generate Manifest
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: Generating Manifest")
    logger.info("=" * 60)
    
    manifest = generate_manifest(
        dataset_name=cfg.dataset_name,
        chunks=chunks,
        config=cfg.to_dict(),
        chunk_dir=chunk_dir,
        total_shots=len(valid_shots),
        total_traces=int(sum(trace_counts))
    )
    
    save_manifest(manifest, manifest_path)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ PREPROCESSING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Dataset: {cfg.dataset_name}")
    logger.info(f"Total shots: {len(valid_shots)}")
    logger.info(f"Total chunks: {processed_chunks}")
    logger.info(f"Chunk directory: {chunk_dir}")
    logger.info(f"Manifest: {manifest_path}")
    log_path = "logs/"
    try:
        log_path = str(logger._core.handlers[1]._path)
    except:
        pass
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()