# file location: src/preprocessing/pipeline.py

"""
Standalone preprocessing pipeline for seismic data.
"""

from pathlib import Path

import torch

from src.config import SeismicConfig
from src.preprocessing.chunker import Chunker
from src.preprocessing.manifest import generate_manifest, save_manifest
from src.preprocessing.processor import ShotProcessor
from src.types import LoggerType
from src.utils.hdf5_utils import load_shot_data, load_shot_indices, validate_hdf5


def run_preprocessing_pipeline(
    cfg: SeismicConfig,
    logger: LoggerType,
    force: bool = False,
) -> Path:
    """
    Run the complete preprocessing pipeline.

    Args:
        cfg: Configuration object
        logger: Logger instance
        force: Force reprocessing even if chunks exist

    Returns:
        Path to the manifest file
    """
    chunk_dir = Path(cfg.chunk_dir) / cfg.dataset_name
    manifest_path = chunk_dir / "manifest.json"

    # Check if preprocessing is needed
    if manifest_path.exists() and not force and not cfg.preprocess:
        logger.info(f"✅ Preprocessing already exists at {manifest_path}")
        return manifest_path

    logger.info(f"\n🔄 Preprocessing {cfg.dataset_name}...")

    # Validate HDF5
    if not validate_hdf5(cfg.hdf5_path):
        raise FileNotFoundError(f"HDF5 validation failed: {cfg.hdf5_path}")

    # Phase 1: Data Discovery
    unique_shots, start_indices, end_indices = load_shot_indices(cfg.hdf5_path)
    total_shots = len(unique_shots)
    trace_counts = end_indices - start_indices

    logger.info(f"Total shots: {total_shots}")
    logger.info(f"Trace counts: min={trace_counts.min()}, max={trace_counts.max()}")

    # Filter valid shots
    valid_mask = trace_counts >= 10
    valid_shots = unique_shots[valid_mask]
    valid_indices = start_indices[valid_mask]
    valid_end_indices = end_indices[valid_mask]

    if len(valid_shots) == 0:
        raise RuntimeError("No valid shots found.")

    # Phase 2: Chunk Assignment
    chunker = Chunker(cfg)
    splits = chunker.assign_splits(valid_shots)

    # Map shot IDs to indices
    shot_to_start = {shot: start for shot, start in zip(valid_shots, valid_indices)}
    shot_to_end = {shot: end for shot, end in zip(valid_shots, valid_end_indices)}

    chunks = {}
    for split_name, shot_list in splits.items():
        chunks[split_name] = chunker.create_chunks(shot_list)
        logger.info(
            f"  {split_name}: {len(shot_list)} shots, {len(chunks[split_name])} chunks"
        )

    # Phase 3: Processing
    processor = ShotProcessor(cfg)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for split_name, chunk_list in chunks.items():
        for chunk in chunk_list:
            chunk_id = chunk["id"]
            shot_ids = chunk["shot_ids"]
            n_shots = chunk["n_shots"]

            data_batch = torch.zeros(
                (n_shots, cfg.target_traces, cfg.n_samples), dtype=torch.float32
            )
            mask_batch = torch.zeros(
                (n_shots, cfg.target_traces, cfg.n_samples), dtype=torch.long
            )

            for i, shot_id in enumerate(shot_ids):
                shot_data, shot_picks = load_shot_data(
                    cfg.hdf5_path,
                    shot_to_start[shot_id],
                    shot_to_end[shot_id],
                    cfg.target_traces,
                    cfg.n_samples,
                )

                processed_data, processed_mask, _ = processor.process_shot(
                    shot_data, shot_picks
                )
                data_batch[i] = torch.tensor(processed_data, dtype=torch.float32)
                mask_batch[i] = torch.tensor(processed_mask, dtype=torch.long)

            chunk_filename = f"chunk_{chunk_id:03d}_{split_name}.pt"
            chunk_path = chunk_dir / chunk_filename

            torch.save(
                {
                    "data": data_batch,
                    "mask": mask_batch,
                    "shot_ids": shot_ids,
                    "split": split_name,
                    "chunk_id": chunk_id,
                    "n_shots": n_shots,
                },
                chunk_path,
            )

    # Phase 4: Generate Manifest
    manifest = generate_manifest(
        dataset_name=cfg.dataset_name,
        chunks=chunks,
        config=cfg.to_dict(),
        chunk_dir=chunk_dir,
        total_shots=len(valid_shots),
        total_traces=int(sum(trace_counts)),
    )
    save_manifest(manifest, manifest_path)

    logger.info(f"✅ Preprocessing complete for {cfg.dataset_name}")
    return manifest_path
