"""
Chunk writing logic for preprocessing pipeline with optimized checksums.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

# ✅ Use consistent buffer size
CHECKSUM_BUFFER_SIZE = 65536  # 64KB


class ChunkWriter:
    """
    Writes processed chunks to disk with optimized checksums.
    """

    def __init__(self, chunk_dir: Path, compute_checksums: bool = True):
        self.chunk_dir = Path(chunk_dir)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.compute_checksums = compute_checksums

    def compute_checksum(
        self, filepath: Path, buffer_size: int = CHECKSUM_BUFFER_SIZE
    ) -> str:
        """Compute SHA-256 checksum with optimized buffer size."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(buffer_size), b""):
                sha256.update(block)
        return sha256.hexdigest()[:16]

    def write_chunk(
        self,
        data_batch: np.ndarray,
        mask_batch: np.ndarray,
        shot_ids: list[int],
        chunk_id: int,
        split: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Write a chunk to disk with checksum.
        """
        filename = f"chunk_{chunk_id:03d}_{split}.pt"
        filepath = self.chunk_dir / filename

        # Convert metadata to tensors
        chunk_data = {
            "data": torch.tensor(data_batch, dtype=torch.float32),
            "mask": torch.tensor(mask_batch, dtype=torch.long),
            "shot_ids": torch.tensor(shot_ids, dtype=torch.int64),
            "split_idx": torch.tensor(
                [0 if split == "train" else 1 if split == "val" else 2]
            ),
            "chunk_id": torch.tensor([chunk_id]),
            "n_shots": torch.tensor([len(shot_ids)]),
            "data_shape": torch.tensor(list(data_batch.shape), dtype=torch.int64),
            "mask_shape": torch.tensor(list(mask_batch.shape), dtype=torch.int64),
        }

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    chunk_data[f"meta_{key}"] = torch.tensor([value])
                elif isinstance(value, list):
                    chunk_data[f"meta_{key}"] = torch.tensor(value)
                else:
                    chunk_data[f"meta_{key}"] = value

        # Save with high protocol for efficiency
        torch.save(chunk_data, filepath)

        # Compute checksum if enabled
        if self.compute_checksums:
            checksum = self.compute_checksum(filepath)
            checksum_path = filepath.with_suffix(".checksum")
            with open(checksum_path, "w") as f:
                f.write(checksum)
            logger.debug(f"Chunk {chunk_id} checksum: {checksum}")

        logger.debug(
            f"Written chunk: {filename} ({filepath.stat().st_size / (1024 * 1024):.1f} MB)"
        )

        return filepath

    def verify_chunk(self, filepath: Path) -> bool:
        """
        Verify a chunk file is valid and loadable.
        """
        try:
            # Try weights_only=True first (safer)
            try:
                chunk = torch.load(filepath, map_location="cpu", weights_only=True)
            except Exception as e:
                # Fallback for backward compatibility with older checkpoints
                logger.debug(
                    f"weights_only=True failed, trying with weights_only=False: {e}"
                )
                chunk = torch.load(filepath, map_location="cpu", weights_only=False)

            required_keys = [
                "data",
                "mask",
                "shot_ids",
                "split_idx",
                "chunk_id",
                "n_shots",
            ]
            for key in required_keys:
                if key not in chunk:
                    logger.error(f"Missing key in {filepath}: {key}")
                    return False

            if chunk["data"].shape[0] != chunk["n_shots"].item():
                logger.error(
                    f"Data shape mismatch in {filepath}: data has {chunk['data'].shape[0]} shots, expected {chunk['n_shots'].item()}"
                )
                return False

            # Verify checksum if available
            if self.compute_checksums:
                checksum_path = filepath.with_suffix(".checksum")
                if checksum_path.exists():
                    with open(checksum_path, "r") as f:
                        stored_checksum = f.read().strip()
                    computed_checksum = self.compute_checksum(filepath)
                    if stored_checksum != computed_checksum:
                        logger.error(
                            f"Checksum mismatch in {filepath}: stored={stored_checksum}, computed={computed_checksum}"
                        )
                        return False

            return True
        except Exception as e:
            logger.error(f"Error verifying {filepath}: {e}")
            return False

    def write_all_chunks(
        self,
        chunks: dict[str, list[dict[str, Any]]],
        data_batches: dict[int, np.ndarray],
        mask_batches: dict[int, np.ndarray],
        shot_ids: dict[int, list[int]],
        metadata: dict[str, Any] | None = None,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> list[Path]:
        """
        Write all chunks from a preprocessing run.

        Returns:
            List of paths to saved chunk files
        """
        saved_paths = []
        total_chunks = sum(len(c) for c in chunks.values())

        if parallel:
            # Parallel writing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for split_name, chunk_list in chunks.items():
                    for chunk in chunk_list:
                        chunk_id = chunk["id"]
                        future = executor.submit(
                            self.write_chunk,
                            data_batches[chunk_id],
                            mask_batches[chunk_id],
                            shot_ids[chunk_id],
                            chunk_id,
                            split_name,
                            metadata,
                        )
                        futures.append((chunk_id, future))

                with tqdm(total=len(futures), desc="Writing chunks (parallel)") as pbar:
                    for chunk_id, future in futures:
                        saved_paths.append(future.result())
                        pbar.update(1)
                        pbar.set_postfix({"current": f"chunk_{chunk_id:03d}"})
        else:
            # Sequential writing
            pbar = tqdm(total=total_chunks, desc="Writing chunks")

            for split_name, chunk_list in chunks.items():
                for chunk in chunk_list:
                    chunk_id = chunk["id"]
                    filepath = self.write_chunk(
                        data_batch=data_batches[chunk_id],
                        mask_batch=mask_batches[chunk_id],
                        shot_ids=shot_ids[chunk_id],
                        chunk_id=chunk_id,
                        split=split_name,
                        metadata=metadata,
                    )
                    saved_paths.append(filepath)
                    pbar.update(1)
                    pbar.set_postfix({"current": filepath.name})

            pbar.close()

        logger.info(f"Written {len(saved_paths)} chunks to {self.chunk_dir}")

        # Verify all chunks
        failed = []
        for filepath in tqdm(saved_paths, desc="Verifying chunks"):
            if not self.verify_chunk(filepath):
                failed.append(filepath)

        if failed:
            logger.error(f"{len(failed)} chunks failed verification")
        else:
            logger.info(f"All {len(saved_paths)} chunks verified successfully")

        return saved_paths
