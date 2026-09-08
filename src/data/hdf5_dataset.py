"""
HDF5 dataset for lazy loading with multiprocessing worker safety, buffer validation, and explicit resource management.
"""

from typing import Any

import h5py
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset, get_worker_info


class HDF5SeismicDataset(Dataset):
    """
    Memory-efficient dataset that loads from HDF5 on-the-fly, safe for multiprocessing workers.
    Uses default HDF5 driver to avoid memory duplication across workers.
    """

    def __init__(
        self,
        hdf5_path: str,
        shot_indices: dict[int, tuple[int, int]],
        shot_ids: list[int],
        target_traces: int = 1578,
        n_samples: int = 751,
        strip_width: int = 8,
    ):
        self.hdf5_path = hdf5_path
        self.shot_indices = shot_indices
        self.shot_ids = shot_ids
        self.target_traces = target_traces
        self.n_samples = n_samples
        self.strip_width = strip_width
        self.half_width = strip_width // 2

        self.file: h5py.File | None = None
        self.group: h5py.Dataset | h5py.Group | Any = None

        logger.info(f"[HDF5] INIT: {len(self)} shots, file={hdf5_path}")
        logger.debug(
            f"[HDF5] target_traces={target_traces}, n_samples={n_samples}, strip_width={strip_width}"
        )

    def __len__(self) -> int:
        return len(self.shot_ids)

    def _register_cleanup(self):
        """Register cleanup function for worker termination."""
        if not self._cleanup_registered:
            atexit.register(self._close)
            self._cleanup_registered = True
            logger.debug(f"[HDF5] Registered cleanup for worker {self._worker_id}")

    def _ensure_file_open(self):
        """
        Ensure HDF5 file handle is open safely for the current process/worker.
        ✅ Uses default driver to avoid memory duplication across multiprocessing workers.
        ✅ Removed swmr=True for static datasets (reduces I/O overhead).
        """
        if self._file is None:
            worker_info = get_worker_info()
            self._worker_id = worker_info.id if worker_info is not None else "main"
            logger.debug(
                f"[HDF5] Opening HDF5 file for worker/process {self._worker_id}: {self.hdf5_path}"
            )

            # ✅ Use default driver (streaming) to avoid memory duplication across workers
            # The 'core' driver would load the entire file into memory for each worker
            # which causes memory blowup with num_workers > 0
            self._file = h5py.File(self.hdf5_path, "r")
            self._group = self._file["TRACE_DATA"]["DEFAULT"]

            # Register cleanup
            self._register_cleanup()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a single shot from HDF5 using worker-safe handles and buffer reuse."""
        self._ensure_file_open()

        shot_id = self.shot_ids[idx]
        start_idx, end_idx = self.shot_indices[shot_id]

        logger.debug(
            f"[HDF5] GET idx={idx} → shot={shot_id}, slice={start_idx}:{end_idx}"
        )

        if self.file is None:
            logger.debug(f"[HDF5] Opening HDF5 file: {self.hdf5_path}")
            self.file = h5py.File(self.hdf5_path, "r", swmr=True)
            self.group = self.file["TRACE_DATA"]["DEFAULT"]
        assert self.group is not None, "HDF5 group is not initialized"
        # Read data
        shot_data = self._group["data_array"][start_idx:end_idx, :]
        shot_picks = self._group["SPARE1"][start_idx:end_idx, 0]

        actual_traces = shot_data.shape[0]

        if self._needs_padding[shot_id]:
            self._data_buffer.fill(0)
            self._picks_buffer.fill(0)
            self._data_buffer[:actual_traces, :] = shot_data
            self._picks_buffer[:actual_traces] = shot_picks
            shot_data = self._data_buffer[: self.target_traces, :].copy()
            shot_picks = self._picks_buffer[: self.target_traces].copy()
        elif self._needs_cropping[shot_id]:
            shot_data = shot_data[: self.target_traces, :].copy()
            shot_picks = shot_picks[: self.target_traces].copy()
        else:
            shot_data = shot_data.copy()
            shot_picks = shot_picks.copy()

        mask = self._create_mask(shot_picks)

        return (
            torch.from_numpy(shot_data).float().unsqueeze(0).contiguous(),
            torch.from_numpy(mask).long().contiguous(),
        )

    def _create_mask(self, picks: np.ndarray) -> np.ndarray:
        n_traces = len(picks)
        mask = np.zeros((n_traces, self.n_samples), dtype=np.int64)

        for i, pick in enumerate(picks):
            if pick <= 0 or pick >= self.n_samples:
                continue
            pick_int = round(pick)
            start = max(0, pick_int - self.half_width)
            end = min(self.n_samples, pick_int + self.half_width + 1)
            mask[i, start:end] = 2
            mask[i, end:] = 1

        return mask

    def _close(self):
        """Internal cleanup method."""
        if self._file is not None:
            logger.debug(
                f"[HDF5] Closing file for worker {self._worker_id}: {self.hdf5_path}"
            )
            self._file.close()
            self._file = None
            self._group = None

    def close(self):
        """Explicit cleanup method - call when done."""
        self._close()

    def get_shot_id(self, idx: int) -> int:
        return int(self.shot_ids[idx])
