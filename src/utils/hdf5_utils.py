"""
HDF5 utility functions with persistent context manager for efficient batch operations.
Thread-safe with locking for multi-worker data loading.
"""

import threading

import h5py
import numpy as np
from loguru import logger


class HDF5SeismicReader:
    """
    Persistent HDF5 reader with thread-safe operations for multi-worker data loading.

    Usage:
        with HDF5SeismicReader(hdf5_path) as reader:
            data, picks = reader.load_shot_data(start_idx, end_idx)
            unique_shots = reader.load_shot_indices()
    """

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self._file = None
        self._group = None
        self._is_open = False
        # ✅ Thread safety lock for HDF5 operations
        self._lock = threading.Lock()
        # Track worker info for debugging
        self._worker_id = None

    def __enter__(self):
        """Open HDF5 file and return self."""
        self._file = h5py.File(self.hdf5_path, "r", swmr=True)
        self._group = self._file["TRACE_DATA"]["DEFAULT"]
        self._is_open = True
        logger.debug(f"[HDF5 Reader] Opened: {self.hdf5_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file and cleanup."""
        self.close()
        if exc_type is not None:
            logger.error(f"[HDF5 Reader] Exception during read: {exc_val}")
        return False  # Don't suppress exceptions

    def close(self):
        """Explicitly close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._group = None
            self._is_open = False
            logger.debug(f"[HDF5 Reader] Closed: {self.hdf5_path}")

    def _ensure_open(self):
        """Ensure the HDF5 file is open."""
        if not self._is_open or self._file is None:
            raise RuntimeError(
                f"HDF5SeismicReader must be used inside a 'with' block. "
                f"Called on {self.hdf5_path}"
            )

    def set_worker_id(self, worker_id: int):
        """Set worker ID for debugging in multi-worker environments."""
        self._worker_id = worker_id
        logger.debug(f"[HDF5 Reader] Worker {worker_id} initialized")

    def load_shot_data(
        self,
        start_idx: int,
        end_idx: int,
        target_traces: int = 1578,
        n_samples: int = 751,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load a single shot's data and picks from HDF5.
        Thread-safe with lock protection.

        Returns:
            shot_data: (target_traces, n_samples) float32
            shot_picks: (target_traces,) float32
        """
        self._ensure_open()

        # ✅ Thread-safe HDF5 read with lock
        with self._lock:
            shot_data = self._group["data_array"][start_idx:end_idx, :]
            shot_picks = self._group["SPARE1"][start_idx:end_idx, 0]

        actual_traces = shot_data.shape[0]

        if actual_traces < target_traces:
            data_padded = np.zeros((target_traces, n_samples), dtype=np.float32)
            picks_padded = np.zeros(target_traces, dtype=np.float32)
            data_padded[:actual_traces, :] = shot_data
            picks_padded[:actual_traces] = shot_picks
            return data_padded, picks_padded
        elif actual_traces > target_traces:
            return shot_data[:target_traces, :].copy(), shot_picks[
                :target_traces
            ].copy()
        else:
            return shot_data.copy(), shot_picks.copy()

    def load_shot_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load shot indices from HDF5 file.
        Thread-safe with lock protection.

        Returns:
            unique_shots: Array of unique shot IDs
            start_indices: Start index of each shot
            end_indices: End index of each shot
        """
        self._ensure_open()

        # ✅ Thread-safe HDF5 read with lock
        with self._lock:
            shotids = self._group["SHOTID"][()].flatten()

        unique_shots, indices = np.unique(shotids, return_index=True)
        end_indices = np.append(indices[1:], len(shotids))

        logger.debug(f"[HDF5 Reader] Found {len(unique_shots)} shots")
        return unique_shots, indices, end_indices

    def get_trace_counts(self) -> np.ndarray:
        """Get trace counts for each shot."""
        unique_shots, start_indices, end_indices = self.load_shot_indices()
        return end_indices - start_indices

    def validate_hdf5(self) -> bool:
        """Validate HDF5 file structure."""
        self._ensure_open()
        try:
            if "TRACE_DATA" not in self._file:
                logger.error(f"TRACE_DATA group not found in {self.hdf5_path}")
                return False
            group = self._file["TRACE_DATA"]["DEFAULT"]
            required_keys = ["data_array", "SHOTID", "SPARE1"]
            for key in required_keys:
                if key not in group:
                    logger.error(f"{key} not found in {self.hdf5_path}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error validating HDF5: {e}")
            return False

    def get_data_shape(self) -> tuple[int, int]:
        """Get the shape of the data array."""
        self._ensure_open()
        with self._lock:
            return self._group["data_array"].shape

    def get_sample_rate(self) -> float:
        """Get the sample rate in microseconds."""
        self._ensure_open()
        with self._lock:
            sample_rate = self._group.get("SAMP_RATE", None)
        if sample_rate is not None:
            sr = sample_rate[()]
            if hasattr(sr, "shape") and sr.size > 0:
                sr = sr[0, 0] if sr.ndim == 2 else sr[0] if sr.ndim == 1 else sr
            return float(sr)
        return None


# ============================================================
# LEGACY FUNCTIONS (Maintained for backward compatibility)
# ============================================================


def load_shot_indices(hdf5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load shot indices from HDF5 file (legacy function)."""
    with HDF5SeismicReader(hdf5_path) as reader:
        return reader.load_shot_indices()


def load_shot_data(
    hdf5_path: str,
    start_idx: int,
    end_idx: int,
    target_traces: int = 1578,
    n_samples: int = 751,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a single shot's data and picks from HDF5 (legacy function)."""
    with HDF5SeismicReader(hdf5_path) as reader:
        return reader.load_shot_data(start_idx, end_idx, target_traces, n_samples)

def validate_hdf5(hdf5_path: str) -> bool:
    """Validate HDF5 file structure (convenience function)."""
    with HDF5SeismicReader(hdf5_path) as reader:
        return reader.validate_hdf5()

