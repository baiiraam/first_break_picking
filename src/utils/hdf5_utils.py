"""
HDF5 utility functions for seismic data.
"""


import h5py
import numpy as np
from loguru import logger


def load_shot_indices(hdf5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load shot indices from HDF5 file.

    Returns:
        unique_shots: Array of unique shot IDs
        start_indices: Start index of each shot
        end_indices: End index of each shot
    """
    with h5py.File(hdf5_path, "r") as f:
        group = f["TRACE_DATA"]["DEFAULT"]
        shotids = group["SHOTID"][()].flatten()

        unique_shots, indices = np.unique(shotids, return_index=True)
        end_indices = np.append(indices[1:], len(shotids))

        logger.debug(f"Found {len(unique_shots)} shots in {hdf5_path}")
        return unique_shots, indices, end_indices


def load_shot_data(
    hdf5_path: str,
    start_idx: int,
    end_idx: int,
    target_traces: int = 1578,
    n_samples: int = 751,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single shot's data and picks from HDF5.

    Returns:
        shot_data: (target_traces, n_samples) float32
        shot_picks: (target_traces,) float32
    """
    with h5py.File(hdf5_path, "r") as f:
        group = f["TRACE_DATA"]["DEFAULT"]

        shot_data = group["data_array"][start_idx:end_idx, :]  # (n_traces, 751)
        shot_picks = group["SPARE1"][start_idx:end_idx, 0]  # (n_traces,)

        actual_traces = shot_data.shape[0]

        if actual_traces < target_traces:
            # Pad with zeros
            data_padded = np.zeros((target_traces, n_samples), dtype=np.float32)
            picks_padded = np.zeros(target_traces, dtype=np.float32)
            data_padded[:actual_traces, :] = shot_data
            picks_padded[:actual_traces] = shot_picks
            shot_data = data_padded
            shot_picks = picks_padded

        return shot_data, shot_picks


def get_trace_counts(
    hdf5_path: str,
    unique_shots: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
) -> np.ndarray:
    """Get trace counts for each shot."""
    return end_indices - start_indices


def validate_hdf5(hdf5_path: str) -> bool:
    """Validate HDF5 file structure."""
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "TRACE_DATA" not in f:
                logger.error(f"TRACE_DATA group not found in {hdf5_path}")
                return False
            group = f["TRACE_DATA"]["DEFAULT"]
            if "data_array" not in group:
                logger.error(f"data_array not found in {hdf5_path}")
                return False
            if "SHOTID" not in group:
                logger.error(f"SHOTID not found in {hdf5_path}")
                return False
            if "SPARE1" not in group:
                logger.error(f"SPARE1 not found in {hdf5_path}")
                return False
        return True
    except Exception as e: # noqa: BLE001
        logger.error(f"Error validating HDF5: {e}")
        return False
