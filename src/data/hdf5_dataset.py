"""
HDF5 dataset for lazy loading with level-based telemetry.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from loguru import logger


class HDF5SeismicDataset(Dataset):
    """
    Memory-efficient dataset that loads from HDF5 on-the-fly.
    
    Logging:
        - INFO: Dataset init
        - DEBUG: Every __getitem__ call with shot_id and slice info
    """
    
    def __init__(
        self,
        hdf5_path: str,
        shot_indices: Dict[int, Tuple[int, int]],
        shot_ids: list,
        target_traces: int = 1578,
        n_samples: int = 751,
        strip_width: int = 8
    ):
        self.hdf5_path = hdf5_path
        self.shot_indices = shot_indices
        self.shot_ids = shot_ids
        self.target_traces = target_traces
        self.n_samples = n_samples
        self.strip_width = strip_width
        self.half_width = strip_width // 2
        
        self.file = None
        self.group = None
        
        logger.info(f"[HDF5] INIT: {len(self)} shots, file={hdf5_path}")
        logger.debug(f"[HDF5] target_traces={target_traces}, n_samples={n_samples}, strip_width={strip_width}")
    
    def __len__(self) -> int:
        return len(self.shot_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single shot from HDF5."""
        shot_id = self.shot_ids[idx]
        start_idx, end_idx = self.shot_indices[shot_id]
        
        logger.debug(f"[HDF5] GET idx={idx} → shot={shot_id}, slice={start_idx}:{end_idx}")
        
        if self.file is None:
            logger.debug(f"[HDF5] Opening HDF5 file: {self.hdf5_path}")
            self.file = h5py.File(self.hdf5_path, 'r', swmr=True)
            self.group = self.file['TRACE_DATA']['DEFAULT']
        
        # Read data
        shot_data = self.group['data_array'][start_idx:end_idx, :]
        shot_picks = self.group['SPARE1'][start_idx:end_idx, 0]
        
        # Pad/crop
        actual_traces = shot_data.shape[0]
        if actual_traces < self.target_traces:
            data_padded = np.zeros((self.target_traces, self.n_samples), dtype=np.float32)
            picks_padded = np.zeros(self.target_traces, dtype=np.float32)
            data_padded[:actual_traces, :] = shot_data
            picks_padded[:actual_traces] = shot_picks
            shot_data = data_padded
            shot_picks = picks_padded
        elif actual_traces > self.target_traces:
            shot_data = shot_data[:self.target_traces, :]
            shot_picks = shot_picks[:self.target_traces]
        
        mask = self._create_mask(shot_picks)
        
        return (
            torch.from_numpy(shot_data).float().unsqueeze(0),
            torch.from_numpy(mask).long()
        )
    
    def _create_mask(self, picks: np.ndarray) -> np.ndarray:
        n_traces = len(picks)
        mask = np.zeros((n_traces, self.n_samples), dtype=np.int64)
        
        for i, pick in enumerate(picks):
            if pick <= 0 or pick >= self.n_samples:
                continue
            pick_int = int(round(pick))
            start = max(0, pick_int - self.half_width)
            end = min(self.n_samples, pick_int + self.half_width + 1)
            mask[i, start:end] = 2
            mask[i, end:] = 1
        
        return mask
    
    def close(self):
        if self.file is not None:
            logger.debug(f"[HDF5] Closing file: {self.hdf5_path}")
            self.file.close()
            self.file = None
            self.group = None
    
    def __del__(self):
        self.close()
    
    def get_shot_id(self, idx: int) -> int:
        return self.shot_ids[idx]