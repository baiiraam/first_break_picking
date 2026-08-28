"""
Shot processing logic for seismic data.
"""

import numpy as np
from typing import Tuple
from loguru import logger


class ShotProcessor:
    """Process individual shots with vectorized operations."""
    
    def __init__(self, target_traces: int = 1578, n_samples: int = 751, strip_width: int = 8):
        self.target_traces = target_traces
        self.n_samples = n_samples
        self.strip_width = strip_width
        self.half_width = strip_width // 2
    
    def create_mask_vectorized(self, picks: np.ndarray) -> np.ndarray:
        """
        Create 3-class segmentation mask using vectorized operations.
        
        Class mapping:
            0: Before first break
            2: Strip around first break
            1: After first break
        """
        n_traces = len(picks)
        mask = np.zeros((n_traces, self.n_samples), dtype=np.int64)
        
        # Create sample index grid using broadcasting
        samples = np.arange(self.n_samples).reshape(1, -1)
        picks_expanded = picks.reshape(-1, 1)
        
        # Vectorized conditions
        strip_mask = (samples >= picks_expanded - self.half_width) & \
                     (samples <= picks_expanded + self.half_width)
        after_mask = samples > picks_expanded + self.half_width
        
        mask[strip_mask] = 2
        mask[after_mask] = 1
        
        # Invalid picks (0 or negative) become class 0
        invalid = (picks <= 0) | (picks >= self.n_samples)
        mask[invalid, :] = 0
        
        return mask
    
    def process_shot(self, shot_data: np.ndarray, shot_picks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single shot: pad/crop and create mask.
        
        Returns:
            processed_data: (target_traces, n_samples) float32
            processed_mask: (target_traces, n_samples) int64
        """
        actual_traces = shot_data.shape[0]
        
        # Pad or crop to target_traces
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
        
        # Create mask
        mask = self.create_mask_vectorized(shot_picks)
        
        return shot_data.astype(np.float32), mask