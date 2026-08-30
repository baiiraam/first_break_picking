"""
Shot processing logic for seismic data with validation and logging.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from loguru import logger


class ShotProcessor:
    """Process individual shots with vectorized operations and validation."""
    
    def __init__(
        self,
        target_traces: int = 1578,
        n_samples: int = 751,
        strip_width: int = 8,
        log_level: str = "INFO"
    ):
        self.target_traces = target_traces
        self.n_samples = n_samples
        self.strip_width = strip_width
        self.half_width = strip_width // 2
        self.log_level = log_level
        self.stats = []
    
    def validate_picks(self, picks: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Validate and clean picks.
        
        Returns:
            cleaned_picks: Picks with invalid values clipped
            stats: Dictionary of validation statistics
        """
        total = len(picks)
        valid_mask = (picks > 0) & (picks < self.n_samples)
        valid_count = np.sum(valid_mask)
        invalid_count = total - valid_count
        
        stats = {
            'total': total,
            'valid': valid_count,
            'invalid': invalid_count,
            'invalid_ratio': invalid_count / total if total > 0 else 0,
        }
        
        if valid_count > 0:
            valid_picks = picks[valid_mask]
            stats['min_pick'] = float(valid_picks.min())
            stats['max_pick'] = float(valid_picks.max())
            stats['mean_pick'] = float(valid_picks.mean())
            stats['median_pick'] = float(np.median(valid_picks))
        else:
            stats['min_pick'] = None
            stats['max_pick'] = None
            stats['mean_pick'] = None
            stats['median_pick'] = None
        
        # Warn if many invalid picks
        if invalid_count > total * 0.1:
            logger.warning(f"High invalid picks: {invalid_count}/{total} ({invalid_count/total:.1%})")
        
        # Clip out-of-range picks to valid range
        cleaned_picks = np.clip(picks, 0, self.n_samples - 1)
        
        return cleaned_picks, stats
    
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
    
    def validate_mask(self, mask: np.ndarray, picks: np.ndarray, shot_id: Optional[int] = None) -> bool:
        """Validate mask quality."""
        # Check that strip exists
        strip_count = np.sum(mask == 2)
        if strip_count == 0:
            logger.warning(f"Shot {shot_id}: No strip (class 2) found in mask!")
            return False
        
        # Check strip is near the pick (within ±10 samples)
        issues = 0
        for i, pick in enumerate(picks):
            if pick > 0 and pick < self.n_samples:
                strip_indices = np.where(mask[i] == 2)[0]
                if len(strip_indices) > 0:
                    strip_center = np.median(strip_indices)
                    if abs(strip_center - pick) > 10:
                        issues += 1
                        if issues <= 3:  # Log only first 3 issues
                            logger.debug(f"Shot {shot_id}, trace {i}: strip center {strip_center:.0f} far from pick {pick:.0f}")
        
        if issues > 0:
            logger.debug(f"Shot {shot_id}: {issues} traces with misaligned strips")
        
        return True
    
    def get_shot_statistics(self, shot_picks: np.ndarray) -> Dict:
        """Compute statistics for a shot."""
        valid_picks = shot_picks[shot_picks > 0]
        
        if len(valid_picks) > 0:
            return {
                'n_traces': len(shot_picks),
                'n_valid': len(valid_picks),
                'n_invalid': len(shot_picks) - len(valid_picks),
                'invalid_ratio': (len(shot_picks) - len(valid_picks)) / len(shot_picks),
                'min_pick': float(valid_picks.min()),
                'max_pick': float(valid_picks.max()),
                'mean_pick': float(valid_picks.mean()),
                'median_pick': float(np.median(valid_picks)),
            }
        else:
            return {
                'n_traces': len(shot_picks),
                'n_valid': 0,
                'n_invalid': len(shot_picks),
                'invalid_ratio': 1.0,
                'min_pick': None,
                'max_pick': None,
                'mean_pick': None,
                'median_pick': None,
            }
    
    def process_shot(
        self,
        shot_data: np.ndarray,
        shot_picks: np.ndarray,
        shot_id: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Process a single shot: validate, pad/crop, create mask.
        
        Returns:
            processed_data: (target_traces, n_samples) float32
            processed_mask: (target_traces, n_samples) int64
            stats: Dictionary of processing statistics
        """
        actual_traces = shot_data.shape[0]
        
        if self.log_level == "DEBUG" and shot_id is not None:
            logger.debug(f"Processing shot {shot_id}: {actual_traces} traces")
        
        # Validate and clean picks
        cleaned_picks, pick_stats = self.validate_picks(shot_picks)
        
        # Pad or crop to target_traces
        if actual_traces < self.target_traces:
            data_padded = np.zeros((self.target_traces, self.n_samples), dtype=np.float32)
            picks_padded = np.zeros(self.target_traces, dtype=np.float32)
            data_padded[:actual_traces, :] = shot_data
            picks_padded[:actual_traces] = cleaned_picks
            shot_data = data_padded
            shot_picks = picks_padded
            if self.log_level == "INFO":
                logger.debug(f"Shot {shot_id}: padded {actual_traces} → {self.target_traces} traces")
        elif actual_traces > self.target_traces:
            shot_data = shot_data[:self.target_traces, :]
            shot_picks = cleaned_picks[:self.target_traces]
            if self.log_level == "INFO":
                logger.debug(f"Shot {shot_id}: cropped {actual_traces} → {self.target_traces} traces")
        else:
            shot_picks = cleaned_picks
        
        # Create mask
        mask = self.create_mask_vectorized(shot_picks)
        
        # Validate mask quality (only if pick valid)
        if np.any(shot_picks > 0):
            self.validate_mask(mask, shot_picks, shot_id)
        
        # Collect statistics
        stats = self.get_shot_statistics(shot_picks)
        stats.update({
            'original_traces': actual_traces,
            'padded_or_cropped': actual_traces != self.target_traces,
        })
        self.stats.append(stats)
        
        if self.log_level == "DEBUG" and shot_id is not None:
            unique, counts = np.unique(mask, return_counts=True)
            class_dist = dict(zip(unique.tolist(), counts.tolist()))
            logger.debug(f"Shot {shot_id}: mask classes {class_dist}")
        
        return shot_data.astype(np.float32), mask, stats
    
    def get_all_stats(self) -> Dict:
        """Get aggregate statistics for all processed shots."""
        if not self.stats:
            return {}
        
        n_valid = [s['n_valid'] for s in self.stats]
        n_invalid = [s['n_invalid'] for s in self.stats]
        n_traces = [s['n_traces'] for s in self.stats]
        
        valid_picks = []
        for s in self.stats:
            if s['min_pick'] is not None:
                valid_picks.append(s['min_pick'])
                valid_picks.append(s['max_pick'])
        
        return {
            'total_shots': len(self.stats),
            'total_traces': sum(n_traces),
            'total_valid': sum(n_valid),
            'total_invalid': sum(n_invalid),
            'avg_valid_per_shot': np.mean(n_valid),
            'avg_invalid_per_shot': np.mean(n_invalid),
            'shots_with_no_valid_picks': sum(1 for s in self.stats if s['n_valid'] == 0),
            'min_pick_overall': min(valid_picks) if valid_picks else None,
            'max_pick_overall': max(valid_picks) if valid_picks else None,
        }
    
    def reset_stats(self):
        """Reset accumulated statistics."""
        self.stats = []