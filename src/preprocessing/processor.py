# file location: src/preprocessing/processor.py

"""
Shot processing logic for seismic data with validation and logging.
"""

from typing import Any

import numpy as np
from loguru import logger

from src.config import SeismicConfig


class ShotProcessor:
    """Process individual shots with vectorized operations and validation."""

    def __init__(self, config: SeismicConfig, log_level: str = "INFO"):
        self.target_traces = config.target_traces
        self.n_samples = config.n_samples
        self.strip_width = config.strip_width
        self.half_width = config.strip_width // 2
        self.log_level = log_level
        self.ignore_index = config.ignore_index
        self.sampling_interval_ms = config.sampling_interval_ms
        self.stats: list[dict[str, Any]] = []

    def validate_picks(self, picks: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Validate and clean picks.
        CONVERTS MILLISECONDS TO SAMPLES!
        """
        total = len(picks)

        # 🆕 CONVERT FROM MILLISECONDS TO SAMPLES
        picks_samples = picks / self.sampling_interval_ms

        # Round to nearest sample
        picks_samples = np.round(picks_samples).astype(np.float32)

        # Clip to valid range [0, n_samples-1]
        picks_samples = np.clip(picks_samples, 0, self.n_samples - 1)

        # Count valid picks (between 0 and n_samples-1)
        valid_mask = (picks_samples > 0) & (picks_samples < self.n_samples)
        valid_count = np.sum(valid_mask)
        invalid_count = total - valid_count

        stats = {
            "total": total,
            "valid": valid_count,
            "invalid": invalid_count,
            "invalid_ratio": invalid_count / total if total > 0 else 0,
        }

        if valid_count > 0:
            valid_picks = picks_samples[valid_mask]
            stats["min_pick"] = float(valid_picks.min())
            stats["max_pick"] = float(valid_picks.max())
            stats["mean_pick"] = float(valid_picks.mean())
            stats["median_pick"] = float(np.median(valid_picks))
        else:
            stats["min_pick"] = None
            stats["max_pick"] = None
            stats["mean_pick"] = None
            stats["median_pick"] = None

        # Warn if many invalid picks
        if invalid_count > total * 0.1:
            logger.warning(
                f"High invalid picks: {invalid_count}/{total} ({invalid_count / total:.1%})"
            )

        return picks_samples, stats

    # In src/preprocessing/processor.py - update create_mask_vectorized

    def create_mask_vectorized(self, picks: np.ndarray) -> np.ndarray:
        """
        Create 3-class segmentation mask using vectorized operations.

        Class mapping:
            -1: Unlabeled / IGNORE (not used in training)
            0: Before first break
            2: Strip around first break
            1: After first break
        """
        n_traces = len(picks)
        mask = np.zeros((n_traces, self.n_samples), dtype=np.int64)

        # Create sample index grid using broadcasting
        samples = np.arange(self.n_samples).reshape(1, -1)
        picks_expanded = picks.reshape(-1, 1)

        # ✅ Use broadcasting directly
        valid_mask = ((picks > 0) & (picks < self.n_samples)).reshape(-1, 1)

        # Vectorized conditions for labeled traces
        strip_mask = (samples >= picks_expanded - self.half_width) & (
            samples <= picks_expanded + self.half_width
        )
        after_mask = samples > picks_expanded + self.half_width

        # Apply to valid traces only
        mask[valid_mask & strip_mask] = 2
        mask[valid_mask & after_mask] = 1

        # Invalid picks become ignore_index
        invalid = (picks <= 0) | (picks >= self.n_samples)
        mask[invalid, :] = self.ignore_index

        return mask

    def validate_mask(
        self, mask: np.ndarray, picks: np.ndarray, shot_id: int | None = None
    ) -> bool:
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
                            logger.debug(
                                f"Shot {shot_id}, trace {i}: strip center {strip_center:.0f} far from pick {pick:.0f}"
                            )

        if issues > 0:
            logger.debug(f"Shot {shot_id}: {issues} traces with misaligned strips")

        return True

    def get_shot_statistics(self, shot_picks: np.ndarray) -> dict:
        """Compute statistics for a shot."""
        valid_picks = shot_picks[shot_picks > 0]

        if len(valid_picks) > 0:
            return {
                "n_traces": len(shot_picks),
                "n_valid": len(valid_picks),
                "n_invalid": len(shot_picks) - len(valid_picks),
                "invalid_ratio": (len(shot_picks) - len(valid_picks)) / len(shot_picks),
                "min_pick": float(valid_picks.min()),
                "max_pick": float(valid_picks.max()),
                "mean_pick": float(valid_picks.mean()),
                "median_pick": float(np.median(valid_picks)),
            }
        else:
            return {
                "n_traces": len(shot_picks),
                "n_valid": 0,
                "n_invalid": len(shot_picks),
                "invalid_ratio": 1.0,
                "min_pick": None,
                "max_pick": None,
                "mean_pick": None,
                "median_pick": None,
            }

    def process_shot(
        self,
        shot_data: np.ndarray,
        shot_picks: np.ndarray,
        shot_id: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
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
        cleaned_picks, _ = self.validate_picks(shot_picks)

        # Pad or crop to target_traces
        if actual_traces < self.target_traces:
            data_padded = np.zeros(
                (self.target_traces, self.n_samples), dtype=np.float32
            )
            picks_padded = np.zeros(self.target_traces, dtype=np.float32)
            data_padded[:actual_traces, :] = shot_data
            picks_padded[:actual_traces] = cleaned_picks
            shot_data = data_padded
            shot_picks = picks_padded
            if self.log_level == "INFO":
                logger.debug(
                    f"Shot {shot_id}: padded {actual_traces} → {self.target_traces} traces"
                )
        elif actual_traces > self.target_traces:
            shot_data = shot_data[: self.target_traces, :]
            shot_picks = cleaned_picks[: self.target_traces]
            if self.log_level == "INFO":
                logger.debug(
                    f"Shot {shot_id}: cropped {actual_traces} → {self.target_traces} traces"
                )
        else:
            shot_picks = cleaned_picks

        # Create mask
        mask = self.create_mask_vectorized(shot_picks)

        # Validate mask quality (only if pick valid)
        if np.any(shot_picks > 0):
            self.validate_mask(mask, shot_picks, shot_id)

        # Collect statistics
        stats = self.get_shot_statistics(shot_picks)
        stats.update(
            {
                "original_traces": actual_traces,
                "padded_or_cropped": actual_traces != self.target_traces,
            }
        )
        self.stats.append(stats)

        if self.log_level == "DEBUG" and shot_id is not None:
            unique, counts = np.unique(mask, return_counts=True)
            class_dist = dict(zip(unique.tolist(), counts.tolist()))
            logger.debug(f"Shot {shot_id}: mask classes {class_dist}")

        return shot_data.astype(np.float32), mask, stats

    def get_all_stats(self) -> dict:
        """Get aggregate statistics for all processed shots."""
        if not self.stats:
            return {}

        n_valid = [s["n_valid"] for s in self.stats]
        n_invalid = [s["n_invalid"] for s in self.stats]
        n_traces = [s["n_traces"] for s in self.stats]

        valid_picks = []
        for s in self.stats:
            if s["min_pick"] is not None:
                valid_picks.append(s["min_pick"])
                valid_picks.append(s["max_pick"])

        return {
            "total_shots": len(self.stats),
            "total_traces": sum(n_traces),
            "total_valid": sum(n_valid),
            "total_invalid": sum(n_invalid),
            "avg_valid_per_shot": np.mean(n_valid),
            "avg_invalid_per_shot": np.mean(n_invalid),
            "shots_with_no_valid_picks": sum(
                1 for s in self.stats if s["n_valid"] == 0
            ),
            "min_pick_overall": min(valid_picks) if valid_picks else None,
            "max_pick_overall": max(valid_picks) if valid_picks else None,
        }

    def reset_stats(self):
        """Reset accumulated statistics."""
        self.stats = []
