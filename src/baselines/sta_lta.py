# file location: src/baselines/sta_lta.py

"""
STA/LTA first-break picker.

Classic short-term average / long-term average onset detector.

For each trace:
    1. Compute a per-sample energy envelope (squared or absolute).
    2. Compute a short-term moving average (STA) of the envelope.
    3. Compute a long-term moving average (LTA) of the envelope.
    4. Compute the ratio STA / LTA.
    5. Report the first sample where the ratio exceeds `threshold`
       and the sample index is at least `min_pick_sample`.

If no sample satisfies the condition, the pick is 0 for that trace.
"""

import numpy as np
from loguru import logger

from src.baselines.base import BaselinePicker, validate_shot_data


class STALTAPicker(BaselinePicker):
    """
    STA/LTA-based first-break picker.

    Args:
        sta_window: Length of the short-term average window, in samples.
        lta_window: Length of the long-term average window, in samples.
            Must be strictly greater than `sta_window`.
        threshold: Ratio threshold for declaring a pick. Typical
            values are 2.0–5.0. Higher threshold → later picks,
            fewer false positives.
        min_pick_sample: Earliest sample index a pick may be reported
            at. Prevents triggering on the first few samples where the
            LTA estimate is not yet stable.
        use_squared_envelope: If True (default), the envelope is the
            squared amplitude; if False, the absolute amplitude. The
            squared envelope emphasizes large amplitudes, which
            generally helps for first-break detection.
        epsilon: Small constant added to LTA to avoid divide-by-zero.

    Raises:
        ValueError: If constructor arguments are invalid.
    """

    def __init__(
        self,
        sta_window: int = 20,
        lta_window: int = 100,
        threshold: float = 3.0,
        min_pick_sample: int = 10,
        use_squared_envelope: bool = True,
        epsilon: float = 1e-8,
    ):
        if sta_window <= 0:
            raise ValueError(f"sta_window must be positive, got {sta_window}")
        if lta_window <= 0:
            raise ValueError(f"lta_window must be positive, got {lta_window}")
        if sta_window >= lta_window:
            raise ValueError(
                f"sta_window ({sta_window}) must be strictly less than "
                f"lta_window ({lta_window})"
            )
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if min_pick_sample < 0:
            raise ValueError(
                f"min_pick_sample must be non-negative, got {min_pick_sample}"
            )

        self.sta_window = sta_window
        self.lta_window = lta_window
        self.threshold = threshold
        self.min_pick_sample = min_pick_sample
        self.use_squared_envelope = use_squared_envelope
        self.epsilon = epsilon

    @property
    def name(self) -> str:
        return "STALTA"

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def pick(self, shot_data: np.ndarray) -> np.ndarray:
        """
        Pick first breaks for all traces in a shot.

        Args:
            shot_data: (n_traces, n_samples) — raw amplitudes.

        Returns:
            (n_traces,) int64 — pick index per trace; 0 = no pick.
        """
        validate_shot_data(shot_data, self.name)

        shot_data = shot_data.astype(np.float64, copy=False)
        n_traces, n_samples = shot_data.shape

        picks = np.zeros(n_traces, dtype=np.int64)

        for i in range(n_traces):
            picks[i] = self._pick_single_trace(shot_data[i])

        n_found = int(np.count_nonzero(picks))
        logger.debug(
            f"[{self.name}] Picked {n_found}/{n_traces} traces "
            f"(sta={self.sta_window}, lta={self.lta_window}, "
            f"thr={self.threshold})"
        )
        return picks

    # ----------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------

    def _pick_single_trace(self, trace: np.ndarray) -> int:
        """
        Pick a single trace.

        Args:
            trace: (n_samples,) — raw amplitudes.

        Returns:
            Pick sample index (int), or 0 if no pick found.
        """
        n_samples = len(trace)

        # If the trace is too short for the LTA window, we cannot pick.
        if n_samples < self.lta_window:
            return 0

        # 1. Envelope
        if self.use_squared_envelope:
            envelope = trace * trace
        else:
            envelope = np.abs(trace)

        # 2. STA and LTA via moving averages
        sta = self._moving_average(envelope, self.sta_window)
        lta = self._moving_average(envelope, self.lta_window)

        # 3. Ratio
        ratio = sta / (lta + self.epsilon)

        # 4. First sample above threshold, respecting min_pick_sample
        #    Search starts at min_pick_sample so we ignore the warm-up
        #    region where the LTA estimate is unstable.
        if self.min_pick_sample >= n_samples:
            return 0

        search = ratio[self.min_pick_sample :]
        above = search > self.threshold

        if not np.any(above):
            return 0

        first_idx = int(np.argmax(above)) + self.min_pick_sample
        return first_idx

    def _moving_average(self, x: np.ndarray, window: int) -> np.ndarray:
        """
        Compute a true centered moving average of `x` over `window` samples.

        At each index i, the output is:
            mean(x[max(0, i-half) : min(n, i+half+1)])

        where `half = window // 2`. The window is normalized by the
        actual number of samples included at each position, so there
        are no boundary artifacts or zero-padding effects.

        Uses a cumulative-sum approach so the whole operation is O(n)
        regardless of window size.

        Args:
            x: (n_samples,) input signal.
            window: window size in samples (>= 1).

        Returns:
            (n_samples,) moving average with the same shape as `x`.
        """
        n = len(x)
        half = window // 2

        # Cumulative sum with a leading zero so cs[i] = sum(x[:i])
        cs = np.concatenate([[0.0], np.cumsum(x)])

        # For each i: sum over [max(0, i-half), min(n, i+half+1))
        idx = np.arange(n)
        left = np.maximum(0, idx - half)
        right = np.minimum(n, idx + half + 1)

        sums = cs[right] - cs[left]
        counts = (right - left).astype(np.float64)

        return sums / counts

    # ----------------------------------------------------------------
    # Introspection
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"STALTAPicker("
            f"sta_window={self.sta_window}, "
            f"lta_window={self.lta_window}, "
            f"threshold={self.threshold}, "
            f"min_pick_sample={self.min_pick_sample}, "
            f"use_squared_envelope={self.use_squared_envelope}"
            f")"
        )

    def to_dict(self) -> dict:
        """Serializable representation for logging and MLflow tags."""
        return {
            "name": self.name,
            "sta_window": self.sta_window,
            "lta_window": self.lta_window,
            "threshold": self.threshold,
            "min_pick_sample": self.min_pick_sample,
            "use_squared_envelope": self.use_squared_envelope,
        }
