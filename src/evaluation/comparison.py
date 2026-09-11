# file location: src/evaluation/comparison.py

"""
Side-by-side comparison of ML model picks and classical baseline picks.

Runs both a trained PyTorch model and a BaselinePicker on the same shots,
extracts picks from each, and produces a per-trace DataFrame with both
methods' predictions and errors.

Pure computation — no MLflow, no plots. Those live in the CLI script.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.baselines.base import BaselinePicker, validate_shot_data
from src.training.metrics import FirstBreakMetrics, extract_picks_from_mask


class BaselineComparison:
    """
    Compares ML model predictions and classical baseline picks on the
    same shots.

    Usage:
        comp = BaselineComparison(ml_model, sta_picker, logger, device)
        result = comp.compare_shots(shots)
    """

    def __init__(
        self,
        ml_model: torch.nn.Module,
        sta_picker: BaselinePicker,
        logger=logger,
        device: torch.device | None = None,
        tolerance_samples: int = 3,
    ):
        self.ml_model = ml_model
        self.sta_picker = sta_picker
        self.logger = logger
        self.device = device if device is not None else torch.device("cpu")
        self.tolerance_samples = tolerance_samples

    def compare_shots(
        self,
        shots: list[tuple[int, np.ndarray, np.ndarray]],
    ) -> dict[str, Any]:
        """
        Run both methods on each shot and build a per-trace DataFrame.

        Args:
            shots: list of (shot_id, shot_data, gt_picks_samples).

        Returns:
            dict with keys:
                per_trace: DataFrame with columns:
                    shot_id, trace_index, gt_pick_sample,
                    sta_lta_pick, sta_lta_error_samples,
                    ml_pick, ml_error_samples
                ml_metrics: dict (FirstBreakMetrics.compute)
                sta_lta_metrics: dict
                n_shots: int
                n_traces: int
        """
        if not shots:
            raise ValueError("compare_shots called with empty shot list")

        ml_metrics = FirstBreakMetrics(tolerance_samples=self.tolerance_samples)
        sta_metrics = FirstBreakMetrics(tolerance_samples=self.tolerance_samples)

        per_trace_rows: list[dict[str, Any]] = []
        total_traces = 0

        self.ml_model.eval()

        for shot_id, shot_data, gt_picks in shots:
            validate_shot_data(shot_data, "Comparison")

            n_traces = shot_data.shape[0]
            total_traces += n_traces
            gt_picks_arr = np.asarray(gt_picks).astype(np.int64)

            # --- STA/LTA ---
            sta_picks = self.sta_picker.pick(shot_data)
            sta_metrics.update(sta_picks, gt_picks_arr)

            # --- ML model ---
            ml_picks = self._run_ml_model(shot_data)
            ml_metrics.update(ml_picks, gt_picks_arr)

            # --- Per-trace rows: only where GT valid for at least one method ---
            for i in range(n_traces):
                gt = int(gt_picks_arr[i])
                if gt <= 0:
                    continue

                sta_p = int(sta_picks[i])
                ml_p = int(ml_picks[i])

                row = {
                    "shot_id": shot_id,
                    "trace_index": i,
                    "gt_pick_sample": gt,
                    "sta_lta_pick": sta_p,
                    "sta_lta_error_samples": abs(sta_p - gt) if sta_p > 0 else -1,
                    "ml_pick": ml_p,
                    "ml_error_samples": abs(ml_p - gt) if ml_p > 0 else -1,
                }
                per_trace_rows.append(row)

            self.logger.debug(
                f"[Compare] Shot {shot_id}: "
                f"STA/LTA picked {int(np.count_nonzero(sta_picks))}/{n_traces}, "
                f"ML picked {int(np.count_nonzero(ml_picks))}/{n_traces}"
            )

        per_trace_df = pd.DataFrame(per_trace_rows)

        sta_result = sta_metrics.compute()
        ml_result = ml_metrics.compute()

        # Add summary counts
        sta_result["n_shots"] = len(shots)
        sta_result["n_traces"] = total_traces
        ml_result["n_shots"] = len(shots)
        ml_result["n_traces"] = total_traces

        self.logger.info(
            f"[Compare] {len(shots)} shots, {total_traces} traces, "
            f"{len(per_trace_df)} rows in per-trace DataFrame"
        )

        return {
            "per_trace": per_trace_df,
            "ml_metrics": ml_result,
            "sta_lta_metrics": sta_result,
            "n_shots": len(shots),
            "n_traces": total_traces,
        }

    # -----------------------------------------------------------------
    # ML inference
    # -----------------------------------------------------------------

    def _run_ml_model(self, shot_data: np.ndarray) -> np.ndarray:
        """
        Run the ML model on a shot, return picks in samples.
        """
        # shot_data: (n_traces, n_samples) float
        # Model expects (B, 1, H, W)
        x = torch.from_numpy(shot_data).float().unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            logits = self.ml_model(x)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()  # (H, W)

        picks = extract_picks_from_mask(pred)  # (n_traces,) int64
        return picks.astype(np.int64)