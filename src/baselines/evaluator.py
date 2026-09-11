# file location: src/baselines/evaluator.py

"""
Evaluation of classical baseline pickers.

Runs a BaselinePicker on a set of shots, collects metrics, and builds
a per-trace DataFrame. Pure computation — no I/O, no MLflow, no plots.
Those concerns live in the CLI script.

Input contract:
    Each shot is a tuple (shot_id, shot_data, ground_truth_picks_samples):
        shot_id:                  int
        shot_data:                (n_traces, n_samples) float
        ground_truth_picks_samples: (n_traces,) int, already in SAMPLES
                                    (not milliseconds), 0 = invalid
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.baselines.base import BaselinePicker, validate_shot_data
from src.training.metrics import FirstBreakMetrics


class BaselineEvaluator:
    """
    Runs a BaselinePicker on a list of shots and collects metrics.

    Usage:
        evaluator = BaselineEvaluator(picker, logger)
        result = evaluator.evaluate_shots(shots)
    """

    def __init__(
        self,
        picker: BaselinePicker,
        logger=logger,
        tolerance_samples: int = 3,
    ):
        self.picker = picker
        self.logger = logger
        self.tolerance_samples = tolerance_samples

    def evaluate_shots(
        self,
        shots: list[tuple[int, np.ndarray, np.ndarray]],
    ) -> dict[str, Any]:
        """
        Run the picker on each shot and collect metrics + per-trace data.

        Args:
            shots: list of (shot_id, shot_data, gt_picks_samples).
                   gt_picks_samples must be int64/float with 0 = invalid.

        Returns:
            dict with keys:
                metrics:    dict from FirstBreakMetrics.compute()
                            plus n_shots, n_traces
                per_trace:  pd.DataFrame with columns:
                            shot_id, trace_index, gt_pick_sample,
                            pred_pick_sample, error_samples, error_ms
                n_shots:    int
                n_traces:   int
                picker_name: str
                picker_params: dict
        """
        if not shots:
            raise ValueError("evaluate_shots called with empty shot list")

        fb_metrics = FirstBreakMetrics(tolerance_samples=self.tolerance_samples)

        per_trace_rows: list[dict[str, Any]] = []
        total_traces = 0

        for shot_id, shot_data, gt_picks in shots:
            # Validate input
            validate_shot_data(shot_data, self.picker.name)

            n_traces = shot_data.shape[0]
            total_traces += n_traces

            # Run the picker
            pred_picks = self.picker.pick(shot_data)

            # Make sure gt_picks is a numpy int array
            gt_picks_arr = np.asarray(gt_picks).astype(np.int64)

            # Update aggregate metrics
            fb_metrics.update(pred_picks, gt_picks_arr)

            # Per-trace rows: only where both GT and prediction are valid
            for i in range(n_traces):
                gt = int(gt_picks_arr[i])
                pred = int(pred_picks[i])
                if gt > 0 and pred > 0:
                    error_samples = abs(pred - gt)
                    per_trace_rows.append(
                        {
                            "shot_id": shot_id,
                            "trace_index": i,
                            "gt_pick_sample": gt,
                            "pred_pick_sample": pred,
                            "error_samples": error_samples,
                        }
                    )

            self.logger.debug(
                f"[Evaluator] Shot {shot_id}: "
                f"picked {int(np.count_nonzero(pred_picks))}/{n_traces} traces"
            )

        per_trace_df = pd.DataFrame(per_trace_rows)

        # Compute aggregate metrics
        metrics = fb_metrics.compute()
        metrics["n_shots"] = len(shots)
        metrics["n_traces"] = total_traces

        self.logger.info(
            f"[Evaluator] {self.picker.name}: "
            f"{len(shots)} shots, {total_traces} traces, "
            f"{len(per_trace_df)} valid comparisons"
        )

        # Serialize picker params for logging
        picker_params = self.picker.to_dict() if hasattr(self.picker, "to_dict") else {}

        return {
            "metrics": metrics,
            "per_trace": per_trace_df,
            "n_shots": len(shots),
            "n_traces": total_traces,
            "picker_name": self.picker.name,
            "picker_params": picker_params,
        }
