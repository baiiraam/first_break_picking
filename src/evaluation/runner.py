# file location: src/evaluation/runner.py

"""
Evaluation runner for seismic datasets.
Handles batch-wise evaluation, metric collection, and detailed error tracking.
"""

from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import SeismicConfig
from src.training.metrics import (
    FirstBreakMetrics,
    SegmentationMetrics,
    extract_picks_from_mask,
)
from src.types import LoggerType


class EvaluationRunner:
    """
    Runs evaluation on a dataset split and collects metrics.

    Example:
        >>> runner = EvaluationRunner(model, device, cfg, logger, detailed=True)
        >>> metrics, detailed_results = runner.evaluate_split(loader, dataset, "test")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device_obj: torch.device,
        cfg: SeismicConfig,
        logger: LoggerType,
        detailed: bool = False,
    ):
        """
        Initialize evaluation runner.

        Args:
            model: PyTorch model to evaluate
            device_obj: Target device (cpu/cuda/mps)
            cfg: Configuration object
            logger: logger instance
            detailed: Whether to collect per-shot detailed metrics
        """
        self.model = model
        self.device_obj = device_obj
        self.cfg = cfg
        self.logger = logger
        self.detailed = detailed

    def evaluate_split(
        self,
        loader: DataLoader,
        dataset_obj: Any,
        split_name: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Evaluate model on a single data split.

        Args:
            loader: DataLoader for the split
            dataset_obj: Dataset object (for shot_id lookup)
            split_name: Name of the split (train/val/test)

        Returns:
            Tuple of (metrics_dict, detailed_results_dict)
        """
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"📊 Evaluating {split_name.upper()} SET")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(
            f"  {split_name}: {len(dataset_obj)} shots, {len(loader)} batches"
        )

        seg_metrics = SegmentationMetrics(num_classes=3)
        fb_metrics = FirstBreakMetrics(tolerance_samples=3)

        shot_errors: list[float] = []
        shot_ids: list[int] = []

        self.logger.info("\nRunning evaluation...")
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(
                tqdm(loader, desc=f"Evaluating {split_name}")
            ):
                x = x.to(self.device_obj)
                y = y.to(self.device_obj)

                outputs = self.model(x)

                preds = torch.argmax(outputs, dim=1)  # (B, H, W)
                seg_metrics.update(preds, y)

                # Iterate over batch
                preds_np = preds.cpu().numpy()  # (B, H, W)
                y_np = y.cpu().numpy()  # (B, H, W)

                for b in range(preds_np.shape[0]):
                    pred_picks_b = extract_picks_from_mask(preds_np[b])  # (H,)
                    true_picks_b = extract_picks_from_mask(y_np[b])  # (H,)

                    fb_metrics.update(pred_picks_b, true_picks_b)

                    if self.detailed:
                        self._collect_detailed_errors(
                            pred_picks=pred_picks_b,
                            true_picks=true_picks_b,
                            batch_idx=batch_idx,
                            shot_b_idx=b,  # NEW parameter
                            dataset_obj=dataset_obj,
                            shot_ids=shot_ids,
                            shot_errors=shot_errors,
                        )

        metrics = self._build_metrics_dict(
            split_name=split_name,
            seg_metrics=seg_metrics,
            fb_metrics=fb_metrics,
            dataset_obj=dataset_obj,
        )

        detailed_results = self._build_detailed_results(
            split_name=split_name,
            shot_ids=shot_ids,
            shot_errors=shot_errors,
        )

        self._log_metrics(metrics, detailed_results)

        return metrics, detailed_results

    def _collect_detailed_errors(
        self,
        pred_picks: np.ndarray,
        true_picks: np.ndarray,
        batch_idx: int,
        shot_b_idx: int,
        dataset_obj: Any,
        shot_ids: list[int],
        shot_errors: list[float],
    ) -> None:
        """Collect per-shot errors for detailed analysis."""
        for i in range(len(pred_picks)):
            if true_picks[i] > 0 and pred_picks[i] > 0:
                error = abs(pred_picks[i] - true_picks[i])
                shot_errors.append(float(error))

                try:
                    global_shot_idx = batch_idx * self.cfg.batch_size + shot_b_idx
                    shot_id = dataset_obj.get_shot_id(global_shot_idx)
                    shot_ids.append(int(shot_id))
                except (AttributeError, IndexError, KeyError):
                    shot_ids.append(global_shot_idx)

    def _build_metrics_dict(
        self,
        split_name: str,
        seg_metrics: SegmentationMetrics,
        fb_metrics: FirstBreakMetrics,
        dataset_obj: Any,
    ) -> dict[str, Any]:
        """Build metrics dictionary for a split."""
        seg_results = seg_metrics.compute()
        fb_results = fb_metrics.compute()

        return {
            split_name: {
                "segmentation": seg_results,
                "first_break": fb_results,
                "n_shots": len(dataset_obj),
            }
        }

    def _build_detailed_results(
        self,
        split_name: str,
        shot_ids: list[int],
        shot_errors: list[float],
    ) -> dict[str, Any]:
        """Build detailed results dictionary."""
        if not self.detailed or not shot_errors:
            return {}

        import pandas as pd

        detailed_df = pd.DataFrame(
            {
                "shot_id": shot_ids,
                "error_samples": shot_errors,
                "error_ms": np.array(shot_errors) * self.cfg.sampling_interval_ms,
            }
        )

        return {
            "split": split_name,
            "dataframe": detailed_df,
        }

    def _log_metrics(
        self,
        metrics: dict[str, Any],
        detailed_results: dict[str, Any],
    ) -> None:
        """Log metrics to console."""
        for split_name, results in metrics.items():
            seg = results["segmentation"]
            fb = results["first_break"]

            self.logger.info(f"\n📊 SEGMENTATION METRICS ({split_name.upper()})")
            self.logger.info("-" * 40)
            self.logger.info(f"  Accuracy: {seg['accuracy']:.4f}")
            self.logger.info(f"  Mean IoU: {seg['mean_iou']:.4f}")
            self.logger.info(f"  Mean F1: {seg['mean_f1']:.4f}")

            self.logger.info("\n  Class-wise IoU:")
            class_names = ["Before", "After", "Strip"]
            for name, iou in zip(class_names, cast(list[float], seg["iou_per_class"])):
                self.logger.info(f"    {name}: {iou:.4f}")

            self.logger.info(f"\n📊 FIRST-BREAK METRICS ({split_name.upper()})")
            self.logger.info("-" * 40)
            self.logger.info(
                f"  Mean Absolute Error (MAE): {fb['mean_absolute_error']:.2f} samples"
            )
            self.logger.info(
                f"  Std Absolute Error: {fb['std_absolute_error']:.2f} samples"
            )
            self.logger.info(
                f"  Median Absolute Error: {fb['median_absolute_error']:.2f} samples"
            )
            self.logger.info(
                f"  Accuracy within ±3 samples: {fb['accuracy_within_tolerance']:.2%}"
            )
            self.logger.info(f"  Total traces evaluated: {fb['total_traces']}")

            if self.detailed and detailed_results:
                errors_array = np.array(detailed_results["dataframe"]["error_samples"])
                self.logger.info("\n📊 ERROR DISTRIBUTION")
                self.logger.info("-" * 40)
                percentiles = [50, 75, 90, 95, 99]
                self.logger.info(f"  Min: {errors_array.min():.2f}")
                self.logger.info(f"  Max: {errors_array.max():.2f}")
                for p in percentiles:
                    self.logger.info(
                        f"  {p}th percentile: {np.percentile(errors_array, p):.2f}"
                    )
                self.logger.info(
                    f"  >5 samples: {(errors_array > 5).mean() * 100:.1f}%"
                )
                self.logger.info(
                    f"  >10 samples: {(errors_array > 10).mean() * 100:.1f}%"
                )
