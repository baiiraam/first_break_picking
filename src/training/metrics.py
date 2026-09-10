# file location: src/training/metrics.py

"""
Evaluation metrics for seismic FBP.
"""

from typing import TypedDict

import numpy as np
import torch
from torch import nn


class SegmentationResults(TypedDict):
    accuracy: float
    mean_iou: float
    mean_f1: float
    iou_per_class: list[float]
    precision_per_class: list[float]
    recall_per_class: list[float]
    f1_per_class: list[float]


class SegmentationMetrics:
    """
    Segmentation metrics for U-Net predictions.
    """

    def __init__(self, num_classes: int = 3, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
        self.total_pixels = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        pred = predictions.cpu().numpy().flatten()
        target = targets.cpu().numpy().flatten()

        # ✅ Fix: Filter out -1 values from both pred and target
        if self.ignore_index >= 0:
            valid_mask = (
                (target != self.ignore_index)
                & (pred >= 0)
                & (pred < self.num_classes)
                & (target >= 0)
                & (target < self.num_classes)
            )
        else:
            valid_mask = (
                (pred >= 0)
                & (pred < self.num_classes)
                & (target >= 0)
                & (target < self.num_classes)
            )

        pred = pred[valid_mask]
        target = target[valid_mask]

        if len(pred) == 0:
            return

        # ✅ Compute idx and bincount
        idx = target * self.num_classes + pred
        counts = np.bincount(idx, minlength=self.num_classes * self.num_classes)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> SegmentationResults:
        """Compute all metrics."""
        cm = self.confusion_matrix

        # Pixel accuracy
        accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0

        # Per-class metrics
        iou_per_class = []
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []

        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = np.sum(cm[:, c]) - tp
            fn = np.sum(cm[c, :]) - tp

            # IoU
            denominator = tp + fp + fn
            iou = tp / denominator if denominator > 0 else 0
            iou_per_class.append(iou)

            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_per_class.append(precision)

            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_per_class.append(recall)

            # F1
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            f1_per_class.append(f1)

        # Mean IoU
        mean_iou = np.mean(iou_per_class)

        # Mean F1
        mean_f1 = np.mean(f1_per_class)

        return {
            "accuracy": float(accuracy),
            "mean_iou": float(mean_iou),
            "mean_f1": float(mean_f1),
            "iou_per_class": [float(x) for x in iou_per_class],
            "precision_per_class": [float(x) for x in precision_per_class],
            "recall_per_class": [float(x) for x in recall_per_class],
            "f1_per_class": [float(x) for x in f1_per_class],
        }


class FirstBreakMetrics:
    """
    Metrics for first break picking accuracy.
    """

    def __init__(self, tolerance_samples: int = 3):
        self.tolerance_samples = tolerance_samples
        self.errors: list[float] = []
        self.within_tolerance: list[bool] = []
        self.total_traces: int = 0
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.errors = []
        self.within_tolerance = []
        self.total_traces = 0

    def update(self, predicted_picks: np.ndarray, true_picks: np.ndarray):
        """
        Update metrics with batch.

        Args:
            predicted_picks: (n_traces,) array of predicted pick positions in samples
            true_picks: (n_traces,) array of ground truth pick positions in samples
        """
        valid_mask = (true_picks > 0) & (predicted_picks > 0)
        pred = predicted_picks[valid_mask]
        true = true_picks[valid_mask]

        if len(pred) == 0:
            return

        errors = np.abs(pred - true)
        self.errors.extend(errors.tolist())
        self.within_tolerance.extend((errors <= self.tolerance_samples).tolist())
        self.total_traces += len(pred)

    def compute(self) -> dict[str, float]:
        """Compute all metrics."""
        if len(self.errors) == 0:
            return {
                "mean_absolute_error": 0.0,
                "std_absolute_error": 0.0,
                "accuracy_within_tolerance": 0.0,
                "median_absolute_error": 0.0,
                "max_absolute_error": 0.0,
                "min_absolute_error": 0.0,
                "total_traces": 0,
            }

        errors = np.array(self.errors)
        within = np.array(self.within_tolerance)

        return {
            "mean_absolute_error": float(np.mean(errors)),
            "std_absolute_error": float(np.std(errors)),
            "accuracy_within_tolerance": float(np.mean(within)),
            "median_absolute_error": float(np.median(errors)),
            "max_absolute_error": float(np.max(errors)),
            "min_absolute_error": float(np.min(errors)),
            "total_traces": self.total_traces,
        }


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute the total gradient norm of all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return float(total_norm**0.5)


def compute_weight_norm(model: nn.Module) -> float:
    """Compute the total weight norm of all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return float(total_norm**0.5)


def compute_layerwise_norms(model: nn.Module) -> dict[str, float]:
    """Compute weight and gradient norms per layer."""
    norms = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            norms[f"weights_{name}"] = p.data.norm(2).item()
            if p.grad is not None:
                norms[f"grads_{name}"] = p.grad.data.norm(2).item()
    return norms


def extract_picks_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Extract first break picks from segmentation mask using vectorized operations.
    """
    n_traces = mask.shape[0]
    picks = np.zeros(n_traces, dtype=np.int64)

    # Find strip positions for all traces at once
    strip_indices = np.argmax(mask == 2, axis=1)
    has_strip = np.any(mask == 2, axis=1)

    # For traces with strip, use strip center
    # For traces without strip, use first after pixel minus 4
    after_indices = np.argmax(mask == 1, axis=1)

    # Use numpy where for vectorized assignment
    picks[has_strip] = strip_indices[has_strip]
    picks[~has_strip] = np.maximum(after_indices[~has_strip] - 4, 0)

    return picks
