"""
Evaluation metrics for seismic FBP.
"""

import numpy as np
import torch
from torch import nn


class SegmentationMetrics:
    """
    Segmentation metrics for U-Net predictions.
    Accumulates metrics on GPU to avoid CPU-GPU synchronization.
    """

    def __init__(self, num_classes: int = 3, ignore_index: int = -1, device: torch.device = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device if device is not None else torch.device("cpu")
        self.reset()

    def reset(self):
        """Reset accumulated metrics on device."""
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64, device=self.device
        )
        self.total_pixels = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix with batch using device tensors.
        No CPU synchronization until compute() is called.
        """
        preds = predictions.detach().flatten()
        targets = targets.detach().flatten()

        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        valid_mask = (
            (preds >= 0)
            & (preds < self.num_classes)
            & (targets >= 0)
            & (targets < self.num_classes)
        )
        valid_pred = preds[valid_mask]
        valid_target = targets[valid_mask]

        if len(valid_pred) > 0:
            indices = self.num_classes * valid_target + valid_pred
            counts = torch.bincount(indices, minlength=self.num_classes ** 2)
            self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

        self.total_pixels += len(preds)

    def compute(self) -> dict[str, float]:
        """Compute all metrics (moves to CPU once per epoch)."""
        cm = self.confusion_matrix.cpu().numpy()

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

            denominator = tp + fp + fn
            iou = tp / denominator if denominator > 0 else 0
            iou_per_class.append(iou)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_per_class.append(precision)

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_per_class.append(recall)

            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            f1_per_class.append(f1)

        return {
            "accuracy": float(accuracy),
            "mean_iou": float(np.mean(iou_per_class)),
            "mean_f1": float(np.mean(f1_per_class)),
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
    return total_norm**0.5


def compute_weight_norm(model: nn.Module) -> float:
    """Compute the total weight norm of all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5


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
    Extract first break picks from segmentation mask.

    Args:
        mask: (n_traces, n_samples) segmentation mask (classes 0, 1, 2)

    Returns:
        picks: (n_traces,) pick positions in samples
    """
    n_traces = mask.shape[0]
    picks = np.zeros(n_traces, dtype=np.int64)

    for i in range(n_traces):
        # Find first occurrence of class 2 (strip) or class 1 (after)
        strip_indices = np.where(mask[i] == 2)[0]
        after_indices = np.where(mask[i] == 1)[0]

        if len(strip_indices) > 0:
            # Pick is the center of the strip
            picks[i] = int(np.median(strip_indices))
        elif len(after_indices) > 0:
            # Pick is the first after pixel minus strip_width/2
            picks[i] = after_indices[0] - 4
        else:
            picks[i] = 0

    return picks
