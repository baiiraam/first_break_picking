# file location: src/explainability/base.py

"""
Abstract base class for explainability methods, and shared utilities.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn


class Explainer(ABC):
    """
    Abstract base for model explainers.

    Subclasses produce a (H, W) heatmap for a given input, highlighting
    the regions that most influenced the model's prediction.
    """

    @abstractmethod
    def explain(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute a heatmap for a single input.

        Args:
            model: PyTorch model, in eval mode.
            input_tensor: (1, C, H, W) input tensor.
            target_class: which class to explain. If None, uses argmax.

        Returns:
            (H, W) float32 heatmap, values normalized to [0, 1].
        """
        raise NotImplementedError


def find_last_conv_layer(model: nn.Module) -> nn.Conv2d:
    """
    Walk the model's module tree and return the last nn.Conv2d.

    This is the standard target for Grad-CAM: the last convolutional
    layer retains spatial structure, so its feature maps can be
    weighted by gradients and upsampled to the input size.

    Raises:
        ValueError: if the model contains no Conv2d layers.
    """
    last_conv: nn.Conv2d | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError(
            "No nn.Conv2d layer found in the model. Grad-CAM requires "
            "at least one convolutional layer."
        )
    return last_conv