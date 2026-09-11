# file location: src/explainability/gradcam.py

"""
Grad-CAM and HiResCAM for convolutional models.

Reference:
    Grad-CAM:  Selvaraju et al., "Grad-CAM: Visual Explanations from
               Deep Networks via Gradient-based Localization" (ICCV 2017)
    HiResCAM:  Draelos & Carin, "Use HiResCAM instead of Grad-CAM for
               faithful explanations of convolutional neural networks"
               (2020)
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.explainability.base import Explainer, find_last_conv_layer


class GradCAM(Explainer):
    """
    Grad-CAM (and HiResCAM) explainer for 2D convolutional models.

    Args:
        target_layer: the conv layer to hook. If None, uses
            `find_last_conv_layer(model)` at explain-time.
        method: "gradcam" (default) or "hirescam".

    Usage:
        cam = GradCAM()
        heatmap = cam.explain(model, input_tensor, target_class=2)
    """

    def __init__(
        self,
        target_layer: nn.Module | None = None,
        method: str = "gradcam",
    ):
        if method not in ("gradcam", "hirescam"):
            raise ValueError(
                f"method must be 'gradcam' or 'hirescam', got {method!r}"
            )
        self.target_layer = target_layer
        self.method = method

    def explain(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute a Grad-CAM heatmap for a single input.

        Args:
            model: PyTorch model in eval mode.
            input_tensor: (1, C, H, W). Must be on the same device as model.
            target_class: which class to explain. None = use argmax.

        Returns:
            (H, W) float32 heatmap, values normalized to [0, 1].
        """
        if input_tensor.dim() != 4:
            raise ValueError(
                f"input_tensor must be 4D (1, C, H, W), got shape "
                f"{tuple(input_tensor.shape)}"
            )
        if input_tensor.shape[0] != 1:
            raise ValueError(
                f"input_tensor must have batch size 1, got "
                f"{input_tensor.shape[0]}"
            )

        # Resolve the target layer
        target_layer = self.target_layer
        if target_layer is None:
            target_layer = find_last_conv_layer(model)

        # Storage for hook outputs
        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            activations["value"] = out.detach()

        def backward_hook(module: nn.Module, grad_in: Any, grad_out: Any) -> None:
            gradients["value"] = grad_out[0].detach()

        # Register hooks; always remove them in finally
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            model.zero_grad(set_to_none=True)

            # Forward
            was_training = model.training
            model.eval()
            logits = model(input_tensor)  # (1, C, H, W)

            # Resolve target class
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1)[0, 0, 0].item())

            if target_class >= logits.shape[1]:
                raise ValueError(
                    f"target_class={target_class} out of range "
                    f"(model has {logits.shape[1]} classes)"
                )

            # Score: sum of the target class logits over the spatial dims.
            # Summing lets gradients flow from every spatial location, which
            # is the right choice for dense prediction tasks like
            # segmentation.
            score = logits[0, target_class].sum()

            # Backward
            score.backward()

            if "value" not in activations:
                raise RuntimeError(
                    "Forward hook did not fire — target layer may not be "
                    "in the forward path."
                )
            if "value" not in gradients:
                raise RuntimeError(
                    "Backward hook did not fire — target layer may not be "
                    "receiving gradients. Is it actually part of the "
                    "forward computation?"
                )

            acts = activations["value"]      # (1, C_conv, H_conv, W_conv)
            grads = gradients["value"]       # (1, C_conv, H_conv, W_conv)

            # Compute channel weights
            if self.method == "gradcam":
                # Classic: weight per channel = global average of gradient
                weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
                cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
            else:  # hirescam
                # Element-wise weighting, then sum over channels
                cam = (grads * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)

            # ReLU: only positive contributions
            cam = F.relu(cam)

            # Upsample to input spatial size
            H, W = input_tensor.shape[2], input_tensor.shape[3]
            cam = F.interpolate(
                cam, size=(H, W), mode="bilinear", align_corners=False
            )  # (1, 1, H, W)

            # Normalize to [0, 1]
            cam_np = cam[0, 0].cpu().numpy().astype(np.float32)
            vmax = float(cam_np.max())
            if vmax > 0:
                cam_np = cam_np / vmax

            # Restore training mode if it was on
            if was_training:
                model.train()

            return cam_np

        finally:
            fwd_handle.remove()
            bwd_handle.remove()