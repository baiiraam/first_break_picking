#!/usr/bin/env python3
"""
Loss function factory for seismic FBP with ignore_index support.
"""

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classes with ignore_index support."""

    def __init__(
        self,
        alpha: list[float] | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -1,  # 🆕 Add this
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index  # 🆕 Store it

        # ✅ Register as buffer to auto-move to device
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 🆕 Create mask for valid pixels
        valid_mask = targets != self.ignore_index

        # 🆕 If no valid pixels, return zero loss
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # 🆕 Replace -1 with 0 to avoid out-of-bounds error
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0

        # Compute CE loss (with reduction='none')
        ce_loss = F.cross_entropy(
            inputs, targets_safe, reduction="none", weight=self.alpha
        )

        # 🆕 Zero out ignored pixels
        ce_loss = ce_loss * valid_mask.float()

        # Focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 🆕 Only reduce over valid pixels
        if self.reduction == "mean":
            return focal_loss.sum() / valid_mask.float().sum()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation with ignore_index support."""

    def __init__(
        self,
        smooth: float = 1e-6,
        num_classes: int = 3,
        ignore_index: int = -1,  # 🆕 Add this
    ):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.ignore_index = ignore_index  # 🆕 Store it

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 🆕 Create mask for valid pixels
        valid_mask = target != self.ignore_index

        # 🆕 If no valid pixels, return zero loss
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # 🆕 Replace -1 with 0 to avoid out-of-bounds error
        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = (
            F.one_hot(target_safe, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        # 🆕 Mask out ignored pixels
        valid_mask_expanded = valid_mask.unsqueeze(1).float()
        pred_masked = pred_soft * valid_mask_expanded
        target_masked = target_one_hot * valid_mask_expanded

        intersection = (pred_masked * target_masked).sum(dim=(0, 2, 3))
        union = pred_masked.sum(dim=(0, 2, 3)) + target_masked.sum(dim=(0, 2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class ComboLoss(nn.Module):
    """Combined loss: CE + Focal + Dice with ignore_index support."""

    def __init__(
        self,
        class_weights: list[float] | None = None,
        dice_weight: float = 0.5,
        focal_gamma: float = 2.0,
        ignore_index: int = -1,  # 🆕 Add this
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma
        self.ignore_index = ignore_index  # 🆕 Store it

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 🆕 Create mask for valid pixels
        valid_mask = target != self.ignore_index

        # 🆕 If no valid pixels, return zero loss
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 🆕 Replace -1 with 0 to avoid out-of-bounds error
        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        # Cross Entropy Loss (with manual ignore_index)
        ce_loss = F.cross_entropy(
            logits, target_safe, reduction="none", weight=self.class_weights
        )
        ce_loss = ce_loss * valid_mask.float()
        ce_loss = ce_loss.sum() / valid_mask.float().sum()

        # Focal Loss (with manual ignore_index)
        probs = F.softmax(logits, dim=1)
        focal = (1 - probs) ** self.focal_gamma * -torch.log(probs + 1e-7)
        focal = focal.gather(1, target_safe.unsqueeze(1)).squeeze(1)
        focal = focal * valid_mask.float()
        focal_loss = focal.sum() / valid_mask.float().sum()

        # Dice Loss (with manual ignore_index)
        target_oh = F.one_hot(target_safe, probs.shape[1]).permute(0, 3, 1, 2).float()
        valid_mask_expanded = valid_mask.unsqueeze(1).float()
        probs_masked = probs * valid_mask_expanded
        target_masked = target_oh * valid_mask_expanded

        dims = (0, 2, 3)
        intersection = (probs_masked * target_masked).sum(dims)
        union = probs_masked.sum(dims) + target_masked.sum(dims)
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice.mean()

        # Combined loss
        combined_ce_focal = 0.5 * ce_loss + 0.5 * focal_loss
        total_loss = (
            1 - self.dice_weight
        ) * combined_ce_focal + self.dice_weight * dice_loss

        if return_components:
            per_class_loss = self._compute_per_class_loss(logits, target)

            return total_loss, {
                "total": total_loss.item(),
                "ce": ce_loss.item(),
                "focal": focal_loss.item(),
                "dice": dice_loss.item(),
                "ce_focal_combined": combined_ce_focal.item(),
                "per_class": per_class_loss,
            }

        return total_loss

    def _compute_per_class_loss(self, logits, target, num_classes=3):
        """Compute loss per class for monitoring."""
        per_class = {}
        for c in range(num_classes):
            mask = target == c
            if mask.sum() > 0:
                class_target = torch.zeros_like(logits[:, c, :, :])
                class_target[mask] = 1.0
                loss = F.binary_cross_entropy_with_logits(
                    logits[:, c, :, :][mask], class_target[mask], reduction="mean"
                )
                per_class[f"class_{c}"] = loss.item()
            else:
                per_class[f"class_{c}"] = 0.0
        per_class["strip"] = per_class.get("class_2", 0.0)
        return per_class


def create_loss_function(config) -> nn.Module:
    """Factory function to create loss function from config."""

    loss_type = getattr(config, "loss_function", "cross_entropy")
    class_weights = getattr(config, "class_weights", [0.2, 0.2, 0.6])
    ignore_index = getattr(
        config, "ignore_index", -1
    )  # 🆕 Get ignore_index from config

    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights),
            ignore_index=ignore_index,  # 🆕 Pass ignore_index
        )

    elif loss_type == "focal":
        return FocalLoss(
            alpha=class_weights,
            gamma=getattr(config, "focal_gamma", 2.0),
            ignore_index=ignore_index,  # 🆕 Pass ignore_index
        )

    elif loss_type == "dice":
        return DiceLoss(
            num_classes=len(class_weights),
            ignore_index=ignore_index,  # 🆕 Pass ignore_index
        )

    elif loss_type == "combo":
        return ComboLoss(
            class_weights=class_weights,
            dice_weight=getattr(config, "dice_weight", 0.5),
            focal_gamma=getattr(config, "focal_gamma", 2.0),
            ignore_index=ignore_index,  # 🆕 Pass ignore_index
        )

    else:
        raise ValueError(f"Unknown loss function: {loss_type}")
