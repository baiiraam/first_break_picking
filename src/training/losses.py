#!/usr/bin/env python3
"""
Loss function factory for seismic FBP.
"""


import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classes."""

    def __init__(
        self,
        alpha: list[float] | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""

    def __init__(self, smooth: float = 1e-6, num_classes: int = 3):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = (
            F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        )

        intersection = (pred_soft * target_one_hot).sum(dim=(0, 2, 3))
        union = pred_soft.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


# ============================================================
# MODIFICATION 1: Add loss component tracking to ComboLoss
# ============================================================

class ComboLoss(nn.Module):
    def __init__(self, class_weights=None, dice_weight=0.5, focal_gamma=2.0):
        super().__init__()
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma
        self.ce_weight = 1 - dice_weight
        # NEW: Flag to control whether to return components
        self.return_components = False

    def forward(self, logits, target):
        # Cross Entropy Loss
        ce_loss = F.cross_entropy(logits, target, weight=self.class_weights)
        
        # Focal Loss
        probs = F.softmax(logits, dim=1)
        focal = (1 - probs) ** self.focal_gamma * -torch.log(probs + 1e-7)
        focal_loss = focal.gather(1, target.unsqueeze(1)).mean()
        
        # Dice Loss
        target_oh = F.one_hot(target, probs.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * target_oh).sum(dims)
        dice = (2 * intersection + 1e-6) / (probs.sum(dims) + target_oh.sum(dims) + 1e-6)
        dice_loss = 1 - dice.mean()
        
        # Combined loss
        combined_ce_focal = 0.5 * ce_loss + 0.5 * focal_loss
        total_loss = (1 - self.dice_weight) * combined_ce_focal + self.dice_weight * dice_loss
        
        # NEW: Return components if requested
        if self.return_components:
            # Per-class losses for monitoring
            per_class_loss = self._compute_per_class_loss(logits, target)
            
            return total_loss, {
                'total': total_loss.item(),
                'ce': ce_loss.item(),
                'focal': focal_loss.item(),
                'dice': dice_loss.item(),
                'ce_focal_combined': combined_ce_focal.item(),
                'per_class': per_class_loss,
            }
        
        return total_loss
    
    # NEW: Helper method for per-class loss
    def _compute_per_class_loss(self, logits, target, num_classes=3):
        """Compute loss per class for monitoring."""
        per_class = {}
        
        # Compute cross-entropy per class
        for c in range(num_classes):
            mask = (target == c)
            if mask.sum() > 0:
                class_target = torch.zeros_like(logits[:, c, :, :])
                class_target[mask] = 1.0
                loss = F.binary_cross_entropy_with_logits(
                    logits[:, c, :, :][mask],
                    class_target[mask],
                    reduction='mean'
                )
                per_class[f'class_{c}'] = loss.item()
            else:
                per_class[f'class_{c}'] = 0.0
        
        # Strip class (class 2) is the most important
        per_class['strip'] = per_class.get('class_2', 0.0)
        
        return per_class

def create_loss_function(config) -> nn.Module:
    """Factory function to create loss function from config."""

    loss_type = getattr(config, "loss_function", "cross_entropy")
    class_weights = getattr(config, "class_weights", [0.2, 0.2, 0.6])

    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

    elif loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=getattr(config, "focal_gamma", 2.0))

    elif loss_type == "dice":
        return DiceLoss(num_classes=len(class_weights))

    elif loss_type == "combo":
        return ComboLoss(
            class_weights=class_weights,
            dice_weight=getattr(config, "dice_weight", 0.5),
            focal_gamma=getattr(config, "focal_gamma", 2.0),
        )

    else:
        raise ValueError(f"Unknown loss function: {loss_type}")
