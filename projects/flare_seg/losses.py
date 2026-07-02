from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Binary segmentation loss for logits shaped [B, 1, H, W]."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if target.ndim == 3:
            target = target.unsqueeze(1)
        target = target.float()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight,
        )
        prob = torch.sigmoid(logits)
        dims = tuple(range(1, prob.ndim))
        intersection = (prob * target).sum(dim=dims)
        union = prob.sum(dim=dims) + target.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth))
        dice_loss = dice_loss.mean()
        loss = self.bce_weight * bce + self.dice_weight * dice_loss

        pred_mask = (prob > 0.5).float()
        target_mask = (target > 0.5).float()
        pred_area = pred_mask.mean()
        target_area = target_mask.mean()
        iou = (
            (pred_mask * target_mask).sum(dim=dims)
            / ((pred_mask + target_mask).clamp(max=1.0).sum(dim=dims) + 1e-6)
        ).mean()

        return {
            "loss": loss,
            "bce": bce.detach(),
            "dice": dice_loss.detach(),
            "iou_0.5": iou.detach(),
            "pred_area": pred_area.detach(),
            "target_area": target_area.detach(),
        }

