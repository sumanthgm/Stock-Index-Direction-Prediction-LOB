"""
Loss functions for the class-imbalance study (Tier-C).

The LOB direction labels are imbalanced (the Stationary class dominates at long
horizons under the fixed threshold), which is why macro-F1 trails weighted-F1.
This module provides Focal loss; the other strategies in the study reuse PyTorch
built-ins (class-weighted CE, label-smoothing CE) and a class-balanced sampler.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017).

    Down-weights well-classified examples by ``(1 - p_t)**gamma`` so training
    focuses on the hard, minority directional classes. ``alpha`` is an optional
    per-class weight vector (length = n_classes); ``gamma=0`` recovers CE.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        # registered as a buffer so .to(device) moves it with the module
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(log_p, target, weight=self.alpha, reduction="none")
        p_t = torch.exp(-ce) if self.alpha is None else log_p.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        loss = (1.0 - p_t) ** self.gamma * ce
        return loss.mean()
