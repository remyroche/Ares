"""Stable IRM loss functions for differentiable models."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


def _bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, sample_weights: Optional[torch.Tensor]) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss


def _focal_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
    sample_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pt = torch.where(torch.eq(targets, 1.0), probs, 1.0 - probs)
    loss = -alpha * torch.pow(1.0 - pt.clamp(min=1e-8), gamma) * torch.log(pt.clamp(min=1e-8))
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss


class StableIRMLoss(nn.Module):
    """Differentiable IRM loss with optional focal base loss and variance penalty."""

    def __init__(
        self,
        base_loss: str = "bce",
        lambda_irm: float = 1.0,
        lambda_variance: float = 1.0,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        min_env_samples: int = 32,
        min_env_samples_end: Optional[int] = None,
        env_subsample_rate: float = 1.0,
        use_amp: bool = False,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.lambda_irm_max = lambda_irm
        self.lambda_variance_max = lambda_variance
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.min_env_samples_start = min_env_samples
        self.min_env_samples_end = min_env_samples_end or min_env_samples
        self.env_subsample_rate = float(max(0.0, min(1.0, env_subsample_rate)))
        self.use_amp = use_amp
        self._anneal_progress = 1.0
        self._update_annealed_values()

    def _update_annealed_values(self):
        progress = self._anneal_progress
        self.lambda_irm = self.lambda_irm_max * progress
        self.lambda_variance = self.lambda_variance_max * progress
        self.min_env_samples = int(
            round(
                self.min_env_samples_start * (1.0 - progress)
                + self.min_env_samples_end * progress
            )
        )
        self.min_env_samples = max(1, self.min_env_samples)

    def set_anneal_progress(self, progress: float):
        self._anneal_progress = float(max(0.0, min(1.0, progress)))
        self._update_annealed_values()

    def _base_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.base_loss.lower() == "focal":
            return _focal_binary_loss(logits, targets, self.focal_alpha, self.focal_gamma, sample_weights)
        return _bce_with_logits(logits, targets, sample_weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        env_ids: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if logits.ndim != 1:
            logits = logits.view(-1)
        if targets.ndim != 1:
            targets = targets.view(-1)
        if sample_weights is not None and sample_weights.ndim != 1:
            sample_weights = sample_weights.view(-1)

        base_loss = self._base_loss(logits, targets, sample_weights).mean()

        unique_envs = torch.unique(env_ids[env_ids >= 0])
        if self.env_subsample_rate < 1.0 and unique_envs.numel() > 0:
            keep_mask = torch.rand_like(unique_envs.float()) <= self.env_subsample_rate
            if keep_mask.any():
                unique_envs = unique_envs[keep_mask]
            else:
                unique_envs = unique_envs[:1]
        irm_penalties = []
        env_means = []
        eps = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        for env in unique_envs:
            mask = env_ids == env
            count = mask.sum()
            if count < self.min_env_samples:
                continue

            dummy_scale = torch.tensor(1.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
            env_logits = logits[mask] * dummy_scale
            env_targets = targets[mask]
            env_weights = sample_weights[mask] if sample_weights is not None else None
            with autocast(enabled=self.use_amp):
                env_loss = self._base_loss(env_logits, env_targets, env_weights).mean()
            grad = torch.autograd.grad(env_loss, dummy_scale, create_graph=True)[0]
            irm_penalties.append(grad.pow(2))

            env_probs = torch.sigmoid(logits[mask].detach())
            env_means.append(env_probs.mean())

        irm_penalty = torch.stack(irm_penalties).mean() if irm_penalties else eps
        if len(env_means) > 1:
            stacked_means = torch.stack(env_means)
            variance_penalty = torch.var(stacked_means)
        else:
            variance_penalty = eps

        total_loss = base_loss + self.lambda_irm * irm_penalty + self.lambda_variance * variance_penalty
        breakdown = {
            "base_loss": base_loss.detach(),
            "irm_penalty": irm_penalty.detach(),
            "variance_penalty": variance_penalty.detach(),
            "total_loss": total_loss.detach(),
        }
        return total_loss, breakdown


def build_env_id_tensor(
    environment_masks: Dict[str, torch.Tensor],
    n_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert dictionary of environment masks to env_id tensor."""
    env_ids = torch.full((n_samples,), -1, dtype=torch.long, device=device)
    for env_idx, mask in enumerate(environment_masks.values()):
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
        if mask_tensor.shape[0] != n_samples:
            raise ValueError("Environment mask length does not match sample count")
        env_ids[mask_tensor] = env_idx
    return env_ids
