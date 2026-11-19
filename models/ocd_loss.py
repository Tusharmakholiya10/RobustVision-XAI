"""
models/ocd_loss.py — OOD-Contrastive Decoding Loss
Penalizes the model when token probabilities shift under visual corruptions
that should not alter semantic content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OODContrastiveDecodingLoss(nn.Module):
    """
    OCD-Loss: Distributional robustness objective for VLM fine-tuning.

    Mathematical formulation:
        L_OCD = E[KL(P(y | x_clean) || P(y | x_corrupted))]

    Encourages the model's output distribution to remain stable across
    mild corruptions that shouldn't alter semantic meaning.

    Args:
        alpha (float): Weight of the OCD loss term vs. standard CE loss.
        temperature (float): Softmax temperature for probability estimation.
        reduction (str): 'mean' or 'sum'.
    """

    def __init__(self, alpha: float = 0.4, temperature: float = 1.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha       = alpha
        self.temperature = temperature
        self.reduction   = reduction

    def forward(self,
                logits_clean:     torch.Tensor,
                logits_corrupted: torch.Tensor,
                labels:           torch.Tensor,
                delta:            torch.Tensor | None = None) -> dict:
        """
        Args:
            logits_clean:     (B, T, V) logits on clean images.
            logits_corrupted: (B, T, V) logits on the same images + corruption.
            labels:           (B, T) ground-truth token ids.
            delta:            (B,) optional uncertainty scores from EUE.
                              If provided, OCD loss is weighted by δ.

        Returns:
            dict with 'total', 'ce_loss', 'ocd_loss'.
        """
        B, T, V = logits_clean.shape

        # Standard cross-entropy on clean logits
        ce_loss = F.cross_entropy(
            logits_clean.view(-1, V),
            labels.view(-1),
            ignore_index=-100,
            reduction=self.reduction,
        )

        # Compute soft distributions
        p_clean     = F.softmax(logits_clean     / self.temperature, dim=-1)   # (B, T, V)
        p_corrupted = F.softmax(logits_corrupted / self.temperature, dim=-1)   # (B, T, V)

        # KL divergence: KL(P_clean || P_corrupted)
        eps     = 1e-8
        kl_div  = (p_clean * (torch.log(p_clean + eps) - torch.log(p_corrupted + eps))).sum(dim=-1)  # (B, T)

        # Mask padding tokens (labels == -100)
        pad_mask = (labels != -100).float()  # (B, T)
        kl_div   = kl_div * pad_mask

        if delta is not None:
            # Weight per-sample by uncertainty score δ
            kl_div = kl_div * delta.unsqueeze(1)

        ocd_loss = kl_div.sum() / (pad_mask.sum() + 1e-8) if self.reduction == "mean" else kl_div.sum()

        total_loss = ce_loss + self.alpha * ocd_loss

        return {
            "total":    total_loss,
            "ce_loss":  ce_loss,
            "ocd_loss": ocd_loss,
        }

    def __repr__(self):
        return (f"OODContrastiveDecodingLoss("
                f"alpha={self.alpha}, T={self.temperature})")
