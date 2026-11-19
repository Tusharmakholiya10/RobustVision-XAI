"""
models/eue.py — Epistemic Uncertainty Estimator (EUE)
Estimates uncertainty at the cross-modal bottleneck of a VLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpistemicUncertaintyEstimator(nn.Module):
    """
    Lightweight uncertainty estimator operating on visual token embeddings.

    Triggered after the vision encoder; outputs a distribution-shift score δ
    in [0, 1]. When δ > threshold, the Grounding Guard is activated.

    Args:
        embed_dim (int): Dimension of vision token embeddings.
        num_heads (int): Multi-head attention heads.
        dropout_rate (float): MC Dropout rate for uncertainty sampling.
        n_samples (int): Number of forward passes for MC Dropout estimation.
    """

    def __init__(self, embed_dim: int = 1024, num_heads: int = 8,
                 dropout_rate: float = 0.15, n_samples: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_samples = n_samples

        # Cross-modal bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Uncertainty head: predicts mean and log-variance
        self.mu_head     = nn.Linear(embed_dim // 4, 1)
        self.logvar_head = nn.Linear(embed_dim // 4, 1)

        # Multi-head self-attention over tokens
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, visual_tokens: torch.Tensor,
                threshold: float = 0.45) -> dict:
        """
        Args:
            visual_tokens: (B, N, D) — batch of N visual token embeddings.
            threshold: δ above which OOD is flagged.

        Returns:
            dict with keys:
              - 'delta'   : (B,) uncertainty score per sample
              - 'is_ood'  : (B,) bool mask
              - 'mu'      : (B,) predicted mean
              - 'logvar'  : (B,) predicted log-variance
        """
        # Attend over token sequence
        attn_out, _ = self.attn(visual_tokens, visual_tokens, visual_tokens)
        x = attn_out.mean(dim=1)  # (B, D) global pooled representation

        # MC Dropout: run n_samples passes
        self.train()  # enable dropout for uncertainty sampling
        samples = []
        for _ in range(self.n_samples):
            z  = self.bottleneck(x)
            mu = self.mu_head(z).squeeze(-1)
            samples.append(mu)
        self.eval()

        samples = torch.stack(samples, dim=0)  # (n_samples, B)
        mean    = samples.mean(dim=0)
        var     = samples.var(dim=0)

        # Normalize variance to [0, 1] using sigmoid
        delta   = torch.sigmoid(var * 5.0)
        is_ood  = delta > threshold

        z       = self.bottleneck(x)
        logvar  = self.logvar_head(z).squeeze(-1)

        return {
            "delta":  delta,
            "is_ood": is_ood,
            "mu":     mean,
            "logvar": logvar,
        }

    def compute_calibration_loss(self, delta: torch.Tensor,
                                  ood_labels: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy calibration loss for supervised uncertainty training."""
        return F.binary_cross_entropy(delta, ood_labels.float())
