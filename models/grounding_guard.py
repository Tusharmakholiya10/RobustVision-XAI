"""
models/grounding_guard.py — Grounding Guard Module
Activated when EUE detects OOD input. Uses SAM + DINOv2 to build
an object-presence prior that constrains LLM hallucination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingGuard(nn.Module):
    """
    Grounding Guard: Triggered for OOD inputs to inject a visual prior.

    In a full implementation this module wraps:
      - Segment Anything Model (SAM ViT-B) for mask generation
      - DINOv2 patch features for semantic patch scoring

    Here we provide the interface and a mock implementation
    suitable for testing without GPU resources.

    Args:
        embed_dim (int): Visual token embedding dimension.
        num_object_slots (int): Number of object concept slots.
        confidence_threshold (float): Min DINOv2 similarity to keep a region.
    """

    def __init__(self, embed_dim: int = 1024, num_object_slots: int = 16,
                 confidence_threshold: float = 0.35):
        super().__init__()
        self.embed_dim            = embed_dim
        self.num_object_slots     = num_object_slots
        self.confidence_threshold = confidence_threshold

        # Cross-attention: visual token Q, object slot K/V
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        # Object-presence confidence head
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_object_slots),
            nn.Sigmoid(),
        )
        # Upweighting projection applied to confident regions
        self.upweight_proj = nn.Linear(embed_dim, embed_dim)

    def build_object_prior(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Builds an object-presence prior from visual tokens.

        In production: visual_tokens come from SAM + DINOv2 features.
        Here: uses self-attention to identify salient token clusters.

        Args:
            visual_tokens: (B, N, D)

        Returns:
            object_slots: (B, num_object_slots, D) object-level representations
        """
        B, N, D = visual_tokens.shape
        # Cluster tokens into object slots via attention pooling
        slot_queries = visual_tokens[:, :self.num_object_slots, :]  # (B, S, D)
        object_slots, _ = self.cross_attn(
            slot_queries, visual_tokens, visual_tokens
        )
        return object_slots  # (B, S, D)

    def forward(self, visual_tokens: torch.Tensor,
                is_ood: torch.Tensor) -> dict:
        """
        Applies grounding guard only on OOD samples (is_ood mask).

        Args:
            visual_tokens: (B, N, D) from vision encoder.
            is_ood: (B,) boolean mask from EUE.

        Returns:
            dict:
              - 'guarded_tokens': (B, N, D) — upweighted visual tokens
              - 'object_priors':  (B, S, D) — object slot representations
              - 'confidence':     (B, S)    — per-slot confidence scores
        """
        object_slots = self.build_object_prior(visual_tokens)         # (B, S, D)
        confidence   = self.confidence_head(object_slots)             # (B, S)

        # Mask low-confidence slots
        conf_mask    = confidence > self.confidence_threshold          # (B, S)
        kept_slots   = object_slots * conf_mask.unsqueeze(-1).float()  # (B, S, D)

        # Cross-attend visual tokens to kept object slots
        attended, _ = self.cross_attn(visual_tokens, kept_slots, kept_slots)
        upweighted  = self.upweight_proj(attended)                     # (B, N, D)

        # Only replace tokens for OOD samples
        guard_mask     = is_ood[:, None, None].float()                 # (B, 1, 1)
        guarded_tokens = guard_mask * upweighted + (1 - guard_mask) * visual_tokens

        return {
            "guarded_tokens": guarded_tokens,
            "object_priors":  kept_slots,
            "confidence":     confidence,
        }
