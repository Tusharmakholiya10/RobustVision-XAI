"""
models/xai_layer.py — XAI Attribution Layer
Generates Grad-CAM and Attention Rollout saliency maps for VLM predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GradCAMLayer(nn.Module):
    """
    Gradient-weighted Class Activation Mapping for vision encoders.

    Hooks into a target layer to compute saliency maps for the
    predicted token sequence given input visual features.

    Args:
        model: Vision encoder (e.g., DINOv2 or CLIP visual).
        target_layer_name: Layer to hook (e.g., 'blocks.11.attn').
    """

    def __init__(self, model: nn.Module, target_layer_name: str):
        super().__init__()
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hook_layer(target_layer_name)

    def _hook_layer(self, name: str):
        layer = dict(self.model.named_modules())[name]

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def compute_saliency(self, visual_features: torch.Tensor,
                          logits: torch.Tensor,
                          target_token_idx: int = 0) -> torch.Tensor:
        """
        Compute GradCAM saliency for a specific output token.

        Args:
            visual_features: (B, N, D) vision encoder output.
            logits: (B, T, V) LLM decoder logits.
            target_token_idx: Which output token to attribute.

        Returns:
            saliency: (B, sqrt(N), sqrt(N)) spatial saliency map.
        """
        # Backprop score for target token
        score = logits[:, target_token_idx, :].max(dim=-1).values
        score.sum().backward(retain_graph=True)

        grads = self.gradients   # (B, N, D)
        acts  = self.activations # (B, N, D)

        # Weight activations by gradient magnitudes
        weights  = grads.mean(dim=-1, keepdim=True)  # (B, N, 1)
        cam      = (weights * acts).sum(dim=-1)       # (B, N)
        cam      = F.relu(cam)

        # Reshape to spatial grid (assume square patch grid)
        N    = cam.shape[-1]
        H    = W = int(N ** 0.5)
        cam  = cam.view(-1, H, W)                     # (B, H, W)
        cam  = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


class AttentionRollout:
    """
    Attention Rollout for ViT-style encoders.
    Propagates attention across all layers to identify attended patches.

    Reference: Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers"
    """

    def __init__(self, head_fusion: str = "mean", discard_ratio: float = 0.9):
        self.head_fusion   = head_fusion
        self.discard_ratio = discard_ratio

    def __call__(self, attention_maps: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            attention_maps: List of (B, H, N, N) attention matrices per layer.

        Returns:
            rollout: (B, N, N) joint attention across all layers.
        """
        result = torch.eye(attention_maps[0].shape[-1]).unsqueeze(0)
        result = result.to(attention_maps[0].device)

        for attn in attention_maps:
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)   # (B, N, N)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1).values
            else:
                raise ValueError(f"Unknown head_fusion: {self.head_fusion}")

            # Discard lowest attention scores (noise)
            flat    = attn_fused.view(attn_fused.shape[0], -1)
            thresh  = torch.quantile(flat, self.discard_ratio, dim=-1, keepdim=True)
            thresh  = thresh.unsqueeze(-1)
            attn_fused = torch.where(attn_fused >= thresh,
                                     attn_fused,
                                     torch.zeros_like(attn_fused))

            # Add residual connection
            attn_fused = attn_fused + torch.eye(attn_fused.shape[-1], device=attn_fused.device)
            attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

            result = torch.bmm(attn_fused, result)

        return result  # (B, N, N)


class XAIAttributionLayer(nn.Module):
    """
    Unified XAI attribution layer combining GradCAM and Attention Rollout.

    Outputs a composite saliency map that flags when the predicted token
    is poorly grounded in the image (potential hallucination indicator).

    Args:
        hallucination_threshold (float): Saliency score below which we
            flag the prediction as potentially hallucinated.
    """

    def __init__(self, hallucination_threshold: float = 0.25):
        super().__init__()
        self.hallucination_threshold = hallucination_threshold
        self.rollout = AttentionRollout(head_fusion="mean", discard_ratio=0.85)

    def forward(self,
                gradcam_map:        torch.Tensor,
                rollout_attentions: list[torch.Tensor],
                predicted_tokens:   torch.Tensor) -> dict:
        """
        Returns a composite saliency map and hallucination flags.

        Args:
            gradcam_map: (B, H, W) from GradCAMLayer.
            rollout_attentions: list of (B, nH, N, N) attention matrices.
            predicted_tokens: (B, T) predicted token ids.

        Returns:
            dict:
              - 'composite_map': (B, H, W) normalized composite saliency
              - 'hallucination_flags': (B,) bool — True if likely hallucinated
              - 'grounding_scores': (B,) float in [0, 1]
        """
        rollout    = self.rollout(rollout_attentions)   # (B, N, N)
        cls_attn   = rollout[:, 0, 1:]                  # (B, N-1) cls-to-patch
        H = W      = int(cls_attn.shape[-1] ** 0.5)
        attn_map   = cls_attn.view(-1, H, W)            # (B, H, W)
        attn_map   = F.interpolate(
            attn_map.unsqueeze(1),
            size=gradcam_map.shape[-2:],
            mode="bilinear", align_corners=False,
        ).squeeze(1)                                     # (B, H, W)

        # Normalize both maps
        def _norm(t):
            mn, mx = t.flatten(1).min(1)[0], t.flatten(1).max(1)[0]
            return (t - mn[:, None, None]) / (mx[:, None, None] - mn[:, None, None] + 1e-8)

        gc_norm    = _norm(gradcam_map)
        attn_norm  = _norm(attn_map)

        composite  = 0.6 * gc_norm + 0.4 * attn_norm   # (B, H, W)

        grounding_scores      = composite.flatten(1).mean(1)  # (B,)
        hallucination_flags   = grounding_scores < self.hallucination_threshold

        return {
            "composite_map":      composite,
            "hallucination_flags": hallucination_flags,
            "grounding_scores":   grounding_scores,
        }
