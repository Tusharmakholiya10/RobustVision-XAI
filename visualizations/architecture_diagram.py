"""
Visualization 3: RobustVision-XAI Architecture Diagram
Renders the full pipeline as a styled block diagram.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os


def draw_box(ax, x, y, w, h, label, sublabel="", color="#1F6FEB", text_color="white",
             fontsize=11, sublabel_size=8):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.04",
                         facecolor=color, edgecolor="white",
                         linewidth=1.4, alpha=0.92, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + (0.07 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.15, sublabel, ha="center", va="center",
                fontsize=sublabel_size, color="#C9D1D9", zorder=4,
                style="italic")


def arrow(ax, x1, y1, x2, y2, label="", color="white"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.6, mutation_scale=16),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.05, my, label, fontsize=7.5, color="#8B949E",
                ha="left", va="center", style="italic", zorder=5)


fig, ax = plt.subplots(figsize=(18, 11))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-1.0, 8.5)
ax.axis("off")

# ── Input ──────────────────────────────────────────────────────────────────────
draw_box(ax, 1.5, 7.5, 2.4, 0.8, "📷 Input Image",
         "Clean or OOD (fog/noise/thermal)", color="#21262D", fontsize=10)

# ── Vision Encoder ─────────────────────────────────────────────────────────────
draw_box(ax, 1.5, 5.8, 2.6, 1.0, "Vision Encoder",
         "DINOv2-L / CLIP-L/14", color="#1F6FEB", fontsize=11)
arrow(ax, 1.5, 7.1, 1.5, 6.3)

# ── EUE ────────────────────────────────────────────────────────────────────────
draw_box(ax, 1.5, 4.0, 2.8, 1.0, "🔍 Uncertainty Estimator",
         "Epistemic Uncertainty (EUE)\nMC Dropout + Deep Ensemble", color="#6E40C9", fontsize=10)
arrow(ax, 1.5, 5.3, 1.5, 4.5)

# ── Decision diamond ──────────────────────────────────────────────────────────
diamond_x, diamond_y = 1.5, 2.6
dx = mpatches.FancyArrowPatch
pts = [(diamond_x, diamond_y+0.45), (diamond_x+0.6, diamond_y),
       (diamond_x, diamond_y-0.45), (diamond_x-0.6, diamond_y)]
diamond = plt.Polygon(pts, closed=True, facecolor="#30363D",
                      edgecolor="#E74C3C", linewidth=2, zorder=3)
ax.add_patch(diamond)
ax.text(diamond_x, diamond_y, "δ > τ?", ha="center", va="center",
        fontsize=10, fontweight="bold", color="white", zorder=4)
arrow(ax, 1.5, 3.5, 1.5, 3.1)

# ── Grounding Guard (OOD path) ─────────────────────────────────────────────────
draw_box(ax, 4.5, 2.6, 2.8, 1.0, "🛡️ Grounding Guard",
         "SAM + DINOv2\nObject-Presence Prior Map", color="#E74C3C", fontsize=10)
ax.annotate("", xy=(3.3, 2.6), xytext=(2.1, 2.6),
            arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=1.6, mutation_scale=16), zorder=2)
ax.text(2.6, 2.75, "OOD", fontsize=8, color="#E74C3C", ha="center", fontweight="bold")

# ── Upweighter ─────────────────────────────────────────────────────────────────
draw_box(ax, 4.5, 1.1, 2.8, 0.9, "⚡ Cross-Attn Upweighter",
         "Visual Token Confidence Boost", color="#B45309", fontsize=10)
arrow(ax, 4.5, 2.1, 4.5, 1.6)

# ── LLM Decoder ────────────────────────────────────────────────────────────────
draw_box(ax, 7.5, 2.6, 2.6, 1.0, "🧠 LLM Decoder",
         "Vicuna-7B / LLaMA-3-8B\n+ OCD-Loss Fine-tuning", color="#0A7D6E", fontsize=10)
# in-distribution path
ax.annotate("", xy=(6.2, 2.6), xytext=(2.1, 2.6),
            arrowprops=dict(arrowstyle="-|>", color="#27AE60", lw=1.6,
                            mutation_scale=16,
                            connectionstyle="arc3,rad=-0.35"), zorder=2)
ax.text(4.5, 1.9, "In-dist", fontsize=8, color="#27AE60", ha="center", fontweight="bold")
# OOD path merges
arrow(ax, 5.9, 1.1, 7.5, 2.1)

# ── XAI Layer ──────────────────────────────────────────────────────────────────
draw_box(ax, 7.5, 0.8, 2.6, 0.9, "📊 XAI Attribution Layer",
         "Grad-CAM + Attention Rollout\nSaliency Heatmaps", color="#6E40C9", fontsize=10)
arrow(ax, 7.5, 2.1, 7.5, 1.3)

# ── Output ─────────────────────────────────────────────────────────────────────
draw_box(ax, 7.5, -0.4, 3.0, 0.9, "✅ Grounded Output",
         "Answer + Confidence Score + Heatmap", color="#1A7F37", fontsize=10)
arrow(ax, 7.5, 0.35, 7.5, -0.05)

# Legend boxes
legend_items = [
    ("#1F6FEB", "Vision Backbone"),
    ("#6E40C9", "Uncertainty / XAI"),
    ("#E74C3C", "Grounding Guard"),
    ("#0A7D6E", "Language Decoder"),
    ("#1A7F37", "Output"),
]
for idx, (c, lbl) in enumerate(legend_items):
    bx = FancyBboxPatch((0.05, 7.5 - idx*0.5), 0.25, 0.3,
                        boxstyle="round,pad=0.02",
                        facecolor=c, edgecolor="none", alpha=0.9, zorder=5)
    ax.add_patch(bx)
    ax.text(0.38, 7.65 - idx*0.5, lbl, fontsize=8, color="#C9D1D9",
            va="center", zorder=6)

ax.set_title("RobustVision-XAI — System Architecture Pipeline",
             fontsize=17, fontweight="bold", color="white",
             pad=16, loc="center")

out = os.path.join(os.path.dirname(__file__), "..", "assets", "3_architecture.png")
plt.tight_layout()
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {os.path.abspath(out)}")
