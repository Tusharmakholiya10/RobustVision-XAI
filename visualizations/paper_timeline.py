"""
Visualization 1: Computer Vision Paper Timeline (2015–2025)
Plots the evolution of landmark CV papers on a styled timeline.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

PAPERS = [
    {"year": 2015.9, "name": "ResNet",      "venue": "CVPR 2016",     "era": "CNN Era",         "color": "#E74C3C",
     "desc": "Skip connections\nenabling 150+ layer\ndeep networks"},
    {"year": 2017.6, "name": "Mask R-CNN",  "venue": "ICCV 2017",     "era": "CNN Era",         "color": "#E74C3C",
     "desc": "Unified detection\n+ pixel-level\nsegmentation"},
    {"year": 2020.2, "name": "SimCLR",      "venue": "ICML 2020",     "era": "Self-Supervised", "color": "#F39C12",
     "desc": "Contrastive SSL\nmatching supervised\nquality"},
    {"year": 2020.7, "name": "ViT",         "venue": "ICLR 2021",     "era": "Transformer Era", "color": "#3498DB",
     "desc": "Pure Transformers\nfor image patches\n16x16 words"},
    {"year": 2020.8, "name": "NeRF",        "venue": "ECCV 2020",     "era": "3D Vision",       "color": "#27AE60",
     "desc": "Neural Radiance\nFields for 3D\nscene rendering"},
    {"year": 2021.3, "name": "CLIP",        "venue": "ICML 2021",     "era": "Vision-Language", "color": "#9B59B6",
     "desc": "Contrastive image-\ntext pre-training\nfor zero-shot tasks"},
    {"year": 2021.6, "name": "DINO",        "venue": "ICCV 2021",     "era": "Self-Supervised", "color": "#F39C12",
     "desc": "Self-supervised\ndistillation with\nsemantic features"},
    {"year": 2023.3, "name": "SAM",         "venue": "ICCV 2023",     "era": "Foundation",      "color": "#1ABC9C",
     "desc": "Promptable\nuniversal zero-shot\nsegmentation"},
    {"year": 2023.6, "name": "3DGS",        "venue": "SIGGRAPH 2023", "era": "3D Vision",       "color": "#27AE60",
     "desc": "Real-time 3D\nGaussian splatting\nfor rendering"},
    {"year": 2023.9, "name": "LLaVA",       "venue": "NeurIPS 2023",  "era": "Multimodal LLM",  "color": "#E91E8C",
     "desc": "Visual instruction\ntuning: VLM via\nLLM alignment"},
]

ERA_COLORS = {
    "CNN Era":         "#E74C3C",
    "Self-Supervised": "#F39C12",
    "Transformer Era": "#3498DB",
    "3D Vision":       "#27AE60",
    "Vision-Language": "#9B59B6",
    "Foundation":      "#1ABC9C",
    "Multimodal LLM":  "#E91E8C",
}

fig, ax = plt.subplots(figsize=(22, 10))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")

ax.axhline(y=0, color="#30363D", linewidth=3, zorder=1)

for x in np.arange(2015, 2026, 1):
    ax.axvline(x=x, color="#21262D", linewidth=0.5, linestyle="--", alpha=0.5)

for i, p in enumerate(PAPERS):
    offset = 1.6 if i % 2 == 0 else -1.8
    color = p["color"]

    ax.plot([p["year"], p["year"]], [0, offset * 0.6], color=color, linewidth=2, zorder=2, alpha=0.8)
    ax.scatter([p["year"]], [0], s=120, color=color, zorder=5, edgecolors="white", linewidth=1.5)

    bbox_props = dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.15,
                      edgecolor=color, linewidth=1.5)
    sign = 1 if offset > 0 else -1
    ax.text(p["year"], offset * 0.75, p["name"], fontsize=11, fontweight="bold",
            color="white", ha="center", va="bottom" if offset > 0 else "top",
            bbox=bbox_props, zorder=6)

    ax.text(p["year"], offset * 0.75 + sign * 0.28, p["venue"], fontsize=7.5,
            color="#8B949E", ha="center", va="bottom" if offset > 0 else "top", zorder=6)

    ax.text(p["year"], offset * 0.75 + sign * 0.60, p["desc"], fontsize=6.8,
            color="#C9D1D9", ha="center", va="bottom" if offset > 0 else "top",
            linespacing=1.4, zorder=6)

legend_patches = [mpatches.Patch(color=c, label=e, alpha=0.85) for e, c in ERA_COLORS.items()]
legend = ax.legend(handles=legend_patches, loc="upper left", fontsize=9,
                   facecolor="#161B22", edgecolor="#30363D",
                   labelcolor="white", framealpha=0.95,
                   title="Research Eras", title_fontsize=9)
legend.get_title().set_color("#8B949E")

ax.set_xlim(2014.8, 2025.2)
ax.set_ylim(-2.5, 2.5)
ax.set_xticks(range(2015, 2026))
ax.set_xticklabels([str(y) for y in range(2015, 2026)], color="#8B949E", fontsize=10)
ax.set_yticks([])
ax.spines[["top","right","left","bottom"]].set_visible(False)

ax.set_title("Computer Vision Landmark Papers: 2015–2025", fontsize=18,
             fontweight="bold", color="white", pad=20)
ax.text(0.5, -0.08, "A decade of breakthroughs from ResNet to LLaVA · RobustVision-XAI Research Project",
        transform=ax.transAxes, ha="center", fontsize=10, color="#8B949E", style="italic")

out = os.path.join(os.path.dirname(__file__), "..", "assets", "1_paper_timeline.png")
plt.tight_layout()
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {os.path.abspath(out)}")
