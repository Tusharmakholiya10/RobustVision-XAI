"""
Visualization 5: Gap Analysis Summary — Bubble Chart
Maps each paper's contribution score vs. its remaining gap severity.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

PAPERS = [
    {"name": "ResNet\n2015",    "contribution": 9.2, "gap_severity": 7.1, "impact": 9.5, "era": "CNN",         "color": "#E74C3C"},
    {"name": "Mask R-CNN\n2017","contribution": 8.5, "gap_severity": 6.8, "impact": 8.8, "era": "CNN",         "color": "#E74C3C"},
    {"name": "SimCLR\n2020",    "contribution": 7.4, "gap_severity": 5.9, "impact": 7.2, "era": "SSL",         "color": "#F39C12"},
    {"name": "ViT\n2020",       "contribution": 9.4, "gap_severity": 7.5, "impact": 9.3, "era": "Transformer", "color": "#3498DB"},
    {"name": "NeRF\n2020",      "contribution": 8.8, "gap_severity": 7.9, "impact": 8.6, "era": "3D",          "color": "#27AE60"},
    {"name": "CLIP\n2021",      "contribution": 9.5, "gap_severity": 8.1, "impact": 9.6, "era": "VL",          "color": "#9B59B6"},
    {"name": "DINO\n2021",      "contribution": 8.2, "gap_severity": 6.4, "impact": 8.0, "era": "SSL",         "color": "#F39C12"},
    {"name": "SAM\n2023",       "contribution": 9.0, "gap_severity": 7.2, "impact": 8.9, "era": "Foundation",  "color": "#1ABC9C"},
    {"name": "3DGS\n2023",      "contribution": 8.7, "gap_severity": 7.6, "impact": 8.5, "era": "3D",          "color": "#27AE60"},
    {"name": "LLaVA\n2023",     "contribution": 9.1, "gap_severity": 8.8, "impact": 9.4, "era": "VLM",         "color": "#E91E8C"},
]

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#161B22")

# grid
for x in np.arange(5, 11, 1):
    ax.axvline(x, color="#21262D", linewidth=0.6, linestyle="--")
for y in np.arange(4, 10.5, 1):
    ax.axhline(y, color="#21262D", linewidth=0.6, linestyle="--")

# quadrant shading
ax.axvspan(5, 7.5, ymin=0, ymax=1,   alpha=0.03, color="#27AE60")  # low gap
ax.axvspan(7.5, 10, ymin=0, ymax=1,  alpha=0.05, color="#E74C3C")  # high gap
ax.axhspan(5, 8.5, xmin=0, xmax=1,  alpha=0.02, color="#3498DB")
ax.axhspan(8.5, 10, xmin=0, xmax=1, alpha=0.04, color="#F39C12")

ax.text(5.1, 9.5, "High Contribution\nLow Remaining Gap\n(Sweet Spot ✓)",
        fontsize=8, color="#27AE60", alpha=0.7, va="top")
ax.text(7.6, 9.5, "High Contribution\nHigh Remaining Gap\n(Target Zone for Research ⚠)",
        fontsize=8, color="#E74C3C", alpha=0.7, va="top")

# bubbles
for p in PAPERS:
    size = p["impact"] ** 2.5 * 18
    ax.scatter(p["gap_severity"], p["contribution"], s=size,
               color=p["color"], alpha=0.80, edgecolors="white", linewidth=1.2, zorder=4)
    ax.text(p["gap_severity"] + 0.08, p["contribution"] + 0.12, p["name"],
            fontsize=8.5, color="white", fontweight="bold",
            ha="left", va="bottom", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0D1117", alpha=0.5, edgecolor="none"))

# RobustVision-XAI project arrow
ax.annotate("🎯 RobustVision-XAI\naddresses this cluster",
            xy=(8.5, 8.9), xytext=(6.2, 8.0),
            arrowprops=dict(arrowstyle="-|>", color="#FFD700", lw=2.0, mutation_scale=18),
            fontsize=10, color="#FFD700", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0D1117", edgecolor="#FFD700", linewidth=1.5))

legend_patches = [
    mpatches.Patch(color="#E74C3C", label="CNN Era"),
    mpatches.Patch(color="#F39C12", label="Self-Supervised"),
    mpatches.Patch(color="#3498DB", label="Transformer Era"),
    mpatches.Patch(color="#27AE60", label="3D Vision"),
    mpatches.Patch(color="#9B59B6", label="Vision-Language"),
    mpatches.Patch(color="#1ABC9C", label="Foundation Models"),
    mpatches.Patch(color="#E91E8C", label="Multimodal LLM"),
]
legend = ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
                   facecolor="#0D1117", edgecolor="#30363D", labelcolor="white",
                   title="Research Era", title_fontsize=9)
legend.get_title().set_color("#8B949E")

ax.set_xlim(5, 10)
ax.set_ylim(5, 10)
ax.set_xlabel("Remaining Gap Severity (→ Higher = More Unresolved)", color="#8B949E",
              fontsize=11, labelpad=10)
ax.set_ylabel("Contribution Score (↑ Higher = More Impactful)", color="#8B949E",
              fontsize=11, labelpad=10)
ax.tick_params(colors="#30363D", length=0)
ax.set_xticklabels([f"{v:.0f}" for v in ax.get_xticks()], color="#8B949E")
ax.set_yticklabels([f"{v:.0f}" for v in ax.get_yticks()], color="#8B949E")
for spine in ax.spines.values():
    spine.set_color("#30363D")

ax.set_title("Contribution vs Remaining Research Gap — 10 Landmark CV Papers\n(Bubble size = real-world impact)",
             fontsize=14, fontweight="bold", color="white", pad=16)

fig.text(0.5, 0.01,
         "Papers in the top-right quadrant have the highest contribution AND the most unresolved gaps → prime research opportunity",
         ha="center", fontsize=8.5, color="#8B949E", style="italic")

out = os.path.join(os.path.dirname(__file__), "..", "assets", "5_gap_bubble_chart.png")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {os.path.abspath(out)}")
