"""
Visualization 2: Research Gap Heatmap
Shows which research gaps each of the 10 papers leaves behind.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

PAPERS = [
    "ResNet\n(2015)", "Mask R-CNN\n(2017)", "SimCLR\n(2020)",
    "ViT\n(2020)", "NeRF\n(2020)", "CLIP\n(2021)",
    "DINO\n(2021)", "SAM\n(2023)", "3DGS\n(2023)", "LLaVA\n(2023)"
]

GAPS = [
    "OOD\nRobustness",
    "Hallucination\n/ Grounding",
    "Interpretability\n(XAI)",
    "Data\nEfficiency",
    "Spatial /\nTemporal",
    "Edge\nDeployment",
    "Open-World\nPerception",
    "Physical\nReasoning",
]

# Severity of gap: 0=not applicable, 1=minor, 2=moderate, 3=severe
DATA = np.array([
    # OOD  Hall  XAI  Data  ST   Edge  OW   Phys
    [3,    1,    3,    1,   1,    2,    2,    1],   # ResNet
    [2,    1,    2,    3,   3,    2,    3,    1],   # Mask R-CNN
    [2,    0,    1,    3,   1,    2,    1,    0],   # SimCLR
    [2,    1,    2,    3,   1,    3,    2,    1],   # ViT
    [1,    0,    1,    2,   3,    3,    1,    2],   # NeRF
    [2,    3,    2,    1,   2,    2,    2,    1],   # CLIP
    [2,    2,    2,    1,   1,    3,    2,    0],   # DINO
    [1,    2,    1,    2,   2,    3,    3,    1],   # SAM
    [1,    2,    1,    2,   3,    3,    2,    3],   # 3DGS
    [3,    3,    3,    2,   2,    2,    2,    2],   # LLaVA
], dtype=float)

fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")

cmap = mcolors.LinearSegmentedColormap.from_list(
    "gap_cmap", ["#161B22", "#F39C12", "#E74C3C"], N=256
)

im = ax.imshow(DATA, cmap=cmap, aspect="auto", vmin=0, vmax=3)

LABELS = {0: "", 1: "Minor", 2: "Moderate", 3: "Severe"}
LABEL_COLORS = {0: "#21262D", 1: "#F0E68C", 2: "#F39C12", 3: "#E74C3C"}

for i in range(len(PAPERS)):
    for j in range(len(GAPS)):
        val = int(DATA[i, j])
        label = LABELS[val]
        if label:
            text_color = "white" if val >= 2 else "#C9D1D9"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=8, fontweight="bold" if val == 3 else "normal",
                    color=text_color)

ax.set_xticks(range(len(GAPS)))
ax.set_yticks(range(len(PAPERS)))
ax.set_xticklabels(GAPS, color="#C9D1D9", fontsize=9.5, linespacing=1.3)
ax.set_yticklabels(PAPERS, color="#C9D1D9", fontsize=9.5, linespacing=1.3)
ax.tick_params(colors="#30363D", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

for i in range(len(PAPERS) + 1):
    ax.axhline(i - 0.5, color="#21262D", linewidth=0.8)
for j in range(len(GAPS) + 1):
    ax.axvline(j - 0.5, color="#21262D", linewidth=0.8)

cbar = plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.025, pad=0.02)
cbar.set_ticks([0, 1, 2, 3])
cbar.set_ticklabels(["N/A", "Minor", "Moderate", "Severe"], color="#C9D1D9")
cbar.ax.yaxis.set_tick_params(color="#C9D1D9")
cbar.outline.set_visible(False)
cbar.ax.set_facecolor("#0D1117")

ax.set_title("Research Gap Severity Heatmap — 10 Landmark CV Papers (2015–2025)",
             fontsize=15, fontweight="bold", color="white", pad=18)
ax.set_xlabel("Identified Research Gaps", color="#8B949E", fontsize=10, labelpad=10)
ax.set_ylabel("Landmark Papers", color="#8B949E", fontsize=10, labelpad=10)

fig.text(0.5, 0.01,
         "Red = Severe unresolved gap  ·  Orange = Moderate gap  ·  Yellow = Minor gap  ·  RobustVision-XAI targets the top-right cluster",
         ha="center", fontsize=8.5, color="#8B949E", style="italic")

out = os.path.join(os.path.dirname(__file__), "..", "assets", "2_gap_heatmap.png")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {os.path.abspath(out)}")
