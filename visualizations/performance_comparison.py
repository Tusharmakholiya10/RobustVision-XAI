"""
Visualization 4: Performance Comparison Charts
Baseline LLaVA-1.5 vs RobustVision-XAI across all evaluation metrics.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor("#0D1117")

BASELINE_COLOR = "#E74C3C"
PROPOSED_COLOR = "#1F6FEB"
BG             = "#161B22"
TEXT           = "#C9D1D9"
GRID           = "#21262D"

# ── Chart 1: Bar chart – Key metrics ──────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor(BG)
metrics     = ["POPE F1↑\n(hallucinat.)", "OOD mAcc↑\n(robustness)", "Explanation\nFaithfulness↑"]
baseline    = [85.9, 42.1, 51.3]
proposed    = [90.2, 58.7, 72.6]
x           = np.arange(len(metrics))
w           = 0.32

b1 = ax1.bar(x - w/2, baseline, w, color=BASELINE_COLOR, alpha=0.85, label="LLaVA-1.5 Baseline")
b2 = ax1.bar(x + w/2, proposed, w, color=PROPOSED_COLOR, alpha=0.85, label="RobustVision-XAI")

for bar, val in zip(b1, baseline):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f"{val}", ha="center", fontsize=8.5, color=BASELINE_COLOR, fontweight="bold")
for bar, val in zip(b2, proposed):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f"{val}", ha="center", fontsize=8.5, color=PROPOSED_COLOR, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(metrics, color=TEXT, fontsize=9.5)
ax1.set_yticks(range(0, 101, 20))
ax1.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], color=TEXT, fontsize=9)
ax1.set_ylim(0, 105)
ax1.yaxis.grid(True, color=GRID, linewidth=0.7)
ax1.set_axisbelow(True)
ax1.spines[["top","right","left","bottom"]].set_color(GRID)
ax1.set_title("Performance Metrics ↑", color="white", fontsize=12, fontweight="bold", pad=10)
ax1.legend(fontsize=8, facecolor="#0D1117", edgecolor=GRID, labelcolor=TEXT)

# ── Chart 2: Line – CHAIR score under OOD severity ────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(BG)
severities = [0, 1, 2, 3, 4, 5]
chair_base = [14.2, 18.7, 25.3, 33.1, 42.8, 56.4]
chair_prop = [9.8,  11.2, 13.9, 17.4, 22.1, 29.8]

ax2.plot(severities, chair_base, "o-", color=BASELINE_COLOR, linewidth=2.2,
         markersize=7, label="LLaVA-1.5 Baseline", markerfacecolor="white",
         markeredgewidth=2, markeredgecolor=BASELINE_COLOR)
ax2.plot(severities, chair_prop, "s-", color=PROPOSED_COLOR, linewidth=2.2,
         markersize=7, label="RobustVision-XAI", markerfacecolor="white",
         markeredgewidth=2, markeredgecolor=PROPOSED_COLOR)

ax2.fill_between(severities, chair_base, chair_prop,
                 color=PROPOSED_COLOR, alpha=0.12)
ax2.set_xticks(severities)
ax2.set_xticklabels([f"S{s}" for s in severities], color=TEXT, fontsize=9)
ax2.set_yticks(range(0, 65, 10))
ax2.set_yticklabels([f"{v}%" for v in range(0, 65, 10)], color=TEXT, fontsize=9)
ax2.yaxis.grid(True, color=GRID, linewidth=0.7)
ax2.xaxis.grid(True, color=GRID, linewidth=0.7)
ax2.set_axisbelow(True)
ax2.spines[["top","right","left","bottom"]].set_color(GRID)
ax2.set_title("CHAIR↓ Hallucination vs OOD Severity", color="white",
              fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("OOD Severity (S0=clean → S5=extreme)", color=TEXT, fontsize=9)
ax2.set_ylabel("CHAIR Score (↓ is better)", color=TEXT, fontsize=9)
ax2.legend(fontsize=8, facecolor="#0D1117", edgecolor=GRID, labelcolor=TEXT)

ax2.annotate("⬇ 47% reduction\nat severity 5",
             xy=(5, 29.8), xytext=(3.4, 40),
             arrowprops=dict(arrowstyle="->", color=PROPOSED_COLOR, lw=1.5),
             fontsize=8, color=PROPOSED_COLOR, fontweight="bold")

# ── Chart 3: Radar – Multi-axis capabilities ────────────────────────────────
ax3 = axes[2]
ax3.set_facecolor(BG)
ax3.remove()
ax3 = fig.add_axes([0.70, 0.08, 0.28, 0.82], polar=True)
ax3.set_facecolor(BG)

categories = ["OOD\nRobustness", "Hallucination\nControl", "XAI\nQuality",
              "Data\nEfficiency", "Speed /\nLatency", "Spatial\nReasoning"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

baseline_vals = [0.42, 0.56, 0.38, 0.65, 0.78, 0.44]
proposed_vals = [0.72, 0.82, 0.76, 0.68, 0.70, 0.63]
baseline_vals += baseline_vals[:1]
proposed_vals += proposed_vals[:1]

ax3.set_theta_offset(np.pi / 2)
ax3.set_theta_direction(-1)
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(categories, color=TEXT, fontsize=8.5, linespacing=1.3)
ax3.set_ylim(0, 1)
ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
ax3.set_yticklabels(["25%", "50%", "75%", "100%"], color="#8B949E", fontsize=7)
ax3.yaxis.grid(True, color=GRID, linewidth=0.7)
ax3.xaxis.grid(True, color=GRID, linewidth=0.7)
ax3.spines["polar"].set_color(GRID)

ax3.plot(angles, baseline_vals, "o-", color=BASELINE_COLOR, linewidth=2, markersize=5)
ax3.fill(angles, baseline_vals, color=BASELINE_COLOR, alpha=0.15)
ax3.plot(angles, proposed_vals, "s-", color=PROPOSED_COLOR, linewidth=2, markersize=5)
ax3.fill(angles, proposed_vals, color=PROPOSED_COLOR, alpha=0.20)

ax3.set_title("Capability Radar", color="white", fontsize=12,
              fontweight="bold", pad=22)

fig.suptitle("RobustVision-XAI vs LLaVA-1.5 Baseline — Performance Analysis",
             fontsize=16, fontweight="bold", color="white", y=0.97)
fig.text(0.5, 0.01,
         "* Projected metrics based on ablation analysis. OOD severity S0=clean, S5=extreme corruption.",
         ha="center", fontsize=8, color="#8B949E", style="italic")

out = os.path.join(os.path.dirname(__file__), "..", "assets", "4_performance_comparison.png")
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {os.path.abspath(out)}")
