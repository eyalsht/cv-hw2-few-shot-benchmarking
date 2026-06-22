"""Generate a dark, modern animated hero GIF from the benchmark results.

Shows mean few-shot accuracy vs. number of shots (K) for the three backbones
(Ridge head) on each dataset side by side -- visualizing the headline finding
that the backbone ranking flips between natural images (Cars) and satellite
imagery (EuroSAT). Lines draw in with a moving marker; loops smoothly.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

df = pd.read_csv("benchmark_results.csv")
HEAD = "ridge"  # best head overall
DATASETS = [("cars", "Stanford Cars  ·  natural images"),
            ("eurosat", "EuroSAT  ·  satellite imagery")]
BACKBONES = ["dinov2", "clip", "convnext"]
LABELS = {"dinov2": "DINOv2-s", "clip": "CLIP-RN50", "convnext": "ConvNeXt-T"}
COLORS = {"dinov2": "#22d3ee", "clip": "#f472b6", "convnext": "#a3e635"}  # cyan / pink / lime
K = [1, 2, 4, 16]
logK = np.log2(K)

BG = "#0d1117"      # github dark
FG = "#c9d1d9"
GRID = "#21262d"

# dense interpolation grid for smooth line drawing
GRID_N = 220
xg = np.linspace(logK[0], logK[-1], GRID_N)

# precompute per (dataset, backbone) interpolated curve
curves = {}
for ds, _ in DATASETS:
    for bb in BACKBONES:
        sub = df[(df.dataset == ds) & (df.backbone == bb) & (df.classifier == HEAD)].sort_values("k")
        y = sub["mean"].to_numpy()
        curves[(ds, bb)] = np.interp(xg, logK, y)

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "font.family": "DejaVu Sans",
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=110)
fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.13, wspace=0.13)

fig.suptitle("Few-Shot Accuracy vs. Shots  —  the backbone ranking flips across domains",
             fontsize=13, fontweight="bold", color="#ffffff", y=0.965)
fig.text(0.5, 0.875, "3 frozen foundation backbones · Ridge head · 5-way · 10,000 episodes / point",
         ha="center", fontsize=8.5, color="#8b949e")

N_HOLD = 26
N_DRAW = GRID_N
TOTAL = N_DRAW + N_HOLD

def setup_ax(ax, title):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.7)
    ax.set_xlim(logK[0] - 0.15, logK[-1] + 1.25)  # room for end-labels
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(logK)
    ax.set_xticklabels([f"{k}" for k in K])
    ax.set_xlabel("shots per class (K)", fontsize=9)
    ax.set_title(title, fontsize=10.5, color="#ffffff", pad=8)

def draw(frame):
    prog = min(frame, N_DRAW)
    n = max(int(prog), 2)
    for ax, (ds, title) in zip(axes, DATASETS):
        ax.clear()
        setup_ax(ax, title)
        ends = []
        for bb in BACKBONES:
            yc = curves[(ds, bb)]
            ax.plot(xg[:n], yc[:n], color=COLORS[bb], lw=2.6, solid_capstyle="round",
                    alpha=0.95, zorder=3)
            ax.plot(xg[:n], yc[:n], color=COLORS[bb], lw=6, alpha=0.12, zorder=2)  # glow
            ax.scatter([xg[n - 1]], [yc[n - 1]], s=42, color=COLORS[bb],
                       edgecolor=BG, linewidth=1.4, zorder=4)
            ends.append([yc[n - 1], bb])
        # de-overlap the end labels vertically (min gap in data units)
        ends.sort()
        min_gap = 0.038
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < min_gap:
                ends[i][0] = ends[i - 1][0] + min_gap
        xlbl = xg[n - 1] + 0.18
        for y_lbl, bb in ends:
            ax.text(xlbl, y_lbl, f"{LABELS[bb]} · {curves[(ds, bb)][n-1]*100:.1f}%",
                    color=COLORS[bb], fontsize=8.2, va="center", fontweight="bold", zorder=5)
    axes[0].set_ylabel("mean accuracy", fontsize=9)
    return []

anim = FuncAnimation(fig, draw, frames=TOTAL, interval=33, blit=False)
anim.save("assets/scaling.gif", writer=PillowWriter(fps=30))
print("wrote assets/scaling.gif")
