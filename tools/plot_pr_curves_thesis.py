"""
Precision-recall figures for the thesis, from outputs/pr_curves/*.csv.
Writes outputs/figures/{pr_voxelwise,pr_lesionwise}.{pdf,png} (PDF is vector,
for \\includegraphics).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Polish decimal separator on all axes
PL = FuncFormatter(lambda v, _pos: f"{v:g}".replace(".", ","))

IN_DIR = "outputs/pr_curves"
OUT_DIR = "outputs/figures"

REGIONS = [("WT", "WT (Whole Tumor)"),
           ("TC", "TC (Tumor Core)"),
           ("ET", "ET (Enhancing Tumor)")]

# fixed order; palette validated for colorblind safety (Okabe-Ito); line style
# duplicates color so the figure still reads in black-and-white print
MODELS = [
    ("base",      "Swin-UNETR Base",      "#0072B2", "-"),
    ("detach",    "Multi-task (Detach)",  "#009E73", "--"),
    ("multitask", "Multi-task (standard)", "#D55E00", "-."),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def style_axes(ax):
    ax.xaxis.set_major_formatter(PL)
    ax.yaxis.set_major_formatter(PL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.85", linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def ap_box(ax, summary, region, col, x0=0.06, y0=0.30):
    """Male zestawienie AP w pustym rogu panelu; kolor probki niesie tozsamosc modelu."""
    ax.text(x0, y0 + 0.10, "AP:", transform=ax.transAxes, fontsize=8, color="0.25")
    y = y0
    for tag, label, color, _ in MODELS:
        v = summary[(summary.model == tag) & (summary.region == region)][col].iloc[0]
        ax.plot([x0 + 0.025], [y + 0.013], marker="s", markersize=4, color=color,
                transform=ax.transAxes, clip_on=False, linestyle="none")
        ax.text(x0 + 0.085, y, "{:.4f}".format(v).replace(".", ","),
                transform=ax.transAxes, fontsize=8, color="0.25")
        y -= 0.085


def figure_legend(fig, n_extra=None):
    handles = [Line2D([0], [0], color=c, linestyle=ls, linewidth=1.8, label=lab)
               for _, lab, c, ls in MODELS]
    if n_extra:
        handles.append(n_extra)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.06))


def plot_voxelwise():
    df = pd.read_csv(f"{IN_DIR}/pr_voxelwise.csv")
    summary = pd.read_csv(f"{IN_DIR}/pr_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.3), sharey=True)
    for ax, (reg, title) in zip(axes, REGIONS):
        style_axes(ax)
        for tag, label, color, ls in MODELS:
            g = df[(df.model == tag) & (df.region == reg)].sort_values("recall")
            ax.plot(g.recall, g.precision, color=color, linestyle=ls,
                    label=label, zorder=3)
        ax.set_xlim(0, 1.002)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, pad=6)
        ax.set_xlabel("Czułość")

        # inset: zoom on the curve's knee, where the models actually differ
        ins = ax.inset_axes([0.09, 0.13, 0.42, 0.38])
        for tag, label, color, ls in MODELS:
            g = df[(df.model == tag) & (df.region == reg)].sort_values("recall")
            ins.plot(g.recall, g.precision, color=color, linestyle=ls, linewidth=1.2)
        ins.set_xlim(0.85, 1.0)
        ins.set_ylim(0.6, 1.005)
        ins.set_xticks([0.85, 1.0])
        ins.set_yticks([0.6, 1.0])
        ins.xaxis.set_major_formatter(PL)
        ins.yaxis.set_major_formatter(PL)
        ins.tick_params(labelsize=6.5, length=2, pad=1)
        for s in ("top", "right"):
            ins.spines[s].set_visible(False)
        ins.grid(True, color="0.9", linewidth=0.4)
        ins.set_axisbelow(True)
        ax.indicate_inset_zoom(ins, edgecolor="0.6", linewidth=0.6, alpha=0.8)

        ap_box(ax, summary, reg, "voxel_ap_micro", x0=0.60, y0=0.28)

    axes[0].set_ylabel("Precyzja")
    figure_legend(fig)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/pr_voxelwise.{ext}")
    plt.close(fig)


def plot_lesionwise():
    df = pd.read_csv(f"{IN_DIR}/pr_lesionwise.csv")
    summary = pd.read_csv(f"{IN_DIR}/pr_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.3), sharey=True)
    for ax, (reg, title) in zip(axes, REGIONS):
        style_axes(ax)
        for tag, label, color, ls in MODELS:
            g = df[(df.model == tag) & (df.region == reg)].sort_values("recall")
            ax.plot(g.recall, g.precision, color=color, linestyle=ls,
                    label=label, zorder=3)
            row = summary[(summary.model == tag) & (summary.region == reg)].iloc[0]
            # argmax working point - Table 4.3 is computed here
            ax.plot(row.lesion_argmax_recall, row.lesion_argmax_precision,
                    marker="*", markersize=11, color=color,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.set_xlim(0, 1.002)
        ax.set_ylim(0.3, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, pad=6)
        ax.set_xlabel("Czułość detekcji")
        ap_box(ax, summary, reg, "lesion_detection_ap")

    axes[0].set_ylabel("Precyzja detekcji")
    star = Line2D([0], [0], marker="*", markersize=10, color="0.35",
                  linestyle="none", label="punkt pracy reguły argmax")
    figure_legend(fig, n_extra=star)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/pr_lesionwise.{ext}")
    plt.close(fig)


HOUSE_STYLE = {
    "base":      dict(color="#0072B2", ls="-",         label="Swin-UNETR Base"),
    "detach":    dict(color="#009E73", ls=(0, (6, 3)),  label="Multi-task (Detach)"),
    "multitask": dict(color="#D55E00", ls=(0, (1.6, 2.2)), label="Multi-task (standard)"),
}
RECALL_CUTOFF = 0.5   # ponizej tego recall krzywe sa plaskie (precyzja ~1) - odcinamy przy kadrowaniu


def _adaptive_lims(values, pad_frac=0.12, pad_min=0.01):
    lo, hi = float(np.min(values)), float(np.max(values))
    pad = max((hi - lo) * pad_frac, pad_min)
    return max(lo - pad, 0.0), min(hi + pad, 1.0)


def _five_ticks(lo, hi):
    return [round(lo + i * (hi - lo) / 4, 3) for i in range(5)]


def plot_lesionwise_house_style():
    df = pd.read_csv(f"{IN_DIR}/pr_lesionwise.csv")
    summary = pd.read_csv(f"{IN_DIR}/pr_summary.csv")

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"]})

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))
    for ax, (reg, title) in zip(axes, REGIONS):
        sub = df[df.region == reg]
        vis = sub[sub.recall >= RECALL_CUTOFF]
        xlo, xhi = _adaptive_lims(vis.recall.values)
        ylo, yhi = _adaptive_lims(vis.precision.values)

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        xt, yt = _five_ticks(xlo, xhi), _five_ticks(ylo, yhi)
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.xaxis.set_major_formatter(PL)
        ax.yaxis.set_major_formatter(PL)

        # light-gray frame on all 4 sides, matching figure_sweep's style
        for s in ax.spines.values():
            s.set_color("0.85")
            s.set_linewidth(1.0)
        ax.grid(True, color="0.88", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors="0.35", labelsize=8)

        for tag, st in HOUSE_STYLE.items():
            g = sub[sub.model == tag].sort_values("recall")
            ax.plot(g.recall, g.precision, color=st["color"], linestyle=st["ls"],
                    linewidth=1.6, zorder=3)
            idx = np.linspace(0, len(g) - 1, min(22, len(g))).astype(int)
            gm = g.iloc[idx]
            ax.plot(gm.recall, gm.precision, "o", color=st["color"],
                    markersize=3.4, markeredgecolor="white", markeredgewidth=0.5, zorder=4)
            row = summary[(summary.model == tag) & (summary.region == reg)].iloc[0]
            r_am, p_am = row.lesion_argmax_recall, row.lesion_argmax_precision
            if xlo <= r_am <= xhi and ylo <= p_am <= yhi:
                ax.plot(r_am, p_am, marker="*", markersize=12, color=st["color"],
                        markeredgecolor="white", markeredgewidth=0.8, zorder=5)

        ax.set_title(title, fontsize=11, fontweight="bold", color="0.15", pad=8)
        ax.set_xlabel("Recall (detekcja ognisk)", fontsize=9, color="0.15")
        if reg == "WT":
            ax.set_ylabel("Precision (detekcja ognisk)", fontsize=9, color="0.15")

    handles = []
    for tag, st in HOUSE_STYLE.items():
        aps = [summary[(summary.model == tag) & (summary.region == r)]
               ["lesion_detection_ap"].iloc[0] for r, _ in REGIONS]
        lbl = (f"{st['label']}  —  AP: WT {aps[0]:.3f} / TC {aps[1]:.3f} / ET {aps[2]:.3f}"
               .replace(".", ","))
        handles.append(Line2D([0], [0], color=st["color"], linestyle=st["ls"],
                              linewidth=1.8, marker="o", markersize=4,
                              markeredgecolor="white", label=lbl))
    handles.append(Line2D([0], [0], marker="*", markersize=10, color="0.35",
                          linestyle="none", label="punkt pracy reguły argmax"))
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.20))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_DIR}/pr_lesionwise_housestyle.{ext}")
    plt.close(fig)
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_voxelwise()
    plot_lesionwise()
    plot_lesionwise_house_style()
    print(f"Saved figures to {OUT_DIR}/")


# Alternate version of Figure 4.2, styled like tools/plot_pr_curves.py