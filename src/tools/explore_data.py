"""Explore the WDBC dataset: summary stats + graphs (class balance,
distributions, correlations, boxplots) saved as PNGs, as recommended
by the subject before implementing the MLP."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLOR_B = "#2a78d6"
COLOR_M = "#eb6834"
CLASS_COLORS = {"B": COLOR_B, "M": COLOR_M}
CLASS_LABELS = {"B": "Benign", "M": "Malignant"}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"

STATS = ["mean", "se", "worst"]
FEATURES = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points",
    "symmetry", "fractal_dimension",
]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def print_summary(df):
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    counts = df["diagnosis"].value_counts()
    total = len(df)
    print("Class balance:")
    for label in ("B", "M"):
        n = counts.get(label, 0)
        print(f"  {CLASS_LABELS[label]:9s} ({label}): {n:4d}  ({n / total:.1%})")

    print("\nMissing values per column:", int(df.isna().sum().sum()))
    print(f"Duplicate ids: {df['id'].duplicated().sum()}")

    print("\nDescribe (mean-group features):")
    mean_cols = [f"{feat}_mean" for feat in FEATURES]
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(df[mean_cols].describe().T[["mean", "std", "min", "max"]])
    print()


def plot_class_balance(df, outdir):
    fig, ax = plt.subplots(figsize=(4, 4))
    counts = df["diagnosis"].value_counts().reindex(["B", "M"])
    bars = ax.bar(
        [CLASS_LABELS[c] for c in counts.index],
        counts.values,
        color=[CLASS_COLORS[c] for c in counts.index],
        width=0.6,
        zorder=3,
    )
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{value} ({value / len(df):.0%})",
            ha="center", va="bottom", fontsize=9, color=INK_PRIMARY,
        )
    ax.set_title("Class balance", color=INK_PRIMARY, fontsize=11, loc="left")
    ax.set_ylabel("Count", color=INK_SECONDARY, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(outdir / "class_balance.png", dpi=150)
    plt.close(fig)


def plot_feature_distributions(df, outdir, stat="mean"):
    cols = [f"{feat}_{stat}" for feat in FEATURES]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, col in zip(axes.flat, cols):
        for label in ("B", "M"):
            values = df.loc[df["diagnosis"] == label, col]
            ax.hist(
                values, bins=20, alpha=0.55, color=CLASS_COLORS[label],
                label=CLASS_LABELS[label], zorder=3,
            )
        ax.set_title(col.replace(f"_{stat}", ""), fontsize=9, color=INK_PRIMARY)
        style_axes(ax)
    fig.suptitle(f"Feature distributions by diagnosis ({stat})", fontsize=12, y=1.08)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
    ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / f"distributions_{stat}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplots(df, outdir, stat="mean"):
    cols = [f"{feat}_{stat}" for feat in FEATURES]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, col in zip(axes.flat, cols):
        data = [df.loc[df["diagnosis"] == label, col] for label in ("B", "M")]
        bp = ax.boxplot(
            data, tick_labels=[CLASS_LABELS[l] for l in ("B", "M")],
            patch_artist=True, widths=0.5, zorder=3,
            medianprops=dict(color=INK_PRIMARY, linewidth=1.5),
            whiskerprops=dict(color=INK_SECONDARY),
            capprops=dict(color=INK_SECONDARY),
        )
        for patch, label in zip(bp["boxes"], ("B", "M")):
            patch.set_facecolor(CLASS_COLORS[label])
            patch.set_alpha(0.6)
            patch.set_edgecolor(CLASS_COLORS[label])
        ax.set_title(col.replace(f"_{stat}", ""), fontsize=9, color=INK_PRIMARY)
        style_axes(ax)
    fig.suptitle(f"Feature spread by diagnosis ({stat})", fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(outdir / f"boxplots_{stat}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df, outdir):
    mean_cols = [f"{feat}_mean" for feat in FEATURES]
    corr = df[mean_cols].corr()
    labels = [c.replace("_mean", "") for c in mean_cols]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color=INK_SECONDARY)
    ax.set_yticklabels(labels, fontsize=8, color=INK_SECONDARY)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, f"{corr.values[i, j]:.1f}",
                ha="center", va="center", fontsize=6.5,
                color=INK_PRIMARY if abs(corr.values[i, j]) < 0.6 else "white",
            )
    ax.set_title("Correlation between mean features", fontsize=11, color=INK_PRIMARY, loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(outdir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_top_correlated_scatter(df, outdir, n=4):
    mean_cols = [f"{feat}_mean" for feat in FEATURES]
    target = (df["diagnosis"] == "M").astype(int)
    correlations = df[mean_cols].corrwith(target).abs().sort_values(ascending=False)
    top_features = correlations.head(n).index.tolist()

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    for ax, feat in zip(axes, top_features):
        for label in ("B", "M"):
            subset = df[df["diagnosis"] == label]
            ax.scatter(
                subset[feat], subset["diagnosis"].map({"B": 0, "M": 1}) +
                np.random.uniform(-0.05, 0.05, len(subset)),
                color=CLASS_COLORS[label], s=25, alpha=0.6, edgecolors="none",
                label=CLASS_LABELS[label], zorder=3,
            )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["B", "M"])
        ax.set_xlabel(feat.replace("_mean", ""), fontsize=9, color=INK_SECONDARY)
        style_axes(ax)
    fig.suptitle(f"Top {n} features most correlated with diagnosis", fontsize=12, y=1.10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
    ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "top_correlated_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="data/data_clean.csv", help="path to the cleaned csv")
    parser.add_argument("--outdir", default="outputs/", help="directory to save the figures")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"error: {csv_path} not found (run src/clean_data.py first)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    print_summary(df)

    plot_class_balance(df, outdir)
    plot_feature_distributions(df, outdir, stat="mean")
    plot_boxplots(df, outdir, stat="mean")
    plot_correlation_heatmap(df, outdir)
    plot_top_correlated_scatter(df, outdir)

    print(f"Figures written to {outdir}/")


if __name__ == "__main__":
    main()
