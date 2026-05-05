#!/usr/bin/env python3
"""
plot_speed_vs_mom.py

Reads benchmark_results.csv (produced by median_benchmark.cpp) and
produces plots_mom.pdf -- a multi-panel figure showing the speed of
each algorithm RELATIVE TO Median-of-Medians (BFPRT) for every
dataset and every input size.

Layout (matches Alexandrescu, "Fast Deterministic Selection",
SEA 2017, Figs. 1--6):
  * one bar-chart subplot per dataset
  * x-axis = input size (log scale, 10^3 ... 10^7)
  * y-axis = relative speed = t_MoM(n) / t_algo(n)
            (so a bar of height > 1 means the algorithm is faster
             than MoM at that size; a bar of height = 1 is the
             MoM reference itself)
  * one coloured bar per algorithm at each size

Usage:
  python3 plot_speed_vs_mom.py [--input benchmark_results.csv]
                               [--output plots_mom.pdf]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe on headless boxes
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASELINE = "MoM-BFPRT"

# Order of algorithms in each grouped bar chart (left-to-right).
ALGO_ORDER = ["MoM-BFPRT", "QuickSelect", "LazySelect"]

# Distinct, print-friendly colours.
ALGO_COLOURS = {
    "MoM-BFPRT":   "#4C72B0",  # blue (the reference; always 1.0)
    "QuickSelect": "#DD8452",  # warm orange
    "LazySelect":  "#55A868",  # green
    # Backwards-compatible alias for older CSVs that called LazySelect "Sampling".
    "Sampling":    "#55A868",
}

# Order of datasets in the figure grid (matches the original plots.pdf order).
DATASETS = ["random", "random01", "organpipe", "sorted", "rotated"]


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------
def load_results(path: Path) -> dict:
    """
    Returns: results[dataset][algo][n] = time_in_seconds (float),
    skipping any rows whose time field is "SKIP".
    """
    results: dict = defaultdict(lambda: defaultdict(dict))
    with path.open(encoding="utf-16") as fh:
        reader = csv.DictReader(fh)
  
        for row in reader:
            t = row["time_s"]
            if t == "SKIP" or t == "":
                continue
            try:
                t_val = float(t)
            except ValueError:
                continue
            n = int(row["n"])
            ds = row["dataset"]
            algo = row["algorithm"]
            # Skip the legacy "Sort-Baseline" rows: this benchmark no
            # longer reports against std::sort.
            if algo == "Sort-Baseline":
                continue
            # Normalise old "Sampling" name to "LazySelect".
            if algo == "Sampling":
                algo = "LazySelect"
            results[ds][algo][n] = t_val
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_panel(ax, ds_name: str, ds_data: dict) -> None:
    """Draw the bar-chart subplot for one dataset."""
    # Determine the union of sizes for which we have BASELINE data
    # (relative speed is undefined where MoM is missing).
    if BASELINE not in ds_data:
        ax.set_title(f"{ds_name} (no MoM data)")
        ax.axis("off")
        return

    sizes = sorted(ds_data[BASELINE].keys())
    if not sizes:
        ax.set_title(f"{ds_name} (no MoM data)")
        ax.axis("off")
        return

    # Algorithms actually present in this dataset, in the canonical order.
    algos_present = [a for a in ALGO_ORDER if a in ds_data]

    n_algos = len(algos_present)
    n_sizes = len(sizes)

    # x-positions for grouped bars.
    group_centers = np.arange(n_sizes)
    bar_width = 0.8 / max(n_algos, 1)

    for i, algo in enumerate(algos_present):
        heights = []
        for n in sizes:
            t_mom = ds_data[BASELINE].get(n)
            t_x   = ds_data[algo].get(n)
            if t_mom is None or t_x is None or t_x == 0.0:
                heights.append(np.nan)
            else:
                heights.append(t_mom / t_x)
        offsets = group_centers + (i - (n_algos - 1) / 2) * bar_width
        ax.bar(
            offsets, heights, width=bar_width,
            color=ALGO_COLOURS.get(algo, "#888888"),
            edgecolor="black", linewidth=0.4,
            label=algo,
        )

    # Reference line at y = 1 (= MoM speed).
    ax.axhline(1.0, color="grey", linewidth=0.7, linestyle="--", zorder=0)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([_fmt_size(n) for n in sizes], fontsize=8)
    ax.set_xlabel("Input size $n$", fontsize=9)
    ax.set_ylabel("Relative speed   ($t_{\\mathrm{MoM}} / t_{\\mathrm{algo}}$)",
                  fontsize=9)
    ax.set_title(f"{ds_name} dataset", fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.7)


def _fmt_size(n: int) -> str:
    """Pretty 10^k labels for axis ticks."""
    if n >= 1_000_000_000:
        return f"$10^{{{int(round(np.log10(n)))}}}$"
    for k in range(2, 11):
        if n == 10 ** k:
            return f"$10^{{{k}}}$"
    # 5e5, 5e6 etc.
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0:
        return f"{n // 1_000}K"
    return str(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  default="benchmark_results.csv")
    parser.add_argument("--output", default="plots_mom.pdf")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.is_file():
        raise SystemExit(f"benchmark CSV not found at {in_path}")

    results = load_results(in_path)

    # 5 datasets -> 3x2 grid (one panel left empty).
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    axes_flat = axes.flatten()

    for ax, ds in zip(axes_flat, DATASETS):
        make_panel(ax, ds, results.get(ds, {}))

    # Hide the last (unused) subplot.
    for j in range(len(DATASETS), len(axes_flat)):
        axes_flat[j].axis("off")

    # Single legend at the bottom of the figure.
    handles, labels = [], []
    for algo in ALGO_ORDER:
        handles.append(
            plt.Rectangle((0, 0), 1, 1,
                          facecolor=ALGO_COLOURS.get(algo, "#888888"),
                          edgecolor="black", linewidth=0.4)
        )
        labels.append(algo)
    fig.legend(handles, labels, loc="lower center",
               ncol=len(ALGO_ORDER), frameon=False,
               bbox_to_anchor=(0.5, 0.01), fontsize=10)

    fig.suptitle(
        "Speed relative to Median-of-Medians (BFPRT) baseline",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
