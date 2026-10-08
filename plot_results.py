"""
Read every results/<case>/<method>.csv written by run_experiment.py and plot, for each case,
the mean metric of each method against the number of evaluations of the target, with a
95% confidence interval for the mean:  mean +/- t_{0.975, R-1} * sd / sqrt(R).

Usage:  python plot_results.py            ->  figures/all_cases.png/.pdf, figures/<case>.png/.pdf,
                                              results/summary.csv
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

CASES = ["coldstart", "mixture", "anisotropic", "bayreg"]
TITLES = {"coldstart": "Cold start", "mixture": "Gaussian mixture",
          "anisotropic": "Anisotropic Gaussian mixture", "bayreg": "Bayesian logistic regression (waveform)"}
YLABEL = {"bayreg": "Test accuracy"}
DEFAULT_YLABEL = "Log sliced Wasserstein distance"
# fixed order and colour per method (categorical palette, colour follows the method)
METHODS = [("midas_eta0.25", "MIDAS: η = 0.25", "#2a78d6", "-"),
           ("midas_eta0.5", "MIDAS: η = 0.5", "#eb6834", "-"),
           ("midas_eta0.75", "MIDAS: η = 0.75", "#1baf7a", "-"),
           ("midas_eta1", "MIDAS: η = 1", "#eda100", "-"),
           ("ais", "AIS (paper settings)", "#4a3aa7", ":"),
           ("kamh", "KAMH (paper settings)", "#e34948", ":"),
           ("ais_tuned", "AIS (tuned)", "#4a3aa7", "--"),
           ("kamh_tuned", "KAMH (tuned)", "#e34948", "--")]


def summarise(path):
    df = pd.read_csv(path).dropna()
    x = df["n_eval"].to_numpy()
    R = df.drop(columns="n_eval").to_numpy()
    n = R.shape[1]
    mean = R.mean(axis=1)
    half = stats.t.ppf(0.975, n - 1) * R.std(axis=1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return x, mean, mean - half, mean + half, n


def draw(ax, case, rows):
    found = False
    for key, label, color, ls in METHODS:
        path = os.path.join("results", case, key + ".csv")
        if not os.path.exists(path):
            continue
        found = True
        x, m, lo, hi, n = summarise(path)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(x, m, ls, color=color, lw=2, marker="o" if key.startswith("ais") else None, ms=4, label=label)
        for xi, mi, l, h in zip(x, m, lo, hi):
            rows.append(dict(case=case, method=key, n_eval=xi, mean=mi, ci95_low=l, ci95_high=h, n_runs=n))
    ax.set_title(TITLES.get(case, case), fontsize=11)
    ax.set_xlabel("Number of evaluations of the target")
    ax.set_ylabel(YLABEL.get(case, DEFAULT_YLABEL))
    ax.grid(True, color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%dk" % (v / 1000) if v else "0"))
    return found


def main():
    os.makedirs("figures", exist_ok=True)
    cases = [c for c in CASES if glob.glob(os.path.join("results", c, "*.csv"))]
    rows = []
    # one figure per case
    for case in cases:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        draw(ax, case, [])
        lo, hi = ax.get_ylim()                       # free space under the curves for the legend
        ax.set_ylim(lo - 0.42 * (hi - lo), hi)
        ax.legend(loc="lower left", ncol=2, fontsize=8, frameon=True, framealpha=0.85, edgecolor="0.85",
                  handlelength=1.6, labelspacing=0.25, columnspacing=1.0, borderpad=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", case + ".png"), dpi=200)
        fig.savefig(os.path.join("figures", case + ".pdf"))
        plt.close(fig)
    # all cases together
    ncol = 2
    nrow = int(np.ceil(len(cases) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.0 * nrow), squeeze=False)
    for ax, case in zip(axes.ravel(), cases):
        draw(ax, case, rows)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Mean over independent runs, shaded band = 95% confidence interval of the mean", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(os.path.join("figures", "all_cases.png"), dpi=200)
    fig.savefig(os.path.join("figures", "all_cases.pdf"))
    pd.DataFrame(rows).to_csv(os.path.join("results", "summary.csv"), index=False)
    print("figures written in figures/, summary in results/summary.csv")


if __name__ == "__main__":
    main()