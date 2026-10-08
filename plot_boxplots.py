"""
Box plots of the gap in sliced Wasserstein distance with respect to MIDAS with eta = 1/4
(mixture case by default), from the CSV files written by run_experiment.py.

For every method M and every run r, at a given budget (default, --pairing index):
    gap_r(M) = log SW_r(M) - log SW_r(MIDAS, eta = 1/4)        (--scale log, default)
    gap_r(M) =     SW_r(M) -     SW_r(MIDAS, eta = 1/4)        (--scale linear)
i.e. run r of M minus run r of eta = 1/4. The MIDAS runs with different eta use the same seeds
(same initial sample from q0), so for them the difference is a paired comparison; AIS and KAMH use
their own seeds, so for them run r is simply matched with run r of eta = 1/4.
With --pairing mean, the reference is instead the mean over the runs of eta = 1/4 (and the box of
eta = 1/4 shows its own spread around 0).
Negative gap = smaller distance than MIDAS with eta = 1/4.

Usage
    python3 plot_boxplots.py                                   # mixture, final budget
    python3 plot_boxplots.py --case mixture anisotropic --budget 20000 62000 --scale linear
Outputs: figures/boxplots_<case>.png/.pdf and results/<case>/boxplot_gaps.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_results import METHODS

REFERENCE = "midas_eta0.25"
SHORT = {"midas_eta0.25": "MIDAS\nη=1/4", "midas_eta0.5": "MIDAS\nη=1/2", "midas_eta0.75": "MIDAS\nη=3/4",
         "midas_eta1": "MIDAS\nη=1", "ais": "AIS\npaper", "kamh": "KAMH\npaper",
         "ais_tuned": "AIS\ntuned", "kamh_tuned": "KAMH\ntuned"}


def load(results, case, key):
    path = os.path.join(results, case, key + ".csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def values_at(df, budget, scale):
    """Runs of one method at the closest available budget <= budget (AIS has its own budgets)."""
    df = df.dropna()
    ok = df[df.n_eval <= budget]
    if ok.empty:
        return None, None
    row = ok.iloc[(ok.n_eval - budget).abs().argmin()]
    v = row.drop("n_eval").to_numpy(float)
    v = v[np.isfinite(v)]
    return (np.exp(v) if scale == "linear" else v), int(row.n_eval)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", nargs="+", default=["mixture"])
    p.add_argument("--budget", nargs="+", type=int, default=[62000])
    p.add_argument("--scale", choices=["log", "linear"], default="log")
    p.add_argument("--pairing", choices=["mean", "index"], default="index")
    p.add_argument("--results", default="results")
    a = p.parse_args()
    os.makedirs("figures", exist_ok=True)
    for case in a.case:
        ref_df = load(a.results, case, REFERENCE)
        if ref_df is None:
            print("no %s for %s, skipped" % (REFERENCE, case))
            continue
        fig, axes = plt.subplots(1, len(a.budget), figsize=(6.6 * len(a.budget), 4.6), squeeze=False)
        rows = []
        for ax, B in zip(axes[0], a.budget):
            ref, _ = values_at(ref_df, B, a.scale)
            data, labels, colors = [], [], []
            for key, label, color, _ in METHODS:
                if a.pairing == "index" and key == REFERENCE:
                    continue                          # identically zero
                df = load(a.results, case, key)
                if df is None:
                    continue
                v, used = values_at(df, B, a.scale)
                if v is None:
                    continue
                if a.pairing == "index":
                    n = min(len(v), len(ref))
                    gap = v[:n] - ref[:n]
                else:
                    gap = v - np.mean(ref)
                data.append(gap)
                labels.append(SHORT.get(key, label))
                colors.append(color)
                rows.append(dict(case=case, budget=B, budget_used=used, method=key, n_runs=len(gap),
                                 median=np.median(gap), q25=np.percentile(gap, 25), q75=np.percentile(gap, 75),
                                 mean=np.mean(gap), share_better=np.mean(gap < 0)))
            bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                            medianprops=dict(color="black", lw=1.5),
                            flierprops=dict(marker="o", ms=3, mfc="none", mec="0.4"))
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.55)
                patch.set_edgecolor(col)
            ax.axhline(0, color="0.3", lw=1, ls="--")
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=8, rotation=0)
            what = "log SW − log SW(η=1/4)" if a.scale == "log" else "SW − SW(η=1/4)"
            ax.set_ylabel(what + (" (same run)" if a.pairing == "index" else " (vs mean of η=1/4)"))
            ax.set_title("%s, %d evaluations of f" % (case, B), fontsize=11)
            ax.grid(True, axis="y", color="0.9", lw=0.8)
            ax.set_axisbelow(True)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
        fig.suptitle("Gap with respect to MIDAS (η = 1/4); below 0 = better than η = 1/4", fontsize=11)
        fig.tight_layout()
        out = os.path.join("figures", "boxplots_%s" % case)
        fig.savefig(out + ".png", dpi=200)
        fig.savefig(out + ".pdf")
        plt.close(fig)
        tab = pd.DataFrame(rows)
        tab.to_csv(os.path.join(a.results, case, "boxplot_gaps.csv"), index=False)
        print("\n=== %s ===" % case)
        print(tab.drop(columns="case").to_string(index=False, float_format="%.3f"))
        print("figure:", out + ".png")


if __name__ == "__main__":
    main()
