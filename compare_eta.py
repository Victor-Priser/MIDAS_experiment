"""
Proportion of runs in which MIDAS with eta = 1/4 is better than MIDAS with eta = 1/2, 3/4, 1
(and, optionally, than the competitors), from the CSV files written by run_experiment.py.

Run r of eta = 1/4 is compared with run r of the other method (same seed for the MIDAS runs).
"Better" = smaller log sliced Wasserstein distance (synthetic cases) or higher test accuracy (bayreg).
For each comparison the script reports:
    better   : proportion of runs where eta = 1/4 is strictly better, with a 95% Wilson interval
    ties     : proportion of exact ties
    p_sign   : two-sided sign test (H0: each method is better in half of the non-tied runs)
plus the proportion of runs where eta = 1/4 is better than ALL of eta = 1/2, 3/4 and 1 at once.

Usage
    python3 compare_eta.py                                         # mixture, final budget
    python3 compare_eta.py --case mixture anisotropic --budget 20000 62000 --competitors
Output: results/<case>/compare_eta.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

REFERENCE = "midas_eta0.25"
OTHERS = ["midas_eta0.5", "midas_eta0.75", "midas_eta1"]
COMPETITORS = ["ais", "kamh", "ais_tuned", "kamh_tuned"]
HIGHER_IS_BETTER = {"bayreg": True}


def runs_at(results, case, key, budget):
    path = os.path.join(results, case, key + ".csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path).dropna()
    ok = df[df.n_eval <= budget]
    if ok.empty:
        return None, None
    row = ok.iloc[(ok.n_eval - budget).abs().argmin()]
    return row.drop("n_eval").to_numpy(float), int(row.n_eval)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return c - h, c + h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", nargs="+", default=["mixture"])
    p.add_argument("--budget", nargs="+", type=int, default=[62000])
    p.add_argument("--competitors", action="store_true", help="also compare with AIS and KAMH")
    p.add_argument("--results", default="results")
    a = p.parse_args()
    pd.set_option("display.width", 160)
    for case in a.case:
        higher = HIGHER_IS_BETTER.get(case, False)
        rows = []
        for B in a.budget:
            ref, used = runs_at(a.results, case, REFERENCE, B)
            if ref is None:
                print("no %s.csv for %s" % (REFERENCE, case))
                break
            wins_all = np.ones(len(ref), bool)
            n_all = len(ref)
            for key in OTHERS + (COMPETITORS if a.competitors else []):
                v, used_v = runs_at(a.results, case, key, B)
                if v is None:
                    continue
                n = min(len(ref), len(v))
                d = (ref[:n] - v[:n]) if higher else (v[:n] - ref[:n])     # > 0 : eta = 1/4 better
                better, ties = int(np.sum(d > 0)), int(np.sum(d == 0))
                worse = n - better - ties
                lo, hi = wilson(better, n)
                p_sign = stats.binomtest(better, better + worse, 0.5).pvalue if better + worse > 0 else np.nan
                rows.append(dict(budget=B, versus=key, budget_versus=used_v, n_runs=n,
                                 better=better / n, ci95_low=lo, ci95_high=hi, ties=ties / n,
                                 worse=worse / n, median_gap=np.median(d), p_sign=p_sign))
                if key in OTHERS:
                    wins_all[:n] &= d > 0
                    n_all = min(n_all, n)
            k = int(np.sum(wins_all[:n_all]))
            lo, hi = wilson(k, n_all)
            rows.append(dict(budget=B, versus="all of eta = 1/2, 3/4, 1", budget_versus=used, n_runs=n_all,
                             better=k / n_all, ci95_low=lo, ci95_high=hi))
        if not rows:
            continue
        tab = pd.DataFrame(rows)
        tab.to_csv(os.path.join(a.results, case, "compare_eta.csv"), index=False)
        print("\n=== %s: proportion of runs where MIDAS eta = 1/4 is better ===" % case)
        print("(median_gap > 0 means eta = 1/4 better; for the synthetic cases it is a gap in log SW)")
        print(tab.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
