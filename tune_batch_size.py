"""
Choice of the batch size m of MIDAS at a FIXED budget of evaluations of the target.

For each case, eta and m, MIDAS is run with the same budget (initial batch 2000 + n_iter * m = 62 000,
so n_iter = 60 000 / m) and the metric is computed with the importance weights at a few budgets.
Everything else is as in run_experiment.py (sqrt(N) Gaussian kernels,
b_n = 0.5/sqrt(d) (sqrt(n m) + 1)^(-1/(4+d)), lambda_n = 1/log(m n + 10), gamma_n = 1/(n+10)).

Conclusion rule (per case and eta, at the final budget):
  * best m = best mean metric;
  * every other m is compared with the best one by a Welch t-test, p-values corrected with Holm;
  * "equivalent" = not significantly different from the best (alpha = 5%).
A global recommendation is the m with the best average rank over all (case, eta), together with the
number of (case, eta) where it is the best or equivalent to the best, and its mean running time.

Usage
    OMP_NUM_THREADS=1 python tune_batch_size.py                       # 4 cases, 4 eta, m = 50 100 300 1000
    OMP_NUM_THREADS=1 python tune_batch_size.py --nrep 20 --m 100 300 600 --case mixture coldstart
Outputs (results/batch_size/): runs.csv, summary.csv, tests.csv, batch_size.png/.pdf
"""
import argparse
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from AIS import AIS
from cases import BS_INI, BUDGET, CASES, get_case, midas_parameters

OUT = os.path.join("results", "batch_size")
EVAL_BUDGETS = [12000, 32000, BUDGET]
_CASE = {}


def case_obj(name):
    if name not in _CASE:
        _CASE[name] = get_case(name)
    return _CASE[name]


def one_run(job):
    case_name, eta, m, seed = job
    C = case_obj(case_name)
    b, gamma, lambd = midas_parameters(C.dim, m=m)
    n_iter = (BUDGET - BS_INI) // m
    np.random.seed(seed)
    t0 = time.perf_counter()
    a = AIS(lambd, gamma, b, eta, C.q0, C.logf, batch_size=m, batch_size_ini=BS_INI, d=C.dim,
            distF=False, Indicator="", log_target=True, kernel="gaussian", n_kernels="sqrt", verbose=False)
    a.iteration(n_iter)
    run_time = time.perf_counter() - t0
    rows = []
    for B in EVAL_BUDGETS:
        N = BS_INI + ((B - BS_INI) // m) * m          # particles available at budget B
        lw = a.newWeight[:N]
        w = np.exp(lw - np.max(lw))
        rows.append(dict(case=case_name, eta=eta, m=m, seed=seed, budget=B, n_iter=n_iter,
                         metric=C.metric(a.X[:N], w), time_s=run_time))
    return rows


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def analyse(df, alpha=0.05):
    final = df[df.budget == BUDGET]
    summ, tests = [], []
    for (case, eta), g in final.groupby(["case", "eta"]):
        higher = case_obj(case).higher_is_better
        s = g.groupby("m").agg(mean=("metric", "mean"), sd=("metric", "std"), n=("metric", "size"),
                               time_s=("time_s", "mean")).reset_index()
        s["ci95"] = stats.t.ppf(0.975, s.n - 1) * s.sd / np.sqrt(s.n)
        best = s.loc[s["mean"].idxmax() if higher else s["mean"].idxmin(), "m"]
        others = [m for m in s.m if m != best]
        pv = [stats.ttest_ind(g[g.m == best].metric, g[g.m == m].metric, equal_var=False).pvalue for m in others]
        padj = dict(zip(others, holm(pv))) if others else {}
        s["rank"] = s["mean"].rank(ascending=not higher)
        s["best"] = s.m == best
        s["p_vs_best_holm"] = [np.nan if m == best else padj[m] for m in s.m]
        s["equivalent_to_best"] = s.best | (s.p_vs_best_holm >= alpha)
        s.insert(0, "eta", eta)
        s.insert(0, "case", case)
        summ.append(s)
        for m in others:
            tests.append(dict(case=case, eta=eta, best_m=best, m=m, p_holm=padj[m]))
    summ = pd.concat(summ, ignore_index=True)
    glob = summ.groupby("m").agg(mean_rank=("rank", "mean"), n_best=("best", "sum"),
                                 n_best_or_equivalent=("equivalent_to_best", "sum"),
                                 n_settings=("rank", "size"), mean_time_s=("time_s", "mean")).reset_index()
    return summ, pd.DataFrame(tests), glob.sort_values("mean_rank")


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {0.25: "#2a78d6", 0.5: "#eb6834", 0.75: "#1baf7a", 1.0: "#eda100"}
    final = df[df.budget == BUDGET]
    cases = [c for c in CASES if c in set(final.case)]
    fig, axes = plt.subplots(1, len(cases), figsize=(4.3 * len(cases), 3.8), squeeze=False)
    for ax, case in zip(axes[0], cases):
        for eta, g in final[final.case == case].groupby("eta"):
            s = g.groupby("m").metric.agg(["mean", "std", "size"]).reset_index()
            h = stats.t.ppf(0.975, s["size"] - 1) * s["std"] / np.sqrt(s["size"])
            col = colors.get(eta, "0.3")
            ax.fill_between(s.m, s["mean"] - h, s["mean"] + h, color=col, alpha=0.15, lw=0)
            ax.plot(s.m, s["mean"], "-o", color=col, lw=2, ms=4, label="η = %g" % eta)
        ax.set_xscale("log")
        ax.set_xticks(sorted(final.m.unique()))
        ax.set_xticklabels([str(m) for m in sorted(final.m.unique())])
        ax.set_title(case, fontsize=11)
        ax.set_xlabel("batch size m")
        ax.set_ylabel(case_obj(case).metric_label)
        ax.grid(True, color="0.9", lw=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("MIDAS, final metric at %d evaluations vs batch size (mean, 95%% CI)" % BUDGET, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "batch_size.png"), dpi=200)
    fig.savefig(os.path.join(OUT, "batch_size.pdf"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", nargs="+", default=CASES)
    p.add_argument("--eta", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--m", nargs="+", type=int, default=[50, 100, 300, 1000])
    p.add_argument("--nrep", type=int, default=10)
    p.add_argument("--cores", type=int, default=None)
    p.add_argument("--analyse_only", action="store_true", help="only re-analyse results/batch_size/runs.csv")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.analyse_only:
        df = pd.read_csv(os.path.join(OUT, "runs.csv"))
    else:
        jobs = [(c, e, m, 70000 + r) for c, e, m, r in itertools.product(a.case, a.eta, a.m, range(a.nrep))]
        jobs.sort(key=lambda j: j[2])                 # small m (longest runs) first
        t0 = time.time()
        with Pool(a.cores) as pool:
            df = pd.DataFrame([row for rows in pool.map(one_run, jobs, chunksize=1) for row in rows])
        df.to_csv(os.path.join(OUT, "runs.csv"), index=False)
        print("%d runs in %.0f s" % (len(jobs), time.time() - t0))
    summ, tests, glob = analyse(df)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    tests.to_csv(os.path.join(OUT, "tests.csv"), index=False)
    pd.set_option("display.width", 160)
    for (case, eta), s in summ.groupby(["case", "eta"]):
        print("\n=== %s, eta = %g (final budget) ===" % (case, eta))
        print(s.drop(columns=["case", "eta", "sd"]).to_string(index=False, float_format="%.3f"))
    print("\n=== Global ranking of m over all (case, eta) ===")
    print(glob.to_string(index=False, float_format="%.2f"))
    best = glob.iloc[0]
    print("\nRecommended m = %d: mean rank %.2f, best or equivalent to the best in %d / %d settings, %.1f s per run."
          % (best.m, best.mean_rank, best.n_best_or_equivalent, best.n_settings, best.mean_time_s))
    plot(df)


if __name__ == "__main__":
    main()
