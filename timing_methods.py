"""
Computation time as a function of the number of evaluations of the target, for MIDAS (several
subsampling sizes), tuned KAMH and tuned AIS. Only the algorithms are timed: the metric is NOT
included in the times (it is computed afterwards, once, at the final budget).

MIDAS subsampling: the policy q_n uses l_n kernels resampled among the N particles, with
    rho = 0.5  -> l_n = sqrt(N)   (default of the experiments)
    rho = 0.6, 0.75 -> l_n = N^rho
    full       -> l_n = N        (original code)
and the bandwidth tuned to l_n as in the paper: b_n = c/sqrt(d) (l_n + 1)^(-1/(4+d)), c = 0.5.
KAMH and AIS use the configurations of results/tuning/best_configs.json (tune_competitors.py).

Usage
    OMP_NUM_THREADS=1 python3 timing_methods.py                       # mixture, 5 runs per method
    OMP_NUM_THREADS=1 python3 timing_methods.py --case coldstart --nrep 10 --subsampling 0.5 0.75 full
Outputs: results/<case>/timing.csv, results/<case>/timing_final_metric.csv, figures/timing_<case>.png/.pdf
Runs are executed one at a time by default (--cores 1) so that the timings are not perturbed.
"""
import argparse
import functools
import json
import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

from AIS import AIS
from annealed_is import AnnealedImportanceSampling
from cases import BS, BS_INI, BUDGET, CHECKPOINTS, get_case, midas_parameters
from classFonction import Function
from kamh import KAMHMulti

AIS_BUDGETS = np.arange(6000, 60001, 6000)
_CASE = {}


def case_obj(name):
    if name not in _CASE:
        _CASE[name] = get_case(name)
    return _CASE[name]


def n_kernels_rule(N, rho):
    return max(1, int(N )) if rho == "full" else max(1, int(N ** float(rho)))


def bandwidth(n, rho, d, c=0.5, m=BS):
    ell = n_kernels_rule(BS_INI + n * m, rho)
    return c / np.sqrt(d) * (ell + 1) ** (-1.0 / (4 + d))


# ----------------------------------------------------------------------------- one timed run
def time_midas(args):
    case, rho, eta, seed, n_iter = args
    C = case_obj(case)
    _, gamma, lambd = midas_parameters(C.dim)
    b = Function(functools.partial(bandwidth, rho=rho, d=C.dim))
    np.random.seed(seed)
    a = AIS(lambd, gamma, b, eta, C.q0, C.logf, batch_size=BS, batch_size_ini=BS_INI, d=C.dim, distF=False,
            Indicator="", log_target=True, kernel="gaussian",
            n_kernels=functools.partial(n_kernels_rule, rho=rho), verbose=False)
    a.iteration(n_iter)
    n_eval = np.cumsum(a.nEvalF)
    t = np.cumsum(a.timeIter)
    N = int(n_eval[-1])
    lw = a.newWeight[:N]
    metric = C.metric(a.X[:N], np.exp(lw - np.max(lw)))
    return dict(method="midas_rho%s" % rho, n_eval=n_eval, time=t, metric=metric)


def time_kamh(args):
    case, cfg, seed, max_eval = args
    C = case_obj(case)
    np.random.seed(seed)
    nc = cfg["n_chains"]
    X0 = np.asarray(C.q0.simulation(nc), float)
    s = KAMHMulti(C.logf, C.dim, n_chains=nc, sigma=cfg["sigma"], gamma=0.2, nu=2.38 / np.sqrt(C.dim),
                  adapt_scale=(cfg["scale"] == "adaptive"))
    cps = [int(c) for c in CHECKPOINTS if c <= max_eval and c // nc >= 2]
    t0 = time.perf_counter()
    chains, out = s.run(X0, cps[-1], checkpoints=cps, callback=lambda ch: time.perf_counter() - t0)
    S = chains[:, int(cfg["burn"] * chains.shape[1]):].reshape(-1, C.dim)
    metric = C.metric(S, np.ones(S.shape[0]))
    ks = sorted(out)
    return dict(method="kamh_tuned", n_eval=np.array(ks, float), time=np.array([out[k] for k in ks]), metric=metric)


def time_ais(args):
    case, cfg, seed, max_eval = args
    C = case_obj(case)
    n_eval, t, metric = [], [], np.nan
    for i, B in enumerate(AIS_BUDGETS[AIS_BUDGETS <= max_eval]):
        K = int(B // (cfg["N"] * cfg["n_mcmc"]))
        if K < 1:
            continue
        np.random.seed(seed * 100 + i)
        s = AnnealedImportanceSampling(C.q0, C.logf, C.dim, K=K, batch_size=cfg["N"], n_mcmc=cfg["n_mcmc"],
                                       schedule=cfg["schedule"], beta_min=cfg["beta_min"], proposal=cfg["proposal"])
        t0 = time.perf_counter()
        X, lw = s.run()
        t.append(time.perf_counter() - t0)
        n_eval.append(K * cfg["N"] * cfg["n_mcmc"])
        metric = C.metric(X, np.exp(lw - np.max(lw)))          # metric of the largest budget
    return dict(method="ais_tuned", n_eval=np.array(n_eval, float), time=np.array(t), metric=metric)


def dispatch(job):
    kind, args = job
    return {"midas": time_midas, "kamh": time_kamh, "ais": time_ais}[kind](args)


# ----------------------------------------------------------------------------- plot
STYLE = {"midas_rho0.5": ("MIDAS, $\\ell_n=N^{1/2}$", "#2a78d6", "-"),
         "midas_rho0.6": ("MIDAS, $\\ell_n=N^{0.6}$", "#1baf7a", "-"),
         "midas_rho0.75": ("MIDAS, $\\ell_n=N^{0.75}$", "#eda100", "-"),
         "midas_rhofull": ("MIDAS, $\\ell_n=N$", "#eb6834", "-"),
         "kamh_tuned": ("KAMH (tuned)", "#e34948", "--"),
         "ais_tuned": ("AIS (tuned)", "#4a3aa7", "--")}


def plot(df, case):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for method, g in df.groupby("method", sort=False):
        s = g.groupby("n_eval").time.agg(["mean", "std", "size"]).reset_index()
        h = stats.t.ppf(0.975, np.maximum(s["size"] - 1, 1)) * s["std"].fillna(0) / np.sqrt(s["size"])
        label, col, ls = STYLE.get(method, (method, "0.3", "-"))
        ax.fill_between(s.n_eval, s["mean"] - h, s["mean"] + h, color=col, alpha=0.15, lw=0)
        ax.plot(s.n_eval, s["mean"], ls, color=col, lw=2, marker="o" if method == "ais_tuned" else None, ms=4, label=label)
    ax.set_xlabel("Number of evaluations of the target")
    ax.set_ylabel("Computation time (s)")
    ax.set_title("%s: computation time (mean, 95%% CI)" % case, fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, color="0.9", lw=0.8, which="both")
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%dk" % (v / 1000) if v else "0"))
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    out = os.path.join("figures", "timing_%s" % case)
    fig.savefig(out + ".png", dpi=200)
    fig.savefig(out + ".pdf")
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", nargs="+", default=["mixture"])
    p.add_argument("--subsampling", nargs="+", default=["0.5", "0.6", "0.75", "full"],
                   help="exponents rho (l_n = N^rho) and/or 'full' (l_n = N)")
    p.add_argument("--eta", type=float, default=0.25)
    p.add_argument("--nrep", type=int, default=5)
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--max_eval", type=int, default=BUDGET, help="smaller value for a quick test")
    a = p.parse_args()
    os.makedirs("figures", exist_ok=True)
    cfgs = json.load(open(os.path.join("results", "tuning", "best_configs.json")))
    n_iter = (a.max_eval - BS_INI) // BS
    for case in a.case:
        jobs = [("midas", (case, rho, a.eta, 8000 + r, n_iter)) for rho in a.subsampling for r in range(a.nrep)]
        jobs += [("kamh", (case, cfgs[case]["kamh"], 8500 + r, a.max_eval)) for r in range(a.nrep)]
        jobs += [("ais", (case, cfgs[case]["ais"], 8700 + r, a.max_eval)) for r in range(a.nrep)]
        t0 = time.time()
        if a.cores == 1:
            res = [dispatch(j) for j in jobs]
        else:
            with Pool(a.cores) as pool:
                res = pool.map(dispatch, jobs, chunksize=1)
        rows, final = [], []
        for i, r in enumerate(res):
            for ne, tt in zip(r["n_eval"], r["time"]):
                rows.append(dict(method=r["method"], run=i, n_eval=ne, time=tt))
            final.append(dict(method=r["method"], metric=r["metric"], total_time=r["time"][-1]))
        df = pd.DataFrame(rows)
        os.makedirs(os.path.join("results", case), exist_ok=True)
        df.to_csv(os.path.join("results", case, "timing.csv"), index=False)
        fin = pd.DataFrame(final).groupby("method", sort=False).agg(
            final_metric_mean=("metric", "mean"), final_metric_sd=("metric", "std"),
            total_time_mean_s=("total_time", "mean")).reset_index()
        fin.to_csv(os.path.join("results", case, "timing_final_metric.csv"), index=False)
        out = plot(df, case)
        print("\n=== %s (%.0f s) ===" % (case, time.time() - t0))
        print(fin.to_string(index=False, float_format="%.3f"))
        print("figure:", out + ".png")


if __name__ == "__main__":
    main()
