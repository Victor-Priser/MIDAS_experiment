"""
Peak memory used by MIDAS (several subsample sizes), tuned KAMH and tuned AIS, as a function of the
number of evaluations of the target. Memory is measured with tracemalloc (NumPy arrays included):
the reported value is the peak of the memory allocated by the algorithm itself during the run
(the error metric is not computed). One run per budget is enough: the allocations are deterministic
up to small fluctuations.

Usage
    OMP_NUM_THREADS=1 python3 memory_methods.py                          # mixture, 4 budgets
    OMP_NUM_THREADS=1 python3 memory_methods.py --case coldstart --budget 8000 32000 62000
    OMP_NUM_THREADS=1 python3 memory_methods.py --budget 8000 --subsampling 0.5 half   # quick test
Outputs: results/<case>/memory.csv and figures/memory_<case>.png/.pdf
"""
import argparse
import functools
import json
import os
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from AIS import AIS
from annealed_is import AnnealedImportanceSampling
from cases import BS, BS_INI, get_case, midas_parameters
from classFonction import Function
from kamh import KAMHMulti
from timing_methods import STYLE, bandwidth, n_kernels_rule

MB = 1024.0 ** 2


def peak_of(fn):
    """Peak memory (MB) allocated while running fn()."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / MB, elapsed


def run_midas(C, rho, budget, eta=0.25, seed=0):
    _, gamma, lambd = midas_parameters(C.dim)
    b = Function(functools.partial(bandwidth, rho=rho, d=C.dim))
    np.random.seed(seed)
    a = AIS(lambd, gamma, b, eta, C.q0, C.logf, batch_size=BS, batch_size_ini=BS_INI, d=C.dim, distF=False,
            Indicator="", log_target=True, kernel="gaussian",
            n_kernels=functools.partial(n_kernels_rule, rho=rho), verbose=False)
    a.iteration((budget - BS_INI) // BS)


def run_kamh(C, cfg, budget, seed=0):
    np.random.seed(seed)
    nc = cfg["n_chains"]
    s = KAMHMulti(C.logf, C.dim, n_chains=nc, sigma=cfg["sigma"], gamma=0.2, nu=2.38 / np.sqrt(C.dim),
                  adapt_scale=(cfg["scale"] == "adaptive"))
    s.run(np.asarray(C.q0.simulation(nc), float), budget)


def run_ais(C, cfg, budget, seed=0):
    K = max(1, int(budget // (cfg["N"] * cfg["n_mcmc"])))
    np.random.seed(seed)
    AnnealedImportanceSampling(C.q0, C.logf, C.dim, K=K, batch_size=cfg["N"], n_mcmc=cfg["n_mcmc"],
                               schedule=cfg["schedule"], beta_min=cfg["beta_min"], proposal=cfg["proposal"]).run()


def plot(df, case):
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for method, g in df.groupby("method", sort=False):
        label, col, ls = STYLE.get(method, (method, "0.3", "-"))
        ax.plot(g.n_eval, g.peak_MB, ls, color=col, lw=2, marker="o", ms=4, label=label)
    ax.set_xlabel("Number of evaluations of the target")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_yscale("log")
    ax.set_title("%s: peak memory of the algorithms" % case, fontsize=11)
    ax.grid(True, color="0.9", lw=0.8, which="both")
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%dk" % (v / 1000) if v else "0"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join("figures", "memory_%s" % case)
    fig.savefig(out + ".png", dpi=200)
    fig.savefig(out + ".pdf")
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", nargs="+", default=["mixture"])
    p.add_argument("--budget", nargs="+", type=int, default=[8000, 20000, 40000, 62000])
    p.add_argument("--subsampling", nargs="+", default=["0.5", "0.6", "0.75", "full"])
    a = p.parse_args()
    os.makedirs("figures", exist_ok=True)
    cfgs = json.load(open(os.path.join("results", "tuning", "best_configs.json")))
    for case in a.case:
        C = get_case(case)
        rows = []
        for B in a.budget:
            for rho in a.subsampling:
                m, t = peak_of(lambda: run_midas(C, rho, B))
                rows.append(dict(method="midas_rho%s" % rho, n_eval=B, peak_MB=m, time_s=t))
            m, t = peak_of(lambda: run_kamh(C, cfgs[case]["kamh"], B))
            rows.append(dict(method="kamh_tuned", n_eval=B, peak_MB=m, time_s=t))
            m, t = peak_of(lambda: run_ais(C, cfgs[case]["ais"], B))
            rows.append(dict(method="ais_tuned", n_eval=B, peak_MB=m, time_s=t))
            print("budget %d done" % B, flush=True)
        df = pd.DataFrame(rows)
        os.makedirs(os.path.join("results", case), exist_ok=True)
        df.to_csv(os.path.join("results", case, "memory.csv"), index=False)
        out = plot(df, case)
        print("\n=== %s: peak memory (MB) ===" % case)
        print(df.pivot(index="method", columns="n_eval", values="peak_MB").round(2).to_string())
        print("figure:", out + ".png")


if __name__ == "__main__":
    main()
