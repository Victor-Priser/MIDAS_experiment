"""
Run MIDAS, Annealed Importance Sampling (AIS) and KAMH on one or several cases and write
one CSV per (case, method):  results/<case>/<method>.csv
    column "n_eval" = number of evaluations of the unnormalised target,
    columns "rep0", "rep1", ... = metric of each independent run at that budget.

Usage
    python run_experiment.py --case coldstart --method midas --nrep 50
    python run_experiment.py --case all --method all --nrep 50          # everything
    python run_experiment.py --case bayreg --method kamh --nrep 10 --cores 4

Settings (paper, Section 4)
    MIDAS : sqrt(N) Gaussian kernels, initial batch 2000, 200 iterations of 300 particles,
            eta in {1/4, 1/2, 3/4, 1}, b_n / gamma_n / lambda_n from cases.midas_parameters;
            metric with the importance weights w_i = f(X_i)/q_{i-1}(X_i), every 20 iterations.
    AIS   : 300 particles, 20 Metropolis updates per intermediate distribution, geometric
            schedule, K = 1..10 intermediate distributions (budget K * 6000).
    KAMH  : one chain from a q0 draw, sigma = 5, nu = 2.38/sqrt(d), gamma = 0.2,
            subsample of 1000 past states; metric on the whole chain (uniform weights).
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from AIS import AIS
from annealed_is import AnnealedImportanceSampling
from cases import BS, BS_INI, CASES, CHECKPOINTS, N_ITER, get_case
from kamh import KAMH, KAMHMulti
import json

ETAS = [0.25, 0.5, 0.75, 1.0]
AIS_K = list(range(1, 11))
AIS_BATCH, AIS_MCMC = 300, 20
_CASE = {}


def case_obj(name):
    if name not in _CASE:
        _CASE[name] = get_case(name)
    return _CASE[name]


# ----------------------------------------------------------------------------- one run of each method
def run_midas(args):
    case_name, eta, seed = args
    C = case_obj(case_name)
    np.random.seed(seed)
    a = AIS(C.lambd, C.gamma, C.b, eta, C.q0, C.logf, batch_size=BS, batch_size_ini=BS_INI, d=C.dim,
            distF=False, Indicator="", log_target=True, kernel="gaussian", n_kernels="sqrt", verbose=False)
    a.iteration(N_ITER)
    out = []
    for N in CHECKPOINTS:
        lw = a.newWeight[:N]                      # log importance weights log f - log q
        w = np.exp(lw - np.max(lw))
        out.append(C.metric(a.X[:N], w))
    return np.array(out)


def run_ais(args):
    case_name, seed = args
    C = case_obj(case_name)
    out = []
    for K in AIS_K:
        np.random.seed(seed * 100 + K)
        s = AnnealedImportanceSampling(C.q0, C.logf, C.dim, K=K, batch_size=AIS_BATCH, n_mcmc=AIS_MCMC)
        X, lw = s.run()
        out.append(C.metric(X, np.exp(lw - np.max(lw))))
    return np.array(out)


def run_kamh(args):
    case_name, seed = args
    C = case_obj(case_name)
    np.random.seed(seed)
    x0 = np.asarray(C.q0.simulation(1), float)[0]
    s = KAMH(C.logf, C.dim, sigma=5.0, nu=2.38 / np.sqrt(C.dim), gamma=0.2, n_sub=1000, adapt_exponent=0.5)
    _, res = s.run(x0, int(CHECKPOINTS[-1]), checkpoints=CHECKPOINTS,
                   callback=lambda X: C.metric(X, np.ones(X.shape[0])))
    return np.array([res[int(t)] for t in CHECKPOINTS])


# ----------------------------------------------------------------------------- tuned competitors
AIS_TUNED_BUDGETS = np.arange(6000, 60001, 6000)


def tuned_config(case_name, method):
    path = os.path.join("results", "tuning", "best_configs.json")
    return json.load(open(path))[case_name][method]


def run_ais_tuned(args):
    """AIS with the configuration chosen by tune_competitors.py; one independent run per budget,
    K = budget // (N * n_mcmc) (NaN when the budget is smaller than one level)."""
    case_name, seed, cfg = args
    C = case_obj(case_name)
    out = []
    for i, B in enumerate(AIS_TUNED_BUDGETS):
        K = int(B // (cfg["N"] * cfg["n_mcmc"]))
        if K < 1:
            out.append(np.nan)
            continue
        np.random.seed(seed * 100 + i)
        s = AnnealedImportanceSampling(C.q0, C.logf, C.dim, K=K, batch_size=cfg["N"], n_mcmc=cfg["n_mcmc"],
                                       schedule=cfg["schedule"], beta_min=cfg["beta_min"], proposal=cfg["proposal"])
        X, lw = s.run()
        out.append(C.metric(X, np.exp(lw - np.max(lw))))
    return np.array(out)


def run_kamh_tuned(args):
    """KAMHMulti with the configuration chosen by tune_competitors.py; the metric uses the states of all
    chains after discarding the first `burn` fraction of each chain."""
    case_name, seed, cfg = args
    C = case_obj(case_name)
    np.random.seed(seed)
    nc = cfg["n_chains"]
    X0 = np.asarray(C.q0.simulation(nc), float)
    s = KAMHMulti(C.logf, C.dim, n_chains=nc, sigma=cfg["sigma"], gamma=0.2, nu=2.38 / np.sqrt(C.dim),
                  adapt_scale=(cfg["scale"] == "adaptive"))

    def cb(ch):
        S = ch[:, int(cfg["burn"] * ch.shape[1]):].reshape(-1, C.dim)
        return C.metric(S, np.ones(S.shape[0]))
    cps = [int(c) for c in CHECKPOINTS if c // nc >= 2]
    _, res = s.run(X0, int(CHECKPOINTS[-1]), checkpoints=cps, callback=cb)
    return np.array([res.get(int(t), np.nan) for t in CHECKPOINTS])


# ----------------------------------------------------------------------------- driver
def save(case_name, method, n_eval, R):
    os.makedirs(os.path.join("results", case_name), exist_ok=True)
    df = pd.DataFrame(np.column_stack([n_eval, R]), columns=["n_eval"] + ["rep%d" % i for i in range(R.shape[1])])
    path = os.path.join("results", case_name, method + ".csv")
    df.to_csv(path, index=False)
    print("  written", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", nargs="+", default=["all"], help="coldstart mixture anisotropic bayreg | all")
    p.add_argument("--method", nargs="+", default=["all"], help="midas ais kamh ais_tuned kamh_tuned | all")
    p.add_argument("--eta", nargs="+", type=float, default=ETAS)
    p.add_argument("--nrep", type=int, default=50)
    p.add_argument("--cores", type=int, default=None)
    a = p.parse_args()
    cases = CASES if "all" in a.case else a.case
    methods = ["midas", "ais", "kamh", "ais_tuned", "kamh_tuned"] if "all" in a.method else a.method
    case_offset = {c: 100000 * i for i, c in enumerate(CASES)}
    with Pool(a.cores) as pool:
        for case_name in cases:
            for method in methods:
                t0 = time.time()
                off = case_offset[case_name]
                if method == "midas":
                    for eta in a.eta:
                        jobs = [(case_name, eta, off + 1000 + r) for r in range(a.nrep)]
                        R = np.column_stack(pool.map(run_midas, jobs, chunksize=1))
                        save(case_name, "midas_eta%g" % eta, CHECKPOINTS, R)
                elif method == "ais":
                    R = np.column_stack(pool.map(run_ais, [(case_name, off + 2000 + r) for r in range(a.nrep)], chunksize=1))
                    save(case_name, "ais", np.array(AIS_K) * AIS_BATCH * AIS_MCMC, R)
                elif method == "kamh":
                    R = np.column_stack(pool.map(run_kamh, [(case_name, off + 3000 + r) for r in range(a.nrep)], chunksize=1))
                    save(case_name, "kamh", CHECKPOINTS, R)
                elif method == "ais_tuned":
                    cfg = tuned_config(case_name, "ais")
                    R = np.column_stack(pool.map(run_ais_tuned, [(case_name, off + 4000 + r, cfg) for r in range(a.nrep)], chunksize=1))
                    save(case_name, "ais_tuned", AIS_TUNED_BUDGETS, R)
                elif method == "kamh_tuned":
                    cfg = tuned_config(case_name, "kamh")
                    R = np.column_stack(pool.map(run_kamh_tuned, [(case_name, off + 5000 + r, cfg) for r in range(a.nrep)], chunksize=1))
                    save(case_name, "kamh_tuned", CHECKPOINTS, R)
                else:
                    raise ValueError(method)
                print("%s / %s : %.0f s" % (case_name, method, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
