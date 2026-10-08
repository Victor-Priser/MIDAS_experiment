"""
Tuning of the competitors (AIS and KAMH) for each case, at the final budget (60 000 / 62 000
evaluations of the target).  Selection on seeds that are NOT used by run_experiment.py.

AIS grid
    batch size N            : 100, 300, 1000
    Metropolis steps / level: 1, 2, 5, 10, 20
    schedule                : geometric (beta_1 = 1e-4, 1e-3, 1e-2), linear
    proposal                : isotropic random walk, population-preconditioned ("diag")
    (the number of levels is K = budget / (N * steps))
KAMH grid (vectorised multi-chain version KAMHMulti)
    proposal scale          : paper (gamma = 0.2, nu = 2.38/sqrt(d), fixed) or adaptive (target acceptance 0.234)
    kernel width sigma      : 5 (paper) or median heuristic
    number of chains        : 1, 10, 50 (budget split between them)
    burn-in                 : first 0, 25 or 50 % of every chain discarded (chosen afterwards)

Usage:  OMP_NUM_THREADS=1 python tune_competitors.py --nrep 10 --nrep_kamh 6
Outputs: results/tuning/ais_tuning.csv, kamh_tuning.csv, best_configs.json
"""
import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from annealed_is import AnnealedImportanceSampling
from cases import CASES, get_case
from kamh import KAMHMulti

AIS_BUDGET = 60000
KAMH_BUDGET = 62000
SCHEDULES = [("geometric", 1e-4), ("geometric", 1e-3), ("geometric", 1e-2), ("linear", 1e-3)]
_CASE = {}


def case_obj(name):
    if name not in _CASE:
        _CASE[name] = get_case(name)
    return _CASE[name]


def ais_job(job):
    case, N, nm, sched, bmin, prop, seed = job
    C = case_obj(case)
    K = AIS_BUDGET // (N * nm)
    np.random.seed(seed)
    s = AnnealedImportanceSampling(C.q0, C.logf, C.dim, K=K, batch_size=N, n_mcmc=nm, schedule=sched,
                                   beta_min=bmin, proposal=prop)
    X, lw = s.run()
    w = np.exp(lw - np.max(lw))
    return dict(case=case, N=N, n_mcmc=nm, schedule=sched, beta_min=bmin, proposal=prop, K=K, seed=seed,
                metric=C.metric(X, w), ess=float(np.sum(w) ** 2 / np.sum(w ** 2)))


def kamh_job(job):
    case, scale, sigma, nc, seed = job
    C = case_obj(case)
    np.random.seed(seed)
    X0 = np.asarray(C.q0.simulation(nc), float)
    s = KAMHMulti(C.logf, C.dim, n_chains=nc, sigma=sigma, gamma=0.2, nu=2.38 / np.sqrt(C.dim),
                  adapt_scale=(scale == "adaptive"))
    chains, _ = s.run(X0, KAMH_BUDGET)
    rows = []
    T = chains.shape[1]
    for burn in [0.0, 0.25, 0.5]:
        S = chains[:, int(burn * T):].reshape(-1, C.dim)
        rows.append(dict(case=case, scale=scale, sigma=str(sigma), n_chains=nc, burn=burn, seed=seed,
                         metric=C.metric(S, np.ones(S.shape[0])), accept=float(np.mean(s.accept_rate))))
    return rows


def best(df, keys, higher):
    g = df.groupby(keys).agg(mean=("metric", "mean"), se=("metric", lambda v: v.std() / np.sqrt(len(v))),
                             n=("metric", "size")).reset_index()
    g = g.sort_values("mean", ascending=not higher)
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", nargs="+", default=CASES)
    p.add_argument("--nrep", type=int, default=10)
    p.add_argument("--nrep_kamh", type=int, default=6)
    p.add_argument("--skip", nargs="*", default=[], help="ais and/or kamh")
    a = p.parse_args()
    os.makedirs("results/tuning", exist_ok=True)
    best_cfg = {}
    if os.path.exists("results/tuning/best_configs.json"):
        best_cfg = json.load(open("results/tuning/best_configs.json"))
    with Pool() as pool:
        if "ais" not in a.skip:
            t0 = time.time()
            grid = list(itertools.product([100, 300, 1000], [1, 2, 5, 10, 20], SCHEDULES, ["isotropic", "diag"]))
            jobs = [(case, N, nm, sc, bm, pr, 90000 + r) for case in a.case for (N, nm, (sc, bm), pr) in grid
                    for r in range(a.nrep)]
            df = pd.DataFrame(pool.map(ais_job, jobs, chunksize=4))
            df.to_csv("results/tuning/ais_tuning.csv", index=False)
            for case in a.case:
                C = case_obj(case)
                g = best(df[df.case == case], ["N", "n_mcmc", "schedule", "beta_min", "proposal"], C.higher_is_better)
                print("\n=== AIS, %s: 8 best configurations ===" % case)
                print(g.head(8).to_string(index=False, float_format="%.3f"))
                b = g.iloc[0]
                best_cfg.setdefault(case, {})["ais"] = dict(N=int(b.N), n_mcmc=int(b.n_mcmc), schedule=b.schedule,
                                                           beta_min=float(b.beta_min), proposal=b.proposal)
            print("AIS tuning: %.0f s" % (time.time() - t0), flush=True)
        if "kamh" not in a.skip:
            t0 = time.time()
            grid = list(itertools.product(["paper", "adaptive"], [5.0, "median"], [1, 10, 50]))
            jobs = [(case, sc, sg, nc, 95000 + r) for case in a.case for (sc, sg, nc) in grid for r in range(a.nrep_kamh)]
            jobs.sort(key=lambda j: j[3])          # single chains (slowest) first
            df = pd.DataFrame([row for rows in pool.map(kamh_job, jobs, chunksize=1) for row in rows])
            df.to_csv("results/tuning/kamh_tuning.csv", index=False)
            for case in a.case:
                C = case_obj(case)
                g = best(df[df.case == case], ["scale", "sigma", "n_chains", "burn"], C.higher_is_better)
                acc = df[df.case == case].groupby(["scale", "sigma", "n_chains"]).accept.mean()
                print("\n=== KAMH, %s: 8 best configurations ===" % case)
                print(g.head(8).to_string(index=False, float_format="%.3f"))
                print("acceptance rates:", acc.round(3).to_dict())
                b = g.iloc[0]
                sig = b.sigma if b.sigma == "median" else float(b.sigma)
                best_cfg.setdefault(case, {})["kamh"] = dict(scale=b.scale, sigma=sig, n_chains=int(b.n_chains),
                                                            burn=float(b.burn))
            print("KAMH tuning: %.0f s" % (time.time() - t0), flush=True)
    json.dump(best_cfg, open("results/tuning/best_configs.json", "w"), indent=2)
    print(json.dumps(best_cfg, indent=2))


if __name__ == "__main__":
    main()
