# MIDAS / AIS / KAMH experiments

Code for the numerical experiments of the MIDAS paper (Section 4): MIDAS (subMIDAS with
sqrt(N) kernels), Annealed Importance Sampling (Neal, 2001) and Kernel Adaptive
Metropolis-Hastings (Sejdinovic et al., 2014) on four cases.

## Files

| file | content |
|---|---|
| `AIS.py` | MIDAS algorithm (class `AIS`), kernels (Gaussian, rectangular, triangular, Epanechnikov) |
| `annealed_is.py` | Annealed Importance Sampling |
| `kamh.py` | Kernel Adaptive Metropolis-Hastings |
| `classProba.py`, `classFonction.py` | distributions and vectorised functions |
| `cases.py` | the four cases (initial distribution, target, metric, MIDAS parameters) |
| `run_experiment.py` | runs the methods and writes `results/<case>/<method>.csv` |
| `plot_results.py` | reads all the CSV files and plots the curves with 95% confidence intervals |
| `datasets/waveform.csv` | data of the Bayesian logistic regression (400 train / 4600 test) |

## Cases (`cases.py`)

- `coldstart`: target N(5/sqrt(d) 1, 0.4²/d I), q0 = N(0, 5/d I), d = 16
- `mixture`: target 0.2 N(m, 0.4²/d I) + 0.8 N(−m, 0.4²/d I) (weights `MIX_WEIGHTS` in `cases.py`; ½, ½ in the paper), m = 1/(2 sqrt(d)) 1, q0 = Student(0, 5/d I, 2 df), d = 16
- `anisotropic`: same mixture (same weights) with covariance 0.4²/d Diag(10, 1, …, 1)
- `bayreg`: Bayesian logistic regression on waveform, d = 22, prior β ~ Gamma(1, rate 0.01), w | β ~ N(0, I/β)

Metric: log sliced Wasserstein distance to the target (synthetic cases), test accuracy of the
posterior predictive rule (bayreg). Budget: 62 000 evaluations of the unnormalised target.

## Parameters

- MIDAS: initial batch 2000, 200 iterations of 300 particles, sqrt(N) Gaussian kernels,
  η ∈ {1/4, 1/2, 3/4, 1}, b_n = 0.5/sqrt(d) (sqrt(300 n) + 1)^(−1/(4+d)), γ_n = 1/(n+10),
  λ_n = 1/log(300 n + 10) (0.5 for n < 10); metric with the importance weights f/q, every 20 iterations.
- AIS: 300 particles, 20 Metropolis updates per intermediate distribution (adaptive step size,
  target acceptance 0.234), geometric schedule, K = 1, …, 10 (budget K × 6000).
- KAMH: one chain started from a q0 draw, σ = 5, ν = 2.38/sqrt(d), γ = 0.2, subsample of 1000 past states.

## Usage

```
pip install numpy scipy pandas matplotlib pot
python run_experiment.py --case all --method all --nrep 50     # every case and method
python run_experiment.py --case mixture --method midas --nrep 50 --eta 0.25 1
python plot_results.py                                           # figures/ and results/summary.csv
```
Each CSV has a column `n_eval` (number of evaluations of the target) and one column per run.

## Tuned competitors (`tune_competitors.py`)

`python tune_competitors.py --nrep 10 --nrep_kamh 4` selects, for each case, the AIS and KAMH
configuration with the best metric at the final budget (seeds disjoint from those of
`run_experiment.py`) and writes `results/tuning/best_configs.json`. Then
`python run_experiment.py --case all --method ais_tuned kamh_tuned --nrep 30` produces
`results/<case>/ais_tuned.csv` and `kamh_tuned.csv`, and `plot_results.py` shows both the paper
settings (dotted) and the tuned versions (dashed).

- AIS grid: batch size N ∈ {30, 50, 100, 300, 1000}, Metropolis steps per level ∈ {1, 2, 5, 10, 20},
  geometric schedule with β_1 ∈ {1e-4, 1e-3, 1e-2, 3e-2, 1e-1} or linear, isotropic or
  population-preconditioned ("diag") random walk; K = budget / (N × steps).
- KAMH grid (`KAMHMulti`, chains run in lockstep): fixed (paper) or adaptive global proposal scale
  (Robbins-Monro, target acceptance 0.234), kernel width σ = 5 or median heuristic,
  1 / 10 / 50 / 100 chains started from q0 (budget split between them), burn-in 0 / 25 / 50 %.
