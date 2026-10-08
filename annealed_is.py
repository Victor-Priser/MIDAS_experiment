"""
Annealed Importance Sampling (AIS), R. M. Neal, "Annealed importance sampling",
Statistics and Computing 11 (2001) 125-139.

Structure inspired by the AIS implementation of the FAB code
(Midgley et al., "Flow Annealed Importance Sampling Bootstrap", fab/sampling_methods/ais.py):
a batch of independent particles is pushed through a sequence of intermediate
distributions with a geometric (or linear) schedule, using Metropolis transitions
whose step size is adapted at each intermediate distribution.

Intermediate distributions (Neal, eq. (3)):
    pi_k(x)  propto  q0(x)^(1 - beta_k) * f(x)^beta_k ,   0 = beta_0 < beta_1 < ... < beta_K = 1.
Algorithm for each particle:
    x_0 ~ q0,   log w = 0
    for k = 1..K:
        log w += (beta_k - beta_{k-1}) * (log f(x_{k-1}) - log q0(x_{k-1}))
        x_k    = n_mcmc Metropolis steps leaving pi_k invariant, started at x_{k-1}
Then sum_i w_i h(x_i) / sum_i w_i estimates E_f[h], and mean(w) estimates the
normalising constant of f.

Budget: K * batch_size * n_mcmc evaluations of f (one per Metropolis proposal), as in
the paper (Section 4.2); the batch_size initial evaluations are not counted.
"""
import numpy as np

from classFonction import Function


def _logdens(F, X):
    """Vectorised log-density: F is a Function (with `vect` for speed) or a callable on [n, d]."""
    if isinstance(F, Function):
        return F.evalAIS(X)
    return np.asarray(F(X), dtype=float).reshape(X.shape[0])


def beta_schedule(K, kind="geometric", beta_min=1e-3):
    """beta_1 < ... < beta_K = 1 (beta_0 = 0 is implicit)."""
    if K == 1:
        return np.ones(1)
    if kind == "geometric":
        return np.geomspace(beta_min, 1.0, K)
    if kind == "linear":
        return np.linspace(1.0 / K, 1.0, K)
    raise ValueError("kind must be 'geometric' or 'linear'")


class AnnealedImportanceSampling:
    """
    q0      : initial distribution (object with .simulation(n) and .logf, e.g. classProba.Student)
    logf    : unnormalised log target (Function with `vect`, or callable on arrays [n, d])
    K       : number of intermediate distributions
    n_mcmc  : Metropolis updates per intermediate distribution (paper: 20)
    step_size : initial isotropic random-walk step; default 2.38/sqrt(d) * (mean std of the q0 particles)
    adapt   : adapt the step size at each Metropolis step towards target_accept
    proposal: "isotropic" -> Y = X + s * eps
              "diag"      -> Y = X + s * std(X) * eps, std(X) = per-coordinate standard deviation of
                             the current particle cloud (population-preconditioned random walk)
    """
    def __init__(self, q0, logf, d, K=10, batch_size=300, n_mcmc=20, schedule="geometric",
                 beta_min=1e-3, step_size=None, adapt=True, target_accept=0.234, adapt_rate=1.0,
                 proposal="isotropic"):
        self.q0, self.logf, self.d = q0, logf, d
        self.K, self.bs, self.n_mcmc = K, batch_size, n_mcmc
        self.betas = beta_schedule(K, schedule, beta_min)
        self.step_size0 = step_size
        self.adapt, self.target_accept, self.adapt_rate = adapt, target_accept, adapt_rate
        if proposal not in ("isotropic", "diag"):
            raise ValueError("proposal must be 'isotropic' or 'diag'")
        self.proposal = proposal

    def run(self):
        d, N = self.d, self.bs
        X = np.asarray(self.q0.simulation(N), dtype=float).reshape(N, d)
        lq0 = _logdens(self.q0.logf, X)
        lf = _logdens(self.logf, X)
        logw = np.zeros(N)
        if self.step_size0 is not None:
            s = self.step_size0
        elif self.proposal == "diag":
            s = 2.38 / np.sqrt(d)
        else:
            s = 2.38 / np.sqrt(d) * np.mean(np.std(X, axis=0))
        self.n_eval_f = 0
        self.accept_rates = np.zeros(self.K)
        beta_prev = 0.0
        for k, beta in enumerate(self.betas):
            # importance weight update (uses the current, already evaluated, state)
            with np.errstate(invalid="ignore"):
                logw += (beta - beta_prev) * (lf - lq0)
            # Metropolis transitions leaving pi_k invariant
            acc_k = 0.0
            for _ in range(self.n_mcmc):
                scale = s * np.std(X, axis=0) if self.proposal == "diag" else s
                Y = X + scale * np.random.standard_normal((N, d))
                lq0_Y = _logdens(self.q0.logf, Y)
                lf_Y = _logdens(self.logf, Y)
                self.n_eval_f += N
                with np.errstate(invalid="ignore"):
                    log_ratio = (1 - beta) * (lq0_Y - lq0) + beta * (lf_Y - lf)
                accept = np.log(np.random.uniform(size=N)) < log_ratio
                X[accept], lq0[accept], lf[accept] = Y[accept], lq0_Y[accept], lf_Y[accept]
                rate = np.mean(accept)
                acc_k += rate / self.n_mcmc
                if self.adapt:   # Robbins-Monro on log(step size), shared by the batch
                    s *= np.exp(self.adapt_rate * (rate - self.target_accept))
            self.accept_rates[k] = acc_k
            beta_prev = beta
        self.X, self.logw, self.step_size = X, logw, s
        return X, logw

    def weights(self):
        w = np.exp(self.logw - np.max(self.logw))
        return w / np.sum(w)

    def log_normalising_constant(self):
        m = np.max(self.logw)
        return m + np.log(np.mean(np.exp(self.logw - m)))
