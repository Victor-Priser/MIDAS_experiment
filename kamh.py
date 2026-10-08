"""
Kernel Adaptive Metropolis-Hastings (KAMH, "Kameleon"),
D. Sejdinovic, H. Strathmann, M. Lomeli Garcia, C. Andrieu, A. Gretton,
"Kernel Adaptive Metropolis-Hastings", ICML 2014.
Implementation follows the authors' reference code (github.com/karlnapf/kameleon-mcmc,
class Kameleon): Gaussian kernel, centred gradient matrix, Gaussian proposal with a
covariance that depends on the current state, and a full Metropolis-Hastings ratio.

Proposal at the current state y, given a subsample z = (z_1, ..., z_n) of the chain history:
    x* ~ N(y, gamma^2 I + nu^2 M_y^T H M_y)
    k(x, z) = exp(-||x - z||^2 / (2 sigma^2))          (Gaussian kernel, "covariance sigma^2 I")
    M_y     = 2 [grad_x k(x, z_1) |_{x=y}, ..., grad_x k(x, z_n) |_{x=y}]^T     (n x d)
    H       = I_n - 1/n 1 1^T                           (centring matrix)
Acceptance probability:
    min(1, f(x*) q(y | x*) / (f(y) q(x* | y)))  -- the proposal is not symmetric.
Adaptation (vanishing): at step t, with probability p_t = t^(-adapt_exponent), z is
redrawn as a uniform subsample (size n_sub) of the chain history; before n_burn steps
(or while z is empty) the proposal is the random walk N(y, gamma^2 I).

Paper settings (v10, Section 4.2): sigma = 5, nu = 2.38/sqrt(d), gamma = 0.2;
one step = one evaluation of the unnormalised target.

Optional (adapt_scale=True, NOT part of the paper settings): the whole proposal covariance
is multiplied by exp(2 s_t), with the vanishing Robbins-Monro update
    s_{t+1} = s_t + t^(-adapt_exponent) (alpha_t - target_accept)
(alpha_t = acceptance probability), as in Andrieu & Thoms (2008), "A tutorial on adaptive
MCMC", Algorithm 4. Useful when gamma = 0.2 is too large for the scale of the target.
"""
import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

from classFonction import Function


class KAMH:
    def __init__(self, logf, d, sigma=5.0, nu=None, gamma=0.2, n_sub=1000, adapt_exponent=0.5,
                 n_burn=0, adapt_scale=False, target_accept=0.234):
        self.logf, self.d = logf, d
        self.sigma2 = sigma ** 2
        self.nu2 = (2.38 / np.sqrt(d) if nu is None else nu) ** 2
        self.gamma2 = gamma ** 2
        self.n_sub, self.adapt_exponent, self.n_burn = n_sub, adapt_exponent, n_burn
        self.Z = None
        self.adapt_scale, self.target_accept = adapt_scale, target_accept
        self.log_scale = 0.0

    # ----- target -----
    def _logf(self, x):
        if isinstance(self.logf, Function):
            return float(self.logf.evalAIS(x[None, :])[0])
        return float(self.logf(x))

    # ----- proposal covariance at y: Cholesky factor and log-determinant -----
    def _proposal(self, y):
        C = self.gamma2 * np.eye(self.d)
        if self.Z is not None:
            diff = self.Z - y                                            # n x d
            k = np.exp(-np.sum(diff ** 2, axis=1) / (2 * self.sigma2))   # n
            M = 2.0 * (k[:, None] * diff) / self.sigma2                  # rows: 2 grad_x k(x, z_i) at x = y
            Mc = M - M.mean(axis=0)                                      # H M
            C += self.nu2 * (Mc.T @ Mc)                                  # M^T H M  (H idempotent)
        L = np.linalg.cholesky(C)
        return L, 2.0 * np.sum(np.log(np.diag(L)))

    @staticmethod
    def _log_gauss(x, mean, L, logdet):
        u = solve_triangular(L, x - mean, lower=True)
        return -0.5 * (u @ u) - 0.5 * logdet

    def run(self, x0, n_steps, checkpoints=None, callback=None):
        """Run the chain for n_steps target evaluations (the evaluation at x0 is the first one).
        callback(chain[:t]) is called at every t in checkpoints."""
        d = self.d
        chain = np.empty((n_steps, d))
        x = np.asarray(x0, dtype=float).reshape(d)
        lfx = self._logf(x)
        chain[0] = x
        Lx, ldx = self._proposal(x)
        checkpoints = set() if checkpoints is None else set(int(c) for c in checkpoints)
        out = {}
        n_acc = 0
        for t in range(1, n_steps):
            # vanishing adaptation of the subsample z
            if t >= max(self.n_burn, 1) and np.random.uniform() < t ** (-self.adapt_exponent):
                idx = np.random.choice(t, size=min(self.n_sub, t), replace=False)
                self.Z = chain[idx].copy()
                Lx, ldx = self._proposal(x)
            sc = np.exp(self.log_scale)                    # global scale (1 if adapt_scale=False)
            y = x + sc * (Lx @ np.random.standard_normal(d))
            lfy = self._logf(y)
            Ly, ldy = self._proposal(y)
            log_ratio = ((lfy - lfx) + self._log_gauss(x, y, sc * Ly, ldy + 2 * d * self.log_scale)
                         - self._log_gauss(y, x, sc * Lx, ldx + 2 * d * self.log_scale))
            if np.log(np.random.uniform()) < log_ratio:
                x, lfx, Lx, ldx = y, lfy, Ly, ldy
                n_acc += 1
            if self.adapt_scale:
                alpha = np.exp(min(0.0, log_ratio)) if np.isfinite(log_ratio) else 0.0
                self.log_scale += t ** (-self.adapt_exponent) * (alpha - self.target_accept)
            chain[t] = x
            if (t + 1) in checkpoints and callback is not None:
                out[t + 1] = callback(chain[:t + 1])
        if n_steps in checkpoints and callback is not None and n_steps not in out:
            out[n_steps] = callback(chain)
        self.chain, self.accept_rate = chain, n_acc / max(1, n_steps - 1)
        return chain, out


# =============================================================================
#  Tunable, vectorised version: several chains run in lockstep
# =============================================================================
class KAMHMulti:
    """
    C independent KAMH chains run in lockstep (the budget is split between them); same
    proposal as KAMH above, with three optional improvements used for tuning:
      sigma = "median"  : kernel width set by the median heuristic on the subsample z
                          (recomputed each time z is redrawn), instead of a fixed value;
      adapt_scale=True  : per-chain global scale s_c of the proposal covariance adapted by the
                          vanishing Robbins-Monro rule log s += t^(-adapt_exponent) (alpha - 0.234)
                          (Andrieu & Thoms 2008, Alg. 4);
      n_chains > 1      : chains started from independent q0 draws (helps multimodal targets).
    The subsample z of each chain is a uniform subsample (size <= n_sub) of its own history,
    redrawn for all chains at the same random times (probability t^(-adapt_exponent)).
    """
    def __init__(self, logf, d, n_chains=1, sigma=5.0, nu=None, gamma=0.2, n_sub=1000,
                 adapt_exponent=0.5, adapt_scale=False, target_accept=0.234):
        self.logf, self.d, self.C = logf, d, n_chains
        self.sigma_mode = sigma
        self.nu2 = (2.38 / np.sqrt(d) if nu is None else nu) ** 2
        self.gamma2 = gamma ** 2
        self.n_sub, self.adapt_exponent = n_sub, adapt_exponent
        self.adapt_scale, self.target_accept = adapt_scale, target_accept

    def _lf(self, X):
        if isinstance(self.logf, Function):
            return self.logf.evalAIS(X)
        return np.asarray(self.logf(X), dtype=float)

    def _chol(self, X):
        """Cholesky factors (C, d, d) of gamma^2 I + nu^2 M^T H M at the states X (C, d)."""
        C, d = X.shape
        cov = np.broadcast_to(self.gamma2 * np.eye(d), (C, d, d)).copy()
        if self.Z is not None:
            diff = self.Z - X[:, None, :]                                     # (C, m, d)
            k = np.exp(-np.sum(diff ** 2, axis=2) / (2 * self.sig2[:, None]))  # (C, m)
            M = 2.0 * k[:, :, None] * diff / self.sig2[:, None, None]
            Mc = M - M.mean(axis=1, keepdims=True)
            cov += self.nu2 * np.einsum("cmi,cmj->cij", Mc, Mc)
        L = np.linalg.cholesky(cov)
        logdet = 2.0 * np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1)
        return L, logdet

    @staticmethod
    def _log_gauss(x, mean, L, logdet):
        u = np.linalg.solve(L, (x - mean)[:, :, None])[:, :, 0]
        return -0.5 * np.sum(u ** 2, axis=1) - 0.5 * logdet

    def _update_sigma(self):
        if self.sigma_mode == "median":
            m = self.Z.shape[1]
            idx = np.random.choice(m, size=min(m, 200), replace=False)
            S = self.Z[:, idx]
            D = np.sum((S[:, :, None, :] - S[:, None, :, :]) ** 2, axis=3)    # (C, s, s)
            iu = np.triu_indices(S.shape[1], 1)
            med = np.median(D[:, iu[0], iu[1]], axis=1) if len(iu[0]) else np.ones(self.C)
            self.sig2 = np.maximum(med, 1e-12)            # sigma = median pairwise distance
        else:
            self.sig2 = np.full(self.C, float(self.sigma_mode) ** 2)

    def run(self, X0, n_eval, checkpoints=None, callback=None):
        """n_eval evaluations of f in total (n_eval // C steps per chain, the start included).
        callback(samples) is called at each budget in checkpoints with the states of all chains so far."""
        C, d = self.C, self.d
        T = int(n_eval // C)
        chains = np.empty((C, T, d))
        X = np.asarray(X0, dtype=float).reshape(C, d).copy()
        lf = self._lf(X)
        chains[:, 0] = X
        self.Z = None
        self.sig2 = np.full(C, 25.0)
        log_s = np.zeros(C)
        L, ld = self._chol(X)
        cps = {} if checkpoints is None else {int(c) // C: int(c) for c in checkpoints}
        out, n_acc = {}, np.zeros(C)
        for t in range(1, T):
            if np.random.uniform() < t ** (-self.adapt_exponent):
                m = min(self.n_sub, t)
                idx = np.stack([np.random.choice(t, size=m, replace=False) for _ in range(C)])
                self.Z = chains[np.arange(C)[:, None], idx]
                self._update_sigma()
                L, ld = self._chol(X)
            s = np.exp(log_s)
            Y = X + s[:, None] * np.einsum("cij,cj->ci", L, np.random.standard_normal((C, d)))
            lfY = self._lf(Y)
            LY, ldY = self._chol(Y)
            lr = (lfY - lf) + self._log_gauss(X, Y, s[:, None, None] * LY, ldY + 2 * d * log_s) \
                - self._log_gauss(Y, X, s[:, None, None] * L, ld + 2 * d * log_s)
            lr = np.where(np.isfinite(lr), lr, -np.inf)
            acc = np.log(np.random.uniform(size=C)) < lr
            X[acc], lf[acc], L[acc], ld[acc] = Y[acc], lfY[acc], LY[acc], ldY[acc]
            n_acc += acc
            if self.adapt_scale:
                log_s += t ** (-self.adapt_exponent) * (np.exp(np.minimum(0.0, lr)) - self.target_accept)
            chains[:, t] = X
            if (t + 1) in cps and callback is not None:
                out[cps[t + 1]] = callback(chains[:, :t + 1])
        if T in cps and callback is not None and cps[T] not in out:
            out[cps[T]] = callback(chains)
        self.chains, self.accept_rate, self.scale = chains, n_acc / max(1, T - 1), np.exp(log_s)
        return chains, out
