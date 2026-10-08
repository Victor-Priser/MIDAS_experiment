import numpy as np
import math as m
import scipy.stats as sc
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
import ot
from classProba import *
from classFonction import *

import time


# ===========================================================================
#  Evaluation helpers
# ===========================================================================

def _evaluate(F, X):
    '''Evaluate F (a Function, or a point-wise callable) on the rows of X.'''
    if isinstance(F, Function):
        return F.evalAIS(X)
    return np.array([F(x) for x in X], dtype=float)


def _log_density(proba, X):
    '''log q(X) for a ProbaContinue (uses logf when available).'''
    logf = getattr(proba, "logf", None)
    if logf is not None:
        return _evaluate(logf, X)
    with np.errstate(divide="ignore"):
        return np.log(_evaluate(proba.f, X))


# ===========================================================================
#  Kernels
# ===========================================================================
#  A kernel is a mixture  sum_j w_j K_{h_j}(x - c_j)  built as
#       Kernel(weights, parameters, d=d)
#  where parameters has shape [m, d, 2]:
#       parameters[:, :, 0] = centers c_j (resampled particles)
#       parameters[:, :, 1] = bandwidth b (per coordinate)
#  K is a product kernel, standardised so that each component has variance
#  b^2 per coordinate (like K_b(x) = b^{-d} K(x/b) with variance b^2 I_d in
#  the paper): only the SHAPE changes from one kernel to another.
#
#  To add a kernel: subclass KernelMixture (or CompactKernelMixture), define
#  var_std (variance of the standard 1D kernel), _logK1 (log of the standard
#  kernel, summed over coordinates), _std (sampling from the standard kernel),
#  and register it in KERNELS.
# ===========================================================================


class KernelMixture(ProbaContinue):
    var_std = 1.0          # variance of the standard 1D kernel
    block_size = 2**22     # max number of elements of intermediate arrays [n, m, d]

    def __init__(self, weights, parameters, d=1):
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / np.sum(self.weights)
        self.centers = np.asarray(parameters[:, :, 0], dtype=float)
        self.b = np.asarray(parameters[:, :, 1], dtype=float)
        self.h = self.b / np.sqrt(self.var_std)      # scale such that the variance is b^2
        self.ih = 1.0 / self.h
        self.d = d
        self.parameters = parameters
        self.esp = self.weights @ self.centers
        # constant: log w_j - sum_k log h_jk
        self.logc = np.log(self.weights) - np.sum(np.log(self.h), axis=1)
        self._prepare()
        self.f = Function(lambda x: self.pdf(np.atleast_2d(x))[0], vect=self.pdf)
        self.logf = Function(lambda x: self.logpdf(np.atleast_2d(x))[0], vect=self.logpdf)

    def _prepare(self):
        pass

    def _logK(self, X):
        '''log prod_k K((x_ik - c_jk)/h_jk), array of shape [n, m].'''
        U = np.abs(X[:, None, :] - self.centers[None]) * self.ih[None]
        return self._logK1(U)

    def logpdf(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n, mm = X.shape[0], self.centers.shape[0]
        step = max(1, self.block_size // max(1, mm * self.d))
        out = np.empty(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            for s in range(0, n, step):
                out[s:s + step] = logsumexp(self._logK(X[s:s + step]) + self.logc, axis=1)
        return out

    def pdf(self, X):
        return np.exp(self.logpdf(X))

    def simulation(self, n):
        idx = np.random.choice(self.weights.shape[0], size=n, p=self.weights)
        return self.centers[idx] + self.h[idx] * self._std((n, self.d))


class GaussianKernelMixture(KernelMixture):
    '''K(u) = (2 pi)^{-1/2} exp(-u^2/2), variance 1.'''
    var_std = 1.0

    def _prepare(self):
        # ||(x - c)/h||^2 = x^2 . (1/h^2) - 2 x . (c/h^2) + ||c/h||^2  -> matrix products (BLAS)
        self.ih2 = self.ih ** 2
        self.cih2 = self.centers * self.ih2
        self.c2 = np.sum(self.centers ** 2 * self.ih2, axis=1)
        self.cst = -0.5 * self.d * np.log(2 * np.pi)

    def _logK(self, X):
        Q = (X ** 2) @ self.ih2.T - 2.0 * (X @ self.cih2.T) + self.c2
        return self.cst - 0.5 * np.maximum(Q, 0.0)

    def _std(self, size):
        return np.random.standard_normal(size)


class CompactKernelMixture(KernelMixture):
    '''
    Product kernels supported on [-1, 1]^d. When the bandwidth of a component is
    the same for every coordinate (which is the case in AIS), the kernel is only
    computed for the (point, center) pairs such that ||x - c||_inf <= h
    (Chebyshev distance computed in C by cdist): the cost is proportional to the
    number of "active" pairs instead of n * m * d.
    '''
    def _prepare(self):
        self.iso = np.allclose(self.h, self.h[:, :1])
        self.ih_iso = self.ih[:, 0]
        self.logc_max = np.max(self.logc)
        self.c_rel = np.exp(self.logc - self.logc_max)

    def _k1(self, U):
        '''standard 1D kernel evaluated at |u| <= 1 (array [p, d]), product over coordinates.'''
        raise NotImplementedError

    def logpdf(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if not self.iso:
            return KernelMixture.logpdf(self, X)
        n, mm = X.shape[0], self.centers.shape[0]
        step = max(1, self.block_size // max(1, mm))
        dens = np.empty(n)
        for s in range(0, n, step):
            Xs = X[s:s + step]
            i, j = np.nonzero(cdist(Xs, self.centers, "chebyshev") * self.ih_iso[None, :] <= 1.0)
            if i.size:
                U = np.abs(Xs[i] - self.centers[j]) * self.ih_iso[j, None]
                val = self.c_rel[j] * self._k1(U)
                dens[s:s + step] = np.bincount(i, weights=val, minlength=Xs.shape[0])
            else:
                dens[s:s + step] = 0.0
        with np.errstate(divide="ignore"):
            return np.log(dens) + self.logc_max


class RectangularKernelMixture(CompactKernelMixture):
    '''K(u) = 1/2 on [-1, 1], variance 1/3  (half-width sqrt(3) b).'''
    var_std = 1.0 / 3.0

    def _k1(self, U):
        return np.full(U.shape[0], 0.5 ** self.d)

    def _logK1(self, U):
        return np.where(np.all(U <= 1.0, axis=2), -self.d * np.log(2.0), -np.inf)

    def _std(self, size):
        return np.random.uniform(-1.0, 1.0, size)


class TriangularKernelMixture(CompactKernelMixture):
    '''K(u) = (1 - |u|)_+ on [-1, 1], variance 1/6  (half-width sqrt(6) b).'''
    var_std = 1.0 / 6.0

    def _k1(self, U):
        return np.prod(1.0 - U, axis=1)

    def _logK1(self, U):
        return np.sum(np.log(np.maximum(1.0 - U, 0.0)), axis=2)

    def _std(self, size):
        # sum of two uniforms on [0, 1] minus 1: triangular distribution on [-1, 1]
        return np.random.uniform(0.0, 1.0, size) + np.random.uniform(0.0, 1.0, size) - 1.0


class EpanechnikovKernelMixture(CompactKernelMixture):
    '''K(u) = 3/4 (1 - u^2)_+ on [-1, 1], variance 1/5  (half-width sqrt(5) b).'''
    var_std = 1.0 / 5.0

    def _k1(self, U):
        return np.prod(0.75 * (1.0 - U ** 2), axis=1)

    def _logK1(self, U):
        return np.sum(np.log(0.75 * np.maximum(1.0 - U ** 2, 0.0)), axis=2)

    def _std(self, size):
        # Devroye's algorithm
        u1, u2, u3 = (np.random.uniform(-1.0, 1.0, size) for _ in range(3))
        return np.where((np.abs(u3) >= np.abs(u2)) & (np.abs(u3) >= np.abs(u1)), u2, u3)


KERNELS = {
    "gaussian": GaussianKernelMixture,
    "rectangular": RectangularKernelMixture,
    "uniform": RectangularKernelMixture,
    "triangular": TriangularKernelMixture,
    "epanechnikov": EpanechnikovKernelMixture,
}


def get_kernel(kernel):
    if isinstance(kernel, str):
        try:
            return KERNELS[kernel.lower()]
        except KeyError:
            raise ValueError("Unknown kernel: '%s'. Available kernels: %s" % (kernel, sorted(KERNELS)))
    return kernel  # class given directly


# ===========================================================================
#  Policy  q_{n+1} = (1 - lambda) g_{n+1} / int g_{n+1} + lambda q0
# ===========================================================================

class Policy(ProbaContinue):
    def __init__(self, kernel, q0, lambd, d=1):
        self.kernel = kernel
        self.q0 = q0
        self.lambd = float(lambd)
        self.d = d
        self.f = Function(lambda x: self.pdf(np.atleast_2d(x))[0], vect=self.pdf)
        self.logf = Function(lambda x: self.logpdf(np.atleast_2d(x))[0], vect=self.logpdf)

    def logpdf(self, X):
        with np.errstate(divide="ignore"):
            return np.logaddexp(np.log1p(-self.lambd) + self.kernel.logpdf(X),
                                np.log(self.lambd) + _log_density(self.q0, X))

    def pdf(self, X):
        return np.exp(self.logpdf(X))

    def simulation(self, n):
        n0 = np.random.binomial(n, self.lambd)
        parts = []
        if n - n0 > 0:
            parts.append(self.kernel.simulation(n - n0))
        if n0 > 0:
            parts.append(np.asarray(self.q0.simulation(n0), dtype=float).reshape(n0, self.d))
        return np.vstack(parts)[np.random.permutation(n)]


def _sliced_wass(X, Y, weights, seed=1):
    n = Y.shape[0]
    return ot.sliced.sliced_wasserstein_distance(X, Y, weights / np.sum(weights), np.full(n, 1.0 / n), seed=seed)


# ===========================================================================
#  Algorithm
# ===========================================================================

class AIS():
    '''
    Indicator  = "Wass" (sliced Wasserstein), "MSE" (error on the mean),
                 "score" (function functionT), "" (no indicator).
    log_target = True if the target is a LOG-density (default: Indicator == "score").
    n_kernels  = number of kernels in q_n as a function of the number of particles N:
                 "sqrt" -> int(sqrt(N)), "half" -> int(N/2) (original code / paper), or a callable N -> int.
    wass_target = target sample used by the sliced Wasserstein indicator:
                 "full"    -> a sample of size bs_ini + nbIter*bs at every evaluation (as in the original code),
                 "current" -> only its first N points (faster, but noisier/biased upwards at small N).
    eval_every = the indicator is computed every `eval_every` iterations (and always at
                 iteration 0 and at the last iteration); nbPoint, indicator, newIndicator,
                 logW and varW are only filled at those iterations. Use 1 for every iteration.
    kernel     = "gaussian", "rectangular", "triangular", "epanechnikov" or a class.
    For speed, the target should be a Function with a `vect` argument
    (vectorised evaluation); otherwise a point-by-point Python loop is used.
    '''
    def __init__(self, lambd, gamma, b, eta, q0, target, batch_size=50, batch_size_ini=2000,
                 bootstrap=False, bootstrap_size=200, d=1, coeff=1, algo=2, distF=True,
                 Indicator="Wass", functionT="", kernel="gaussian", kernel_kwargs=None, verbose=True,
                 log_target=None, eval_every=10,
                 n_kernels="sqrt", wass_target="full"):
        self.d = d
        self.lambd = lambd
        self.q0 = q0
        self.gamma = gamma
        self.b = b if isinstance(b, Function) else Function(lambda x: b)
        self.eta = eta
        self.target = target
        self.ftarget = target.f if distF else target
        self.q = q0
        self.bs = batch_size
        self.bs_ini = batch_size_ini
        self.bootstrap = bootstrap
        self.bootstrap_size = bootstrap_size
        self.typeError = Indicator
        self.algo = algo
        self.C = coeff
        self.delta = 1 / 2
        self.score = functionT
        self.verbose = verbose
        self.log_target = (Indicator == "score") if log_target is None else log_target
        self.eval_every = max(1, int(eval_every))
        if callable(n_kernels):
            self.n_kernels = n_kernels
        elif n_kernels == "sqrt":
            self.n_kernels = lambda N: int(np.sqrt(N))
        elif n_kernels == "half":
            self.n_kernels = lambda N: int(N / 2)
        else:
            raise ValueError("n_kernels must be 'sqrt', 'half' or a callable")
        if wass_target not in ("full", "current"):
            raise ValueError("wass_target must be 'full' or 'current'")
        self.wass_target = wass_target
        self.kernelName = kernel if isinstance(kernel, str) else getattr(kernel, "__name__", str(kernel))
        self.Kernel = get_kernel(kernel)
        self.kernel_kwargs = {} if kernel_kwargs is None else dict(kernel_kwargs)
        for name in ["X", "weight", "indicator", "newIndicator", "nbPoint", "Zn", "newZ", "newWeight",
                     "varW", "logW", "time", "timeF", "timeQ", "timeSim", "timeIter", "nEvalF", "nKernels"]:
            setattr(self, name, np.array([]))

    # ---- sampling + evaluation of f and log q_n (timed) ----
    def _draw(self, nb):
        t0 = time.perf_counter()
        X = np.asarray(self.q.simulation(nb), dtype=float).reshape(nb, self.d)
        t1 = time.perf_counter()
        fX = _evaluate(self.ftarget, X)
        t2 = time.perf_counter()
        logqX = self.q.logpdf(X) if isinstance(self.q, Policy) else _log_density(self.q, X)
        t3 = time.perf_counter()
        self.timeSim = np.append(self.timeSim, t1 - t0)
        self.timeF = np.append(self.timeF, t2 - t1)
        self.timeQ = np.append(self.timeQ, t3 - t2)
        self.nEvalF = np.append(self.nEvalF, nb)
        return X, fX, logqX

    def iteration(self, nbIter):
        Ntot = self.bs_ini + nbIter * self.bs
        self.X = np.zeros([Ntot, self.d])
        self.H = np.zeros([Ntot, self.d])          # bandwidth attached to each particle
        self.weight = np.zeros(Ntot)               # mixed weights w^eta (log-weights if log_target)
        self.newWeight = np.zeros(Ntot)            # importance weights f/q (log if log_target)
        logT = self.log_target
        if logT:
            self.weight[:] = -np.inf
            self.newWeight[:] = -np.inf
        if self.typeError == "Wass":               # target sample drawn only once
            self.Ytarget = np.asarray(self.target.simulation(Ntot), dtype=float).reshape(Ntot, self.d)

        for n in range(0, nbIter + 1):
            self.time = np.append(self.time, time.time())
            tIter = time.perf_counter()
            if self.verbose and n > 0 and n % 20 == 0:
                print(n)
                with open("it" + str(n) + ".csv", "a") as fic:
                    np.savetxt(fic, np.ones(1), delimiter='\t')

            nb = self.bs_ini if n == 0 else self.bs
            N = self.bs_ini + n * self.bs
            X, fX, logqX = self._draw(nb)
            self.X[N - nb:N] = X
            self.H[N - nb:N] = self.b.eval(n) * np.ones(self.d)
            g = self.gamma.eval(n)

            if logT:   # fX = log f
                weight = self.C * fX - logqX
                if n > 0:
                    self.weight[:N - nb] += np.log(1 - g)
                self.weight[N - nb:N] = np.log(g) + self.eta * weight - np.log(nb)
                self.newWeight[N - nb:N] = weight
                kernelWeights = np.exp(self.weight[:N] - np.max(self.weight[:N]))
            else:
                weight = self.C * fX * np.exp(-logqX)
                if n > 0:
                    self.weight[:N - nb] *= (1 - g)
                self.weight[N - nb:N] = g / nb * np.power(weight, self.eta)
                self.newWeight[N - nb:N] = weight
                kernelWeights = self.weight[:N].copy()
            mz = np.mean(weight)
            self.newZ = np.append(self.newZ, mz if n == 0 else mz / (n + 1) + self.newZ[-1] * n / (n + 1))

            Zn = np.sum(kernelWeights)
            if n > 0:
                self.Zn = np.append(self.Zn, Zn)
            if Zn == 0 or not np.isfinite(Zn):
                kernelWeights = np.full(N, 1.0 / N)
            else:
                kernelWeights /= Zn

            # resample n_kernels(N) centers and build the new policy
            nKernels = max(1, int(self.n_kernels(N)))
            idx = np.random.choice(N, size=nKernels, p=kernelWeights)
            kernelParameters = np.stack([self.X[idx], self.H[idx]], axis=2)     # [m, d, 2]
            self.gauss = self.Kernel(np.full(nKernels, 1.0 / nKernels), kernelParameters, d=self.d, **self.kernel_kwargs)
            self.q = Policy(self.gauss, self.q0, self.lambd.eval(n), d=self.d)
            self.nKernels = np.append(self.nKernels, nKernels)
            self.timeIter = np.append(self.timeIter, time.perf_counter() - tIter)

            # ---- indicators (not included in timeIter), every eval_every iterations ----
            if n % self.eval_every != 0 and n != nbIter:
                continue
            self.nbPoint = np.append(self.nbPoint, N)
            if self.typeError == "Wass":
                Y = self.Ytarget if self.wass_target == "full" else self.Ytarget[:N]
                w = self.weight[:N]
                self.indicator = np.append(self.indicator, _sliced_wass(self.X[:N], Y, w))
                self.newIndicator = np.append(self.newIndicator, _sliced_wass(self.X[:N], Y, self.newWeight[:N]))
                wv = w if n == 0 else self.newWeight[:N]   # (as in the original version)
                self.varW = np.append(self.varW, np.mean((wv - 1) ** 2))
                self.logW = np.append(self.logW, 1 / ((self.delta - 1) * np.log(N)) *
                                      np.log(N ** (self.delta - 1) * np.sum((w / np.sum(w)) ** self.delta)))
            elif self.typeError == "score":
                nw = np.exp(self.newWeight[:N] - np.max(self.newWeight[:N]))
                self.indicator = np.append(self.indicator, self.score.eval([self.X[:N], kernelWeights]))
                self.newIndicator = np.append(self.newIndicator, self.score.eval([self.X[:N], nw / np.sum(nw)]))
            elif self.typeError == "MSE":
                for w, name in [(self.weight[:N], "indicator"), (self.newWeight[:N], "newIndicator")]:
                    err = np.linalg.norm(self.target.esp - w @ self.X[:N] / np.sum(w)) ** 2
                    setattr(self, name, np.append(getattr(self, name), err))

    # ---- computation-time summary ----
    def timing_summary(self):
        nF = np.sum(self.nEvalF)
        tF, tQ, tS, tT = np.sum(self.timeF), np.sum(self.timeQ), np.sum(self.timeSim), np.sum(self.timeIter)
        return {
            "kernel": self.kernelName,
            "n_eval_f": int(nF),
            "total_time_s": tT,
            "f_time_s": tF,
            "q_time_s": tQ,
            "sampling_time_s": tS,
            "other_time_s": tT - tF - tQ - tS,
            "time_per_f_eval_us": 1e6 * tF / nF,          # cost of one evaluation of f alone
            "total_time_per_f_eval_us": 1e6 * tT / nF,    # cost of the algorithm per evaluation of f
            "f_share": tF / tT,
        }

    def plot_density(self):
        plt.figure()
        abs = np.linspace(-10, 10, 100).reshape(-1, 1)
        plt.plot(abs, self.q.f.evalAIS(abs))
        plt.plot(abs, self.q0.f.evalAIS(abs))
        plt.plot(abs, self.target.f.evalAIS(abs))
        plt.savefig("density")

    def plot_error(self):
        plt.figure()
        plt.plot(self.nbPoint, np.log(self.indicator))
        plt.savefig("error")

    def sim(self, n=10000):
        return self.gauss.simulation(n)

    def mode_found(self, mod1, mod2, nbIter):
        N = self.bs_ini + nbIter * self.bs
        X, w = self.X[:N], self.newWeight[:N]
        dist = 0.5 * np.linalg.norm(mod1 - mod2)
        S = np.sum(w)
        m1 = np.sum(w[np.linalg.norm(X - mod1, axis=1) <= dist])
        m2 = np.sum(w[np.linalg.norm(X - mod2, axis=1) <= dist])
        return m1 / S, m2 / S
