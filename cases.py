"""
The four test cases of the paper (Section 4.4 and 4.5), with everything a method needs:
    dim, q0 (initial distribution), logf (unnormalised log target, vectorised),
    metric(X, w) (quality of the weighted particles), and the MIDAS parameters.

Cases
    coldstart   : target N(5/sqrt(d) 1, 0.4^2/d I), q0 = N(0, 5/d I), d = 16
    mixture     : target 0.2 N(m, 0.4^2/d I) + 0.8 N(-m, 0.4^2/d I), m = 1/(2 sqrt(d)) 1
                  (weights set by MIX_WEIGHTS; the paper uses 1/2, 1/2),
                  q0 = Student(0, 5/d I, df = 2), d = 16
    anisotropic : same as mixture with covariance 0.4^2/d Diag(10, 1, ..., 1)
    bayreg      : Bayesian logistic regression on 'waveform' (d = 22, 400 train / 4600 test),
                  prior beta ~ Gamma(1, rate 0.01), w | beta ~ N(0, I/beta)

Metric
    synthetic cases : log sliced Wasserstein distance between the weighted particles and a
                      sample of the target of the same size (50 projections, POT)
    bayreg          : test accuracy of the posterior predictive rule
                      P(c = 1 | z) = sum_i w_i sigmoid(w_i^T z) > 1/2
"""
import os

import numpy as np
import ot
import pandas as pd
from scipy.special import expit, gammaln

from classFonction import Function
from classProba import Mixture, Normal, ProbaContinue, Student

HERE = os.path.dirname(os.path.abspath(__file__))

# common budget: MIDAS with an initial batch of 2000 and 200 iterations of 300 particles
BS_INI, BS, N_ITER = 2000, 300, 200
BUDGET = BS_INI + N_ITER * BS          # 62000 evaluations of f
CHECKPOINTS = BS_INI + BS * np.arange(0, N_ITER + 1, 20)   # 2000, 8000, ..., 62000


# ----------------------------------------------------------------------------- metrics
def _dedup(X, w):
    """Remove zero weights and merge repeated rows (MCMC chains repeat rejected states)."""
    keep = w > 0
    X, w = X[keep], w[keep]
    U, inv = np.unique(X, axis=0, return_inverse=True)
    if U.shape[0] < X.shape[0]:
        w = np.bincount(inv.reshape(-1), weights=w, minlength=U.shape[0])
        X = U
    return X, w


def log_sliced_wasserstein(target, X, w, n_projections=50, seed=1):
    n = len(X)                                   # target sample of the same size as the particle set
    X, w = _dedup(np.asarray(X, float), np.asarray(w, float))
    Y = np.asarray(target.simulation(n), float)
    return float(np.log(ot.sliced.sliced_wasserstein_distance(
        X, Y, w / np.sum(w), np.full(Y.shape[0], 1.0 / Y.shape[0]), n_projections=n_projections, seed=seed)))


# ----------------------------------------------------------------------------- MIDAS parameters
def midas_parameters(d, c=0.5, m=BS):
    """b_n = c/sqrt(d) (sqrt(n m) + 1)^(-1/(4+d)) (c tuned with tune_bandwidth.py),
    gamma_n = 1/(n+10), lambda_n = 1/log(m n + 10) (0.5 for n < 10)  -- paper, Section 4.1."""
    b = Function(lambda n: c / np.sqrt(d) * (np.sqrt(n * m) + 1) ** (-1.0 / (4 + d)))
    gamma = Function(lambda n: 1.0 / (n + 10))
    lambd = Function(lambda n: 0.5 if n < 10 else 1.0 / np.log(10 + m * n))
    return b, gamma, lambd


class Case:
    name = ""
    log_metric_label = ""

    def metric(self, X, w):
        raise NotImplementedError


# ----------------------------------------------------------------------------- synthetic cases
class SyntheticCase(Case):
    metric_label = "log sliced Wasserstein distance"
    higher_is_better = False

    def __init__(self, name, q0, target, d):
        self.name, self.q0, self.target, self.dim = name, q0, target, d
        self.logf = target.logf
        self.b, self.gamma, self.lambd = midas_parameters(d)

    def metric(self, X, w):
        return log_sliced_wasserstein(self.target, X, w)


def coldstart(d=16):
    q0 = Normal(np.zeros(d), np.eye(d) * 5 / d, d=d)
    target = Normal(5 * np.ones(d) / np.sqrt(d), np.eye(d) * 0.4 ** 2 / d, d=d)
    return SyntheticCase("coldstart", q0, target, d)


# weights of the two components (+m, -m) of the mixture and anisotropic cases
MIX_WEIGHTS = (0.2, 0.8)


def mixture(d=16, anisotropic=False, weights=MIX_WEIGHTS):
    q0 = Student(np.zeros(d), np.eye(d) * 5 / d, df=2, d=d)
    m = np.ones(d) / np.sqrt(d) / 2
    diag = np.ones(d)
    if anisotropic:
        diag[0] = 10.0
    cov = np.diag(diag) * 0.4 ** 2 / d
    target = Mixture(np.array(weights, dtype=float), np.array([Normal(m, cov, d=d), Normal(-m, cov, d=d)]), d=d)
    return SyntheticCase("anisotropic" if anisotropic else "mixture", q0, target, d)


# ----------------------------------------------------------------------------- Bayesian logistic regression
def _log_gamma(beta, a, rate):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a * np.log(rate) - gammaln(a) + (a - 1) * np.log(beta) - rate * beta
    return np.where(beta > 0, r, -np.inf)


def _log_normal_prec(w, beta):
    k = w.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = -k / 2 * np.log(2 * np.pi) + k / 2 * np.log(beta) - beta / 2 * np.sum(w ** 2, axis=1)
    return np.where(beta > 0, r, -np.inf)


class BayesQ0(ProbaContinue):
    """q0: beta ~ Gamma(a, rate b), w | beta ~ N(0, I/beta) (as in the original bayReg.py, a = b = 1)."""
    def __init__(self, d, a=1.0, b=1.0):
        self.d, self.a, self.b = d, a, b

        def logv(X):
            X = np.atleast_2d(X)
            return _log_normal_prec(X[:, :-1], X[:, -1]) + _log_gamma(X[:, -1], a, b)
        self.logf = Function(lambda x: logv(x)[0], vect=logv)
        self.f = Function(lambda x: np.exp(logv(x)[0]), vect=lambda X: np.exp(logv(X)))

    def simulation(self, n):
        beta = np.random.gamma(self.a, 1 / self.b, n)
        w = np.random.standard_normal((n, self.d - 1)) / np.sqrt(beta)[:, None]
        return np.column_stack([w, beta])


class BayesRegCase(Case):
    metric_label = "test accuracy"
    higher_is_better = True

    def __init__(self, path=os.path.join(HERE, "datasets", "waveform.csv"), a=1.0, b=0.01):
        data = pd.read_csv(path)
        d = data.shape[1] - 2          # 21 features + beta = 22
        z = data.iloc[:, :d - 1].to_numpy(float)
        c = data.iloc[:, d - 1].to_numpy(float)
        train = data.iloc[:, d].to_numpy() == 1
        test = data.iloc[:, d + 1].to_numpy() == 1
        self.zTrain, self.cTrain, self.zTest, self.cTest = z[train], c[train], z[test], c[test]
        self.name, self.dim = "bayreg", d
        self.q0 = BayesQ0(d)
        zc = self.zTrain * self.cTrain[:, None]

        def logt(X):
            X = np.atleast_2d(X)
            w, beta = X[:, :-1], X[:, -1]
            mv = -np.sum(np.logaddexp(0, -(w @ zc.T)), axis=1)    # sum_i log sigmoid(c_i w^T z_i)
            return _log_gamma(beta, a, b) + _log_normal_prec(w, beta) + mv
        self.logf = Function(lambda x: logt(x)[0], vect=logt)
        self.b, self.gamma, self.lambd = midas_parameters(d)

    def metric(self, X, w, chunk=4000):
        X, w = _dedup(np.asarray(X, float), np.asarray(w, float))
        w = w / np.sum(w)
        zc = self.zTest * self.cTest[:, None]
        p = np.zeros(zc.shape[0])                 # posterior predictive prob. of the true label
        for s in range(0, X.shape[0], chunk):
            p += expit(zc @ X[s:s + chunk, :-1].T) @ w[s:s + chunk]
        return float(np.mean(p > 0.5))


def get_case(name):
    if name == "coldstart":
        return coldstart()
    if name == "mixture":
        return mixture()
    if name == "anisotropic":
        return mixture(anisotropic=True)
    if name == "bayreg":
        return BayesRegCase()
    raise ValueError("unknown case: %s" % name)


CASES = ["coldstart", "mixture", "anisotropic", "bayreg"]
