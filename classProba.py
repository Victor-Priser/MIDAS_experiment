import numpy as np
import math as m
import scipy.stats as sc
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from classFonction import Function
import time as t
import ot


def logP(loga, logb):
    return loga + np.log(1 + np.exp(logb - loga))


class ProbaDiscrete:
    def __init__(self):
        pass


class Multinomial(ProbaDiscrete):
    '''Draws n rows of `values` with probabilities `weights`.'''
    def __init__(self, values=[], weights=[]):
        self.values = np.asarray(values)
        self.weights = np.asarray(weights, dtype=float)

    def simulation(self, n):
        p = self.weights / np.sum(self.weights)
        idx = np.random.choice(p.shape[0], size=n, p=p)
        return self.values[idx]

    def testComplexity(self, n=4e6, nbSimulation=1):
        k = 10000
        times = []
        while k < n:
            x = np.random.uniform(0, 1, k)
            x = x / np.sum(x)
            value = np.zeros([k, 1])
            value[:, 0] = np.linspace(1, k, k)
            start = t.time()
            Multinomial(value, x).simulation(nbSimulation)
            end = t.time()
            times.append(np.log(end - start))
            k += 100000
            print(k)
        plt.figure()
        plt.plot(times)
        plt.savefig("time")


class ProbaContinue:
    def __init__(self, f):
        self.f = f

    def plotDensity(self, name="test", n=10000, start=-10, end=10):
        x = np.linspace(start, end, n)
        y = self.f.evalVector(x)
        plt.figure()
        plt.plot(x, y)
        plt.savefig(name)

    def testSimulation(self, n=1000, name="testSim"):
        p = self.simulation(n)
        self.plotDensity()
        plt.hist(p, density=True)
        plt.savefig(name)


class Normal(ProbaContinue):
    def __init__(self, mu=0, sigma=1, d=1):
        if (d == 1):   # sigma is the standard deviation
            dist = sc.norm(mu, sigma)
            self.f = Function(lambda x: dist.pdf(x), vect=lambda X: dist.pdf(np.asarray(X).reshape(-1)))
            self.logf = Function(lambda x: dist.logpdf(x), vect=lambda X: dist.logpdf(np.asarray(X).reshape(-1)))
            self.F = Function(lambda x: dist.cdf(x))
            self.esp = mu
            self.var = sigma**2
            self.sig = sigma
        else:          # sigma is the covariance matrix
            dist = sc.multivariate_normal(mu, sigma)   # frozen distribution: Cholesky computed only once
            self.f = Function(lambda x: dist.pdf(x), vect=lambda X: np.atleast_1d(dist.pdf(X)))
            self.logf = Function(lambda x: dist.logpdf(x), vect=lambda X: np.atleast_1d(dist.logpdf(X)))
            self.F = Function(lambda x: dist.cdf(x))
            self.esp = mu
            self.sig = sigma
        self.dist = dist
        self.d = d

    def simulation(self, n):
        if (self.d == 1):
            x = np.zeros([n, 1])
            x[:, 0] = np.random.normal(self.esp, self.sig, n)
        else:
            x = np.random.multivariate_normal(self.esp, self.sig, n)
        return x

    def __add__(self, other):
        return Normal(self.esp + other.esp, self.sig)

    def __radd__(self, other):
        return Normal(self.esp + other.esp, self.sig)


class Gamma(ProbaContinue):
    def __init__(self, a=1, b=0.001):  # a = shape, b = rate
        dist = sc.gamma(a, 0, 1 / b)
        self.f = Function(lambda x: dist.pdf(x), vect=lambda X: dist.pdf(np.asarray(X).reshape(-1)))
        self.logf = Function(lambda x: dist.logpdf(x), vect=lambda X: dist.logpdf(np.asarray(X).reshape(-1)))
        self.F = Function(lambda x: dist.cdf(x))
        self.esp = a * b
        self.a = a
        self.b = b

    def simulation(self, n):
        x = np.zeros([n, 1])
        x[:, 0] = np.random.gamma(self.a, 1 / self.b, n)
        return (x)


class Uniform(ProbaContinue):
    def __init__(self, start, end):
        self.s = start
        self.e = end

    def simulation(self, n):
        return np.random.uniform(self.s, self.e, n)


class Student(ProbaContinue):
    def __init__(self, loc=0, shape=1, df=1, d=1):
        if (d == 1):
            dist = sc.t(df, loc, shape)
            self.f = Function(lambda x: dist.pdf(x), vect=lambda X: dist.pdf(np.asarray(X).reshape(-1)))
            self.logf = Function(lambda x: dist.logpdf(x), vect=lambda X: dist.logpdf(np.asarray(X).reshape(-1)))
        else:   # shape is the scale matrix
            dist = sc.multivariate_t(loc, shape, df=df)
            self.f = Function(lambda x: dist.pdf(x), vect=lambda X: np.atleast_1d(dist.pdf(X)))
            self.logf = Function(lambda x: dist.logpdf(x), vect=lambda X: np.atleast_1d(dist.logpdf(X)))
            self.F = Function(lambda x: dist.cdf(x))
        self.dist = dist
        self.loc = loc
        self.shape = shape
        self.df = df
        self.d = d

    def simulation(self, n):
        if (self.d == 1):
            return (self.loc + self.shape * np.random.standard_t(self.df, n)).reshape(n, 1)
        x = np.random.chisquare(self.df, n) / self.df
        z = np.random.multivariate_normal(np.zeros(self.d), self.shape, n)
        return self.loc + (z.T / np.sqrt(x)).T


class Mixture(ProbaContinue):
    def __init__(self, weights, proba, d=1):
        self.weights = np.asarray(weights, dtype=float)
        self.proba = proba
        self.d = d

        def g(x):
            g = 0
            for i in range(len(proba)):
                g += weights[i] * proba[i].f.eval(x)
            return g

        def gv(X):
            return sum(weights[i] * proba[i].f.evalAIS(X) for i in range(len(proba)))
        self.f = Function(g, vect=gv)
        if all(hasattr(p, "logf") for p in proba):
            logw = np.log(self.weights)

            def lgv(X):
                L = np.column_stack([logw[i] + proba[i].logf.evalAIS(X) for i in range(len(proba))])
                return logsumexp(L, axis=1)
            self.logf = Function(lambda x: lgv(np.atleast_2d(x))[0], vect=lgv)
        if all(hasattr(p, "esp") for p in proba):
            self.esp = sum(self.weights[i] * np.asarray(proba[i].esp) for i in range(len(proba)))

    def simulation(self, n):
        nb = np.random.multinomial(n, self.weights)
        sim = [np.asarray(self.proba[j].simulation(int(k))).reshape(int(k), self.d)
               for j, k in enumerate(nb) if k > 0]
        res = np.vstack(sim)
        return res[np.random.permutation(n)]


class GaussianMixture(ProbaContinue):
    '''
    Mixture of Gaussians with diagonal covariance:
      let X be a standard normal vector and (H, Y) a discrete vector such that
      P(H = h_i, Y = X_i) = W_i; then Z = H*X + Y is a mixture of k normals with
      weights W_i, means X_i and standard deviations h_i.
    parameters: [n, 2] if d == 1, [n, d, 2] otherwise;
                parameters[..., 0] = means, parameters[..., 1] = standard deviations.
    (Vectorised version; the Gaussian kernel used by AIS is GaussianKernelMixture in AIS.py.)
    '''
    def __init__(self, weights, parameters, d=1, ind=True):  # careful: standard deviations, not variances
        self.weights = np.asarray(weights, dtype=float)
        parameters = np.asarray(parameters, dtype=float)
        if parameters.ndim == 2:
            parameters = parameters[:, None, :]
        self.parameters = parameters
        self.d = d
        self.ind = ind
        mu, sd = parameters[:, :, 0], parameters[:, :, 1]
        logc = np.log(self.weights) - np.sum(np.log(sd), axis=1) - d / 2 * np.log(2 * np.pi)

        def lgv(X):
            X = np.asarray(X, dtype=float).reshape(-1, d)
            Q = np.sum(((X[:, None, :] - mu[None]) / sd[None]) ** 2, axis=2)
            return logsumexp(logc - Q / 2, axis=1)
        self.logf = Function(lambda x: lgv(x)[0], vect=lgv)
        self.f = Function(lambda x: np.exp(lgv(x)[0]), vect=lambda X: np.exp(lgv(X)))
        self.esp = np.dot(self.weights, mu)

    def simulation(self, n):
        idx = np.random.choice(self.weights.shape[0], size=n, p=self.weights / np.sum(self.weights))
        return self.parameters[idx, :, 1] * np.random.standard_normal((n, self.d)) + self.parameters[idx, :, 0]


def wass(target, X, weight, d=1):
    n = np.shape(X)[0]
    Y = target.simulation(n)
    weightY = 1 / n * np.ones(n)
    if d == 1:
        return ot.sliced.sliced_wasserstein_distance(X, Y, weight / np.sum(weight), weightY)
    return ot.sliced.sliced_wasserstein_distance(X, Y, weight / np.sum(weight), weightY, seed=1)
