import numpy as np


class Function:
    '''
    Function from R^d to R.
      func : evaluation at ONE point x (array of size d)
      vect : (optional) vectorised evaluation on an array X of shape [n, d],
             returning an array of size n. When provided, evalAIS uses it
             (much faster than a point-by-point Python loop).
    '''
    def __init__(self, func=lambda x: 0, vect=None):
        self.f = func
        self.vect = vect
        self.fv = np.vectorize(func)

    def eval(self, x):
        return self.f(x)

    def evalVector(self, x):
        return self.fv(x)

    def evalAIS(self, x):
        x = np.asarray(x)
        n = np.shape(x)[0]
        if self.vect is not None:
            return np.asarray(self.vect(x), dtype=float).reshape(n)
        res = np.empty(n)
        for i in range(n):
            res[i] = self.f(x[i, :])
        return res

    def evalMesh(self, x, y):
        pts = np.column_stack([np.ravel(x), np.ravel(y)])
        return self.evalAIS(pts).reshape(np.shape(x))

    def __add__(self, other):
        def res(x):
            return self.eval(x) + other.eval(x)
        vect = None
        if self.vect is not None and getattr(other, "vect", None) is not None:
            vect = lambda X: self.evalAIS(X) + other.evalAIS(X)
        return Function(res, vect)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        def res(x):
            return other * self.eval(x)
        vect = None if self.vect is None else (lambda X: other * self.evalAIS(X))
        return Function(res, vect)

    def __rmul__(self, other):
        return self.__mul__(other)
