# -*- coding: utf-8 -*-
import numpy as np

class SphereFunction(object):
    def __init__(self, n):
        self.n = n
        self.name = "%dD-SphereFunction" % n

    def evaluate(self, x):
        return np.sum(x**2)

class HimmelblauFunction(object):
    def __init__(self, n):
        self.n = n
        self.name = "%dD-HimmelblauFunction" % n

    def evaluate(self, x):
        return (x[0]**2 + x[1] - 11.)**2 + (x[0] + x[1]**2 - 7.)**2

class KTabletFunction(object):
    def __init__(self, n, k=None):
        self.n = n
        if k is None:
            self.k = int(n/4) # default k value
        else:
            self.k = k
        self.name = "%dD-%d-tabletFunction" % (n, self.k)

    def evaluate(self, x):
        if len(x) < 2:
            raise ValueError('dimension must be greater one')
        return np.sum(x[0:self.k]**2) + np.sum((100*x[self.k:self.n])**2)
        # return x[0]**2 + (100*x[1])**2

class EllipsoidFunction(object):
    def __init__(self, n):
        self.n = n
        self.name = "%dD-EllipsoidFunction" % n
        self.aratio = 1000**(np.arange(n)/(n-1)).reshape(n, 1)

    def evaluate(self, x):
        if len(x) < 2:
            raise ValueError('dimension must be greater one')
        #return np.sum([(1000**(i / (self.n-1)) * x[i])**2 for i in range(self.n)])
        return np.sum((self.aratio * x)**2)

class RosenbrockChainFunction(object):
    def __init__(self, n):
        self.n = n
        self.name = "%dD-RosenbrockChainFunction" % n

    def evaluate(self, x):
        if len(x) < 2:
            raise ValueError('dimension must be greater one')
        #return np.sum([100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(self.n-1)])
        return np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)


