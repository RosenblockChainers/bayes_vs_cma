import numpy as np

class RealSolution(object):
    """real value solution"""
    def __init__(self, **params):
        self.f = float('nan')
        self.x = np.zeros([params['dim'], 1])
        self.z = np.zeros([params['dim'], 1])
        self.feasible = True

    def __repr__(self):
        return "f(x)=%s, x=%s" % (str(self.f), str(self.x.T))