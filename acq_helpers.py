import numpy as np
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import minimize
import sys


def acq_min(ac, gp, bounds):
    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(100000, bounds.shape[0])) #shape[0]はdim
    ys = ac(x_tries, gp=gp)
    x_min = x_tries[ys.argmin()]
    min_acq = ys.min()

    x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(250, bounds.shape[0]))

    for x_try in x_seeds:
        res = minimize(lambda x: ac(x.reshape(1, -1), gp=gp), # 目的関数
                        x_try.reshape(1, -1), # 初期点
                        bounds=bounds,
                        method="L-BFGS-B"
                        )

        if min_acq is None or res.fun <= min_acq:
            x_min = res.x
            min_acq = res.fun

    return np.clip(x_min, bounds[:, 0], bounds[:, 1])

class UtilityFunction(object):

    def __init__(self, kind, kappa, xi):
        self.kappa = kappa

        self.xi = xi

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)

    @staticmethod
    def _ucb(x, gp, kappa):
        mean, std = gp.predict(x, return_std=True)
        return mean - kappa * std
