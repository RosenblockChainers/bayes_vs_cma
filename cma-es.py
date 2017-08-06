#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
import random
import function
import math
from functools import reduce
import sys, datetime, os, time
import pickle

class RealSolution(object):
    def __init__(self, **params):
        self.f = float('nan')
        self.x = np.zeros([params['dim'], 1])
        self.z = np.zeros([params['dim'], 1])

def log_generation(log, g, evals, fval):
    log['g'].append(g)
    log['evals'].append(evals)
    log['fval'].append(fval)

def plot_fval(df, path, save=True):
    plt.figure()
    plt.title('%s/eval' % path)
    plt.plot(df['evals'], df['fval'])
    plt.xlim([0, params['max_evals']])
    plt.ylim([1e-10, 1e+6])
    plt.yscale('log')
    plt.xlabel('# evaluations')
    plt.ylabel(r'$f(\mathbf{x}_\mathrm{best})$')
    plt.grid()
    plt.minorticks_on()
    if save:
        plt.savefig('%s/eval.pdf' % path)
        plt.close()
    else:
        plt.show()

def main(**params):
    dim = params['dim']
    lamb = params['population_size']
    m = params['mean']
    sigma = params['sigma']
    A = params['A']
    np.random.seed(params['seed'])
    obj_func = params['obj_func']
    max_evals = params['max_evals']
    path = params['path']

    mu = int(lamb/2.)
    weights = np.array([max(0., math.log(lamb/2. + 1.) - math.log(i+1)) for i in range(mu)])
    weights = weights / sum(weights)
    mueff = 1. / np.sum(weights**2, axis=0)
    cs = (mueff+2.) / (dim+mueff+5.)
    cc = (4.+mueff/dim) / (dim+4.+2.*mueff/dim)
    c1 = 2. / (math.pow(dim+1.3, 2)+mueff)
    cmu = min(1-c1, 2*(mueff-2.+1./mueff) / (math.pow(dim+2., 2) + mueff))
    chiN = math.sqrt(dim) * (1. - 1./(4.*dim) + 1./(21.*dim*dim))
    damps = 1. + 2.*max(0., math.sqrt((mueff-1.)/(dim+1.))-1.) + cs
    eta_m = 1.

    ps = np.zeros([dim, 1])
    pc = np.zeros([dim, 1])

    log = {'g':[], 'evals':[], 'fval':[]}
    log_data = {'logfile':[], 'g':[]}

    g = 0
    no_of_evals = 0

    solutions = [RealSolution(**params) for i in range(lamb)]

    best = float("inf")

    # log output
    log_generation(log, g, no_of_evals, best)

    X = np.array([])
    while no_of_evals < max_evals:

        for s in solutions:
            s.z = np.random.randn(dim, 1)
            s.x = m + sigma * A.dot(s.z)
            s.f = obj_func.evaluate(s.x)
        no_of_evals += lamb
        g += 1

        solutions.sort(key=lambda s: s.f)

        # append to X
        for i in range(lamb):
            if len(X) == 0:
                X = solutions[i].x.reshape(1, -1)[0]
            else:
                X = np.vstack((X, solutions[i].x.reshape(1, -1)[0]))

        if solutions[0].f < best:
            best = solutions[0].f

        # logging
        log_generation(log, g, no_of_evals, best)
        if dim == 2:
            fname = '%s/logData%s.obj' % (path, str(g))
            log_data['logfile'].append(fname)
            log_data['g'].append(g)
            f = open(fname, 'wb')
            pickle.dump({'solutions': solutions, 'mean': m, 'A': A, 'X': X}, f)
            f.close()

        print(no_of_evals, ": ", best[0])

        # evolution path p_simga
        wz = np.sum([weights[i]*solutions[i].z for i in range(mu)], axis=0)
        ps = (1.-cs) * ps + np.sqrt(cs*(2.-cs)*mueff) * wz

        # hsig
        hsig = 1. if np.linalg.norm(ps) / math.sqrt(1.-math.pow(1.-cs, 2.*(g+1))) / chiN < 1.4 + 2./(dim+1.) else 0.

        # m and pc
        m += eta_m * sigma * A.dot(wz)
        pc = (1.-cc)*pc + math.sqrt(cc*(2.-cc)*mueff)*A.dot(wz)

        # grad C
        invA = np.linalg.inv(A)
        invA_pc = invA.dot(pc)
        rank_one = invA_pc.dot(invA_pc.T) + (1.-hsig)*cc*(2.-cc)*np.eye(dim, dtype=float) - np.eye(dim, dtype=float)
        rank_mu = reduce(lambda a,b: a+b, [weights[i]*(np.dot(solutions[i].z, solutions[i].z.T) - np.eye(dim, dtype=float)) for i in range(mu)])
        C = np.dot(A.dot(np.eye(dim, dtype=float) + c1*rank_one + cmu*rank_mu), A.T)

        sigma *= math.exp((cs/damps)*((np.linalg.norm(ps)/chiN) -1.))

        # eigenvalue decomposition
        e, v = np.linalg.eigh(C)
        A = np.dot(v.dot(np.diag(list(map(lambda a: math.sqrt(a), e)))), v.T)

    # output to csv from dataframe
    df = pd.DataFrame(log)
    df.index.name = '#index'
    df.to_csv('%s/log.csv' % path, seq=',')
    plot_fval(df, path)

    df_obj = pd.DataFrame(log_data)
    df.index.name = '#index'
    df.to_csv('%s/log_data.csv' % path, seq=',')

if __name__ == '__main__':
    params = {}
    params['seed'] = random.randint(0, 2**32 -1)
    params['dim'] = 2
    params['population_size'] = 6
    params['max_evals'] = 50
    # params['obj_func'] = function.SphereFunction(params['dim'])
    params['obj_func'] = function.HimmelblauFunction(params['dim'])
    params['mean'] = np.zeros([params['dim'], 1])
    params['sigma'] = 2.
    params['A'] = np.eye(params['dim'])

    # path
    time_name = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    path = 'log/' + params['obj_func'].name + '_' + time_name
    if not os.path.isdir(path):
        os.makedirs(path)
    print('create directory which is ' + path)
    params['path'] = path

    main(**params)

    print('seed is ' + str(params['seed']))
    print('create directory which is ' + path)
