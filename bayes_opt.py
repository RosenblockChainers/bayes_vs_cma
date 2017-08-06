#coding:utf-8
import function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from acq_helpers import UtilityFunction, acq_min
import sys, datetime, os, time
import pickle

def log_generation(log, g, evals, fval):
    log['g'].append(g)
    log['evals'].append(evals)
    log['fval'].append(fval)

def plot_fval(df, path, save=True):
    plt.figure()
    plt.title('%s/eval' % path)
    plt.plot(df['evals'], df['fval'])
    plt.xlim([0, params['max_evals']])
    plt.ylim([1e-2, 1e+6])
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
    np.random.seed(params['seed'])
    obj_func = params['obj_func']
    path = params['path']
    max_evals = params['max_evals']
    bounds = params['bounds']

    X = None
    Y = None
    acq = 'ucb'
    kappa=2.576
    xi = 0.
    util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
    # internal GP regressor
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=1.5),
        n_restarts_optimizer=25
    )
    # verbose
    res = {}
    res['min'] = {'min_val': None,
                'min_params': None}
    res['all'] = {'values': [], 'params': []}

    # logging
    log = {'g':[], 'evals':[], 'fval':[]}
    log_data = {'logfile':[], 'g':[]}

    best = float("inf")
    best_param = np.empty(dim)

    init_points_num = 5
    l = [np.random.uniform(b[0], b[1], size=init_points_num) for b in bounds]
    init_points = list(map(list, zip(*l)))
    y_init = []
    for x in init_points:
        y_init.append(obj_func.evaluate(np.array(x)))
    X = np.asarray(init_points)
    Y = np.asarray(y_init)
    # finish init
    g = 1
    no_of_evals = init_points_num

    # log output
    log_generation(log, g, no_of_evals, best)
    if dim == 2:
        fname = '%s/logData%s.obj' % (path, str(g))
        log_data['logfile'].append(fname)
        log_data['g'].append(g)
        f = open(fname, 'wb')
        pickle.dump({'X': X}, f)
        f.close()

    while no_of_evals < max_evals:
        g += 1

        """
        Step 1.
        find x_t by optimizing the acq function over the GP :
        x_t = argmin_{x} u(x|D_{1:t-1})
        """
        # print("Step 1. x_t = argmin_{x} u(x|D_{1:t-1})")
        x_t = acq_min(ac=util.utility,
                    gp=gp,
                    bounds=bounds)
        # print("x_t:", x_t)

        """
        Step 2.
        sample the objective function: y_t = f(x_t) + epsilon_t
        """
        # print("Step 2. y_t = f(x_t) + epsilon_t")
        y_t = obj_func.evaluate(x_t)
        no_of_evals += 1

        """
        Step 3.
        augment the data D_{1:t} = {D_{1:t-1}, (x_t, y_t)} and update the GP
        """
        # append most recently generated values to X and Y arrays
        X = np.vstack((X, x_t.reshape(1, -1)))
        Y = np.append(Y, y_t)
        # update the GP
        gp.fit(X, Y)

        # verbose
        best = Y.min()
        print(no_of_evals, ": ", best)

        res['min'] = {'min_val': best,
                    'min_params': X[Y.argmin()]}
        res['all']['values'].append(Y[-1])
        res['all']['params'].append(X[-1])

        log_generation(log, g, no_of_evals, best)
        if dim == 2:
            fname = '%s/logData%s.obj' % (path, str(g))
            log_data['logfile'].append(fname)
            log_data['g'].append(g)
            f = open(fname, 'wb')
            pickle.dump({'X': X}, f)
            f.close()

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
    params['seed'] = np.random.randint(0, 2**32 -1)
    # params['seed'] = 170001
    params['seed'] = 3809026734
    params['dim'] = 2
    params['max_evals'] = 50
    # params['obj_func'] = function.KTabletFunction(params['dim'], k=1)
    # params['obj_func'] = function.SphereFunction(params['dim'])
    params['obj_func'] = function.HimmelblauFunction(params['dim'])
    time_name = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    path = 'log/' + params['obj_func'].name + '_' + time_name
    if not os.path.isdir(path):
        os.makedirs(path)
    print('create directory which is ' + path)
    params['path'] = path

    params['bounds'] = np.zeros([params['dim'], 2])
    # params['bounds'][:,0] = -1.
    # params['bounds'][:,1] = 1.
    params['bounds'][:,0] = -4.
    params['bounds'][:,1] = 4.


    main(**params)

    print('seed is ' + str(params['seed']))
    print('create directory which is ' + path)

