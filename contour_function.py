import math
import numpy as np

def rosen(x, y):
    #return 100.0*(y - x**2)**2 + (x - 1)**2
    return 100.0*(x - y**2)**2 + (x - 1)**2

def ktablet(x, y):
    return x**2 + (100*y)**2

def sphere(x, y):
    return x**2 + y**2

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y - 7)**2

def valley(x, y):
    return -3*np.power(1-x,2)*np.exp(-x**2-(y+1)**2) + 10*(x/5.0 - x**5 - y**5)*np.exp(-x**2-y**2)+3*np.exp(-(x+1)**2 - y**2)

def fletcher_and_powell(x, y):
    d = 2
    a = [[-78, 28], [-13, -50]]
    b = [[97, -25], [30, 25]]
    alpha = [0.435934, 0.550595]
    z = []
    z.append(x)
    z.append(y)
    A = []
    B = []
    for i in range(d):
        tmpA = 0
        tmpB = 0
        for j in range(d):
            tmpA += (a[i][j]*np.sin(alpha[j])) + (b[i][j]*np.cos(alpha[j]))
            tmpB += (a[i][j]*np.sin(z[j])) + (b[i][j]*np.cos(z[j]))
        A.append(tmpA)
        B.append(tmpB)
    ret = 0
    for i in range(d):
        ret += (A[i] - B[i])**2
    return ret

def double_sphere(x, y):
    depth = 0.1
    u1 = 2.56
    u2 = -2.56
    s = 0.1
    return min((x - u1)**2 + (y - u1)**2, (depth*2) + (s*((x - u2)**2 + (y - u2)**2)))
