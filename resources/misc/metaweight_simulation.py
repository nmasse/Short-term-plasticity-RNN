import numpy as np
import matplotlib.pyplot as plt
from parameters import *

par['num_mw'] = 10
par['mw_steps'] = 100
par['mw_dt'] = 0.001

import metaweight as mw
np.set_printoptions(formatter={'float': lambda x: " {0:0.1f}".format(x)})

w = np.array([[0.]])
m = np.zeros([1, 1, par['num_mw']])
m[...,5] = 10

def update_set(weight, metaweights):
    g_scaling = np.array([[1]])
    weight_delta, metaweight_delta = mw.adjust(weight, metaweights, g_scaling)
    return weight+weight_delta, metaweights+metaweight_delta, np.sum([np.sum(weight_delta), np.sum(metaweight_delta)])

def print_pair(i, weight, metaweights, da):
    print(str(i).ljust(8), '>', np.around(weight[0], 2), np.around(metaweights[0,0], 2), '{:0.6f}'.format(da))

print('Index    = Number of seconds run')
print('Matrices = Weight + metaweights')
print('Value    = Total change in matrices\' values since last output')
print_pair(0, w, m, 0)
print('-'*79)
da = 0
for i in range(int(1e+9)):
    w, m, d = update_set(w, m)
    da += d
    print(str(int(i%1000)).ljust(4), '-----', '{:0.12f}'.format(d), end='\r')
    if i%1000 == 0 and i != 0:
        print_pair(int(i/10), w, m, da)
        da = 0
