import numpy as np
from parameters import *

g = np.power(2., -np.arange(1, par['num_mw']+1)-2)
C = np.power(2., np.arange(2, par['num_mw']+2)-1)

def adjust(weight, U, g_scaling):
    """
    Takes in a weight matrix, U metaweight set (for that weight matrix), and a
    scaling factor or matrix for g.  Performs the metaweight operation, then
    returns values for the weights and metaweights.
    """
    if len(weight.shape) == 2:
        g_set = np.einsum('ij,k->ijk', g_scaling, g)
    elif len(weight.shape) == 3:
        g_set = np.einsum('ijk,l->ijkl', g_scaling, g)

    for j in range(par['mw_steps']):
        weight_prime = weight + g_set[...,0]*(U[...,0] - weight)*par['mw_dt']

        U_prime = np.copy(U)
        U_prime[...,0] = U[...,0] + (1/C[0])*(g_set[...,0]*(weight - U[...,0])+g_set[...,0+1]*(U[...,0+1] - U[...,0]))*par['mw_dt']
        U_prime[...,-1] = U[...,-1] + (1/C[-1])*g_set[...,-1]*(U[...,-1-1] - U[...,-1])*par['mw_dt']
        for i in range(1, par['num_mw']-1):
            U_prime[...,i] = U[...,i] + (1/C[i])*(g_set[...,i]*(U[...,i-1] - U[...,i])+g_set[...,i+1]*(U[...,i+1] - U[...,i]))*par['mw_dt']

        if j != (par['mw_steps']-1):
            weight = weight_prime
            U = U_prime

    return weight_prime, U_prime
