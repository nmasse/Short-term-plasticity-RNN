import numpy as np
from parameters import *

g = np.power(2., -np.arange(1., par['num_mw']+1.)-2.)
C = np.power(2., np.arange(2., par['num_mw']+2.)-1.)

def adjust(weight, U, g_scaling, C_scaling, R):
    """
    Takes in a weight matrix, U metaweight set (for that weight matrix), and a
    scaling factor or matrix for g.  Performs the metaweight operation, then
    returns values for the weights and metaweights.
    """

    print(np.shape(weight), np.shape(U), np.shape(g_scaling), np.shape(C_scaling), np.shape(R))

    if len(weight.shape) == 2:
        g_set = np.einsum('ij,k->ijk', g_scaling, g)
        C_set = np.einsum('ij,k->ijk', C_scaling, C)
    elif len(weight.shape) == 3:
        g_set = np.einsum('ijk,l->ijkl', g_scaling, g)
        C_set = np.einsum('ijk,l->ijkl', C_scaling, C)

    init_weight = np.copy(weight)
    init_U      = np.copy(U)

    for j in range(par['mw_steps']):
        weight_prime = weight + g_set[...,0]*(U[...,0] - weight)*par['mw_dt']

        U_prime = np.copy(U)
        if par['num_mw'] == 1:
            U_prime[...,0] = U[...,0] + (1/C_set[...,0]*(g_set[...,0]*(weight - U[...,0])))*par['mw_dt']
        else:
            U_prime[...,0] = U[...,0] + (1/C_set[...,0])*(g_set[...,0]*(weight - U[...,0])+g_set[...,1]*(U[...,1] - U[...,0]))*par['mw_dt']
            U_prime[...,-1] = U[...,-1] + (1/C_set[...,-1])*g_set[...,-1]*(U[...,-2] - U[...,-1])*par['mw_dt']

        for i in range(1, par['num_mw']-1):
            U_prime[...,i] = U[...,i] + (1/C_set[...,i])*(g_set[...,i]*(U[...,i-1] - U[...,i])+g_set[...,i+1]*(U[...,i+1] - U[...,i]))*par['mw_dt']

        if j != (par['mw_steps']-1):
            weight = weight_prime
            U = U_prime

    """
    mw_string = str(np.round(np.mean(weight_prime), 4)).ljust(8)
    for j in range(par['num_mw']):
        string = ' - ' + str(np.round(np.mean(U_prime[...,j]), 4)).ljust(8)
        mw_string += string
    print('\n' + mw_string + '\n' + str(np.mean(weight_prime - np.mean(U_prime, -1))) + '\n----------')
    """

    R_prime = weight_prime - np.sum(U_prime, axis=-1)

    return np.float32(weight_prime - init_weight), np.float32(U_prime - init_U), np.float32(R_prime - R)
