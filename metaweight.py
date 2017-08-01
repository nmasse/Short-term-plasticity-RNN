import numpy as np
from parameters import *

class MetaweightSet:

    def __init__(self):
        self.mws_dict = {
            'W_stim_dend_u' : np.zeros(np.append(par['input_to_hidden_dend_dims'],par['num_mw']), dtype=np.float32),
            'W_stim_dend_g' : np.full(np.append(par['input_to_hidden_dend_dims'],par['num_mw']), par['cascade_strength'], dtype=np.float32),
            'W_stim_soma_u' : np.zeros(np.append(par['input_to_hidden_soma_dims'],par['num_mw']), dtype=np.float32),
            'W_stim_soma_g' : np.full(np.append(par['input_to_hidden_soma_dims'],par['num_mw']), par['cascade_strength'], dtype=np.float32),

            'W_td_dend_u' : np.zeros(np.append(par['td_to_hidden_dend_dims'],par['num_mw']), dtype=np.float32),
            'W_td_dend_g' : np.full(np.append(par['td_to_hidden_dend_dims'],par['num_mw']), par['cascade_strength'], dtype=np.float32),
            'W_td_soma_u' : np.zeros(np.append(par['td_to_hidden_soma_dims'],par['num_mw']), dtype=np.float32),
            'W_td_soma_g' : np.full(np.append(par['td_to_hidden_soma_dims'],par['num_mw']), par['cascade_strength'], dtype=np.float32),

            'W_rnn_dend_u' : np.zeros(np.append(par['hidden_to_hidden_dend_dims'],par['num_mw']), dtype=np.float32),
            'W_rnn_dend_g' : np.full(np.append(par['hidden_to_hidden_dend_dims'],par['num_mw']), par['cascade_strength'], dtype=np.float32),
            'W_rnn_soma_u' : np.zeros(np.append(par['hidden_to_hidden_soma_dims'],par['num_mw']), dtype=np.float32),
            'W_rnn_soma_g' : np.full(np.append(par['hidden_to_hidden_soma_dims'],par['num_mw']), par['cascade_strength'], dtype=np.float32),

            'W_out_u' : np.zeros((par['n_output'], par['n_hidden'], par['num_mw']), dtype=np.float32),
            'W_out_g' : np.full((par['n_output'], par['n_hidden'], par['num_mw']), par['cascade_strength'], dtype=np.float32),
        }

global mws
mws = MetaweightSet()

def adjust(x, name):
    for index, w in np.ndenumerate(x):
        x[index] = omega_prime(mws.mws_dict[name + "_u"][index], w, mws.mws_dict[name + "_g"][index], par['alpha_mw'])
        mws.mws_dict[name + "_u"][index] = omega(mws.mws_dict[name + "_u"][index], w, mws.mws_dict[name + "_g"][index], par['alpha_mw'])
    return x

# Calculate new metaweights

def omega(u, weight, g, alpha):

    for i in range(len(u)-1):
        if i == 0:
            u[i] = weight        
        elif i == len(u)-1:
            u[i] = (-1*g[i-1]*(u[i]-u[i-1])) * alpha
        else:
            u[i] = (g[i]*(u[i+1]-u[i]) - g[i-1]*(u[i]-u[i-1])) * alpha

    return u

# Calculate new weight, taking metaweights into consideration

def omega_prime(previous_u, weight, g, alpha):

	new_weight = g[0]*(previous_u[1] - weight)*alpha + weight

	return new_weight
