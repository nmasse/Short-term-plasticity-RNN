import numpy as np
from parameters import *

class MetaweightSet:

    def __init__(self):
        self.mws_dict = {
            'W_stim_dend_u' : np.zeros(np.append(par['input_to_hidden_dend_dims'],par['num_mw']), dtype=np.float32),
            'W_stim_dend_g' : np.zeros(np.append(par['input_to_hidden_dend_dims'],par['num_mw']), dtype=np.float32),
            'W_stim_soma_u' : np.zeros(np.append(par['input_to_hidden_soma_dims'],par['num_mw']), dtype=np.float32),
            'W_stim_soma_g' : np.zeros(np.append(par['input_to_hidden_soma_dims'],par['num_mw']), dtype=np.float32),

            'W_td_dend_u' : np.zeros(np.append(par['td_to_hidden_dend_dims'],par['num_mw']), dtype=np.float32),
            'W_td_dend_g' : np.zeros(np.append(par['td_to_hidden_dend_dims'],par['num_mw']), dtype=np.float32),
            'W_td_soma_u' : np.zeros(np.append(par['td_to_hidden_soma_dims'],par['num_mw']), dtype=np.float32),
            'W_td_soma_g' : np.zeros(np.append(par['td_to_hidden_soma_dims'],par['num_mw']), dtype=np.float32),

            'W_rnn_dend_u' : np.zeros(np.append(par['hidden_to_hidden_dend_dims'],par['num_mw']), dtype=np.float32),
            'W_rnn_dend_g' : np.zeros(np.append(par['hidden_to_hidden_dend_dims'],par['num_mw']), dtype=np.float32),
            'W_rnn_soma_u' : np.zeros(np.append(par['hidden_to_hidden_soma_dims'],par['num_mw']), dtype=np.float32),
            'W_rnn_soma_g' : np.zeros(np.append(par['hidden_to_hidden_soma_dims'],par['num_mw']), dtype=np.float32),

            'W_out_u' : np.zeros((par['n_hidden'], par['n_output'], par['num_mw']), dtype=np.float32),
            'W_out_g' : np.zeros((par['n_hidden'], par['n_output'], par['num_mw']), dtype=np.float32)
        }

global mws
mws = MetaweightSet()

def adjust(x, name):
    print(name, np.shape(x))

    return x

u = np.zeros(par['num_mw'])

# Calculate new metaweights

def omega(u, weight, C, g, alpha):

	for i in range(len(u)):
		if i == 0:
			u[i] = weight
		else:
			u[i] = 1/C[i] * (g[i]*(u[i+1]-u[i]) - g[i-1]*(u[i]-u[i-1])) * alpha

	return u

# Calculate new weight, taking metaweights into consideration

def omega_prime(previous_u, weight, C, g, alpha):

	new_weight = g[0]/C[0]*(previous_u[1] - weight)*alpha + weight

	return new_weight
