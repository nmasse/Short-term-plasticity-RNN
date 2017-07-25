import numpy as np
from parameters import *

def main():
    pass

u = np.zeros(par['num_mw'])

# Calculate new metaweights

def omega(u, weight, C, g, alpha):

	for i in range(len(u)):
		if i = 0:
			u[i] = weight
		else:
			u[i] = 1/C[i] * (g[i]*(u[i+1]-u[i]) - g[i-1]*(u[i]-u[i-1])) * alpha

	return u

# Calculate new weight, taking metaweights into consideration

def omega_prime(previous_u, weight, C, g, alpha):

	new_weight = g[0]/C[0]*(previous_u[1] - weight)*alpha + weight

	return new_weight