import numpy as np
import itertools

rules   = 2
locs    = 4

rule_tuned = 3 * rules
locs_tuned = 3 * locs

n_td    = rule_tuned + locs_tuned
n_vip   = 7
n_som   = 9
n_den   = 36


###
### ITERATE OVER THESE SEVEN PARAMETER
### Find those in the 16% region (6 dendrites per neuron)
###

W_td_c  = 0.3
W_vip_c = 0.7
W_som_c = 1.5

W_td_p  = 0.2
W_vip_p = 0.2
W_som_p = 0.3

b = 1.1

iterations = 10

##############################

def td_vectors(rules, locs, rule_tuned, locs_tuned):
    """
    Returns a set of all possible td vectors
    """

    rule_tuning = np.zeros([rules, rule_tuned])
    m = rule_tuned//rules
    if rule_tuned == 0:
        return np.array([0]*rules)
    elif m == 0:
        print('ERROR: Use more rule neurons than rules.')
        quit()

    for r in range(rules):
        if r == rules-1:
            rule_tuning[r, r*m:] = 1
        else:
            rule_tuning[r, r*m:r*m+m] = 1

    locs_tuning = np.zeros([locs, locs_tuned])
    m = locs_tuned//locs
    if locs_tuned == 0:
        return np.array([0]*locs)
    elif m == 0:
        print('ERROR: Use more loc neurons than locs.')
        quit()

    for r in range(locs):
        if r == locs-1:
            locs_tuning[r, r*m:] = 1
        else:
            locs_tuning[r, r*m:r*m+m] = 1

    td_vector_set = np.zeros([rules*locs, rule_tuned+locs_tuned])
    for r, l in itertools.product(range(rules), range(locs)):
        td_vector_set[r*locs+l] = np.concatenate([rule_tuning[r], locs_tuning[l]], axis=0)
    return td_vector_set


def establish_connections(m, c, p):
    """
    Establishes random connections in m of value c, based on probability p
    """
    for i, j in itertools.product(range(np.shape(m)[0]), range(np.shape(m)[1])):
        if np.random.rand() <= p:
            m[i][j] = c

    return m


def run_calculation(td, W_td, W_vip, W_som):

    VIP     = np.matmul(W_td, td)
    SOM     = np.matmul(W_vip, VIP)
    den     = np.matmul(W_som, SOM)

    return den

iteration_set = np.zeros([iterations, rules*locs, n_den])
for i in range(iterations):
    td_set = td_vectors(rules, locs, rule_tuned, locs_tuned)

    W_td    = np.zeros([n_vip, n_td])
    W_vip   = np.zeros([n_som, n_vip])
    W_som   = np.zeros([n_den, n_som])

    W_td = establish_connections(W_td, W_td_c, W_td_p)
    W_vip = establish_connections(W_vip, W_vip_c, W_vip_p)
    W_som = -establish_connections(W_som, W_som_c, W_som_p)+b

    # Yields one iteration's worth of dendrite inhibition possibilites
    den = np.zeros([rules*locs, n_den])
    for n in range(rules*locs):
        den[n] = run_calculation(td_set[n], W_td, W_vip, W_som)
    iteration_set[i] = den

for i, j, k in itertools.product(range(iterations), range(rules*locs), range(n_den)):
    if iteration_set[i,j,k] < 1:
        iteration_set[i,j,k] = 1
    else:
        iteration_set[i,j,k] = 0

print('Open:', np.round(np.sum(iteration_set)/np.size(iteration_set), 2))

activity = np.zeros([rules*locs, n_den])
for i, j, k in itertools.product(range(iterations), range(rules*locs), range(n_den)):
    if iteration_set[i,j,k] == 1:
        activity[j,k] += 1
    else:
        pass

print(np.int8(activity))
