import numpy as np
import itertools

def prune_connections(x, p):

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if np.random.rand() > p:
                x[i,j] = 0
    return x

def get_dend(p2, p3, p_td, p_td2):
    n_0, n_1, n_2, n_3 = 24, 80, 80, 1
    p_vip, p_som = p2, p3
    beta = 1.0
    scale = 1.2*(n_1/40)+0.55

    num_iters = 1000
    dend = np.zeros((num_iters, 2))

    for i in range(num_iters):
        W_vip = np.ones((n_2, n_1))
        W_som = np.ones((n_3, n_2))
        W_vip = prune_connections(W_vip, p_vip)
        W_som = prune_connections(W_som, p_som)


        W_td = np.zeros((n_1, n_0))
        W_td2 = np.zeros((n_2, n_0))
        for k in range(n_0//2):
            for m in range(n_1//2):
                W_td[m,k] = 1
                W_td[m+(n_1//2),k+(n_0//2)] = 1
        for k in range(n_0//2):
            for m in range(n_2//2):
                W_td2[m,k] = 1
                W_td2[m+(n_2//2),k+(n_0//2)] = 1
        W_td = prune_connections(W_td, p_td)
        W_td2 = prune_connections(W_td2, p_td2)


        for j in range(2):
            if j == 0:
                td = np.zeros((n_0,1))
                td[:n_0//2,0] = 2
            else:
                td = np.zeros((n_0,1))
                td[n_0//2:,0] = 2

            VIP = np.matmul(W_td, td)
            vip = np.matmul(W_vip, VIP)
            td2 = np.matmul(W_td2, td)
            SOM = np.maximum(0,beta - vip/scale + td2)
            dend[i,j] = np.matmul(W_som, SOM)<1

    print('Dendrite analysis...')
    print(np.mean(dend[:,0]), np.mean(dend[:,1]), np.mean(dend[:,0]*dend[:,1]), np.mean(dend[:,0])*np.mean(dend[:,1]))

# p2, p3, p_td, p_td2
# vip->som, som->dend, td->vip, td->som
get_dend(0.275, 0.35, 0.2, 0.5)


