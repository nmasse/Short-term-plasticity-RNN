import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Assume 2D for now
size = 35
T = 500
k = 1 #1.38064852e-23
beta = (k*T)**-1

def make_sigma(size):
    return np.random.choice([-1., 1.], [size, size])

def make_J(size):
    # Ferromagnetic model (0, 1)
    return np.abs(np.random.normal(size=[size-1, size])), np.abs(np.random.normal(size=[size, size-1]))

def make_h(size, scale=1):
    base = np.zeros([size, size])
    template = 0*np.arange(size)/(size)
    for i in range(size):
        base[i] = template
    return base

def config_prob(connections, neighbors):
    H = 0
    for c, n in zip(connections, neighbors):
        H += c*n
    return H

def update_sigma(sigma, J, h, size):
    J_d1, J_d2 = J

    updated_sigma = np.zeros(np.shape(sigma))
    probs = np.zeros(np.shape(sigma))
    for i, j in itertools.product(range(size), range(size)):
        connections = [J_d1[i%(size-1),j], J_d1[(i+1)%(size-1),j%(size-1)], J_d2[i,j%(size-1)], J_d2[i%(size-1),(j+1)%(size-1)]]
        neighbors   = [sigma[(i+1)%size,j], sigma[i-1,j], sigma[i,(j+1)%size], sigma[i,j-1]]
        probs[i,j]  = config_prob(connections, neighbors)

    normalization = np.sum(np.exp(-beta*probs))
    probs /= normalization

    print(probs)
    quit()

    return updated_sigma

def state_gen():
    cnt = 0
    states = []
    while cnt < 1000:
        if cnt == 0:
            sigma   = make_sigma(size)
            J       = make_J(size)
        h = make_h(size, scale)
        sigma = update_sigma(sigma, J, h, cnt**2)
        cnt += 1
        yield sigma

fig = plt.figure()
im = plt.imshow(make_sigma(size), cmap='magma', vmin=0, vmax=1)

sigma   = make_sigma(size)
J       = make_J(size)
h       = make_h(size)

def update_fig(j):
    global sigma
    sigma = update_sigma(sigma, J, h, size)
    im.set_array(sigma)
    return [im]

ani = animation.FuncAnimation(fig, update_fig, blit=True, interval=50, repeat=False)
plt.show()
