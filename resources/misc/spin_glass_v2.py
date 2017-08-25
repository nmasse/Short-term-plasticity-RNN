import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation

size = 35

T = 1.
k = 0.1 #1.38064852e-23
beta = T**-1 #(k*T)**-1
spin_up = np.array([1.,0.])
spin_down = np.array([0.,1.])

def make_sigma():
    template = np.random.choice([-1, 1], [size, size])
    sigma = np.zeros([size, size, 2])
    for i, j in itertools.product(range(size), range(size)):
        sigma[i,j,:] = spin_up if template[i,j]==1 else spin_down
    return sigma

def make_J():
    return np.random.normal(size=[size, size])

def make_h():
    h = np.zeros([size, size])
    template = np.arange(size)/size
    h += 2*template
    return h

def site_probability(connections, neighbors, h):
    H_site = -np.sum([c*n for c, n in zip(connections, neighbors)], axis=0) - h
    return np.exp(-beta*H_site), H_site

def normalization_constant(H):
    return np.sum(np.exp(-beta*H))

def normalize_to_one(p):
    return p / p.sum(keepdims=True)

def calc_new_state(sigma, J, h):
    global T
    T -= 0.01

    new_sigma   = np.zeros([size, size, 2])
    sample      = np.random.rand(size, size)

    for i, j in itertools.product(range(size), range(size)):
        neighbors   = [sigma[i,(j+1)%size], sigma[i,j-1], sigma[(i+1)%size,j], sigma[i-1,j]]
        connections = [J[i,(j+1)%size]+J[i,j], J[i,j-1]+J[i,j], J[(i+1)%size,j]+J[i,j], J[i-1,j]+J[i,j]]
        prob, H     = site_probability(connections, neighbors, h[i,j])

        if sample[i,j] > normalize_to_one(prob)[0]:
            new_sigma[i,j] = spin_up
        else:
            new_sigma[i,j] = spin_down

    return new_sigma

fig = plt.figure()
im = plt.imshow(make_sigma()[:,:,0], cmap='magma', vmin=0, vmax=1)

sigma   = make_sigma()
J       = make_J()
h       = make_h()

def update_fig(j):
    global sigma
    sigma = calc_new_state(sigma, J, h)
    im.set_array(sigma[:,:,0])
    return [im]

ani = animation.FuncAnimation(fig, update_fig, blit=True, interval=50, repeat=False)
plt.show()
