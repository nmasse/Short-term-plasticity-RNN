import numpy as np
from parameters import *
import model
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import random as rd
import pickle

task_list = ['chunking']


def try_model(gpu_id):

    try:
        # Run model
        model.main(gpu_id)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

def plot(filename, num_pulses, savename):
    x = pickle.load(open('./savedir/'+filename, 'rb'))
    color = [(rd.uniform(0,1),rd.uniform(0,1),rd.uniform(0,1)) for i in range(num_pulses)]
    for i in range(num_pulses):
        plt.plot(np.mean(x['synaptic_sample_decoding'][0,i,:,:], axis = 0), color =color[i])
        plt.bar(np.array(x['timeline'])[np.arange(-2, -2*n-1, -2)], height = 0.04, width = 1, color = 'k')
        plt.bar(np.array(x['timeline'])[np.arange(1,2*n+1,2)], height = 0.04, width = 1, color = 'k')
        #bar(x, height, width, *, align='center', **kwargs)
    plt.savefig("./savedir/"+savename+"_synaptic.png")
    plt.close()

    for i in range(num_pulses):
        plt.plot(np.mean(x['neuronal_sample_decoding'][0,i,:,:], axis = 0), color =color[i])
        plt.bar(np.array(x['timeline'])[np.arange(-2, -2*n-1, -2)], height = 0.04, width = 1, color = 'k')
        plt.bar(np.array(x['timeline'])[np.arange(1,2*n+1,2)], height = 0.04, width = 1, color = 'k')
    plt.savefig("./savedir/"+savename+"_neuronal.png")
    plt.close()

    for i in range(num_pulses):
        plt.plot(np.mean(x['combined_decoding'][0,i,:,:], axis = 0), color =color[i])
        plt.bar(np.array(x['timeline'])[np.arange(-2, -2*n-1, -2)], height = 0.04, width = 1, color = 'k')
        plt.bar(np.array(x['timeline'])[np.arange(1,2*n+1,2)], height = 0.04, width = 1, color = 'k')
    plt.savefig("./savedir/"+savename+"_combined.png")
    plt.close()


# Second argument will select the GPU to use
# Don't enter a second argument if you want TensorFlow to select the GPU/CPU
try:
    gpu_id = sys.argv[1]
    print('Selecting GPU ', gpu_id)
except:
    gpu_id = None


num_pulses = [1,3, 8, 10, 12, 14, 18, 20,22,24,26,28,30,32,34,36]

for task in task_list:
    for n in num_pulses:
        for i in range(2):
            if i == 0:
                print('Training network on ', task,' task, ', n, ' pulses, with cue...')
                save_fn = task + '_' + str(n) + '_cue_on.pkl'
                updates = {'trial_type': task, 'save_fn': save_fn, 'num_pulses': n, 'order_cue': True}
                update_parameters(updates)
                try_model(gpu_id)
                plot(save_fn, n, savename=str(n)+'_pulses_cue_on')
            elif i == 1:
                print('Training network on ', task,' task, ', n, ' pulses, without cue...')
                save_fn = task + '_' + str(n) + '_cue_off.pkl'
                updates = {'trial_type': task, 'save_fn': save_fn, 'num_pulses': n, 'order_cue': False}
                update_parameters(updates)
                try_model(gpu_id)
                plot(save_fn, n, savename=str(n)+'_pulses_cue_off')
