import numpy as np
from parameters import *
import model
import sys


task_list = ['chunking']


def try_model(gpu_id):

    try:
        # Run model
        model.main(gpu_id)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

# Second argument will select the GPU to use
# Don't enter a second argument if you want TensorFlow to select the GPU/CPU
try:
    gpu_id = sys.argv[1]
    print('Selecting GPU ', gpu_id)
except:
    gpu_id = None


num_pulses = [3, 4, 5, 6, 7, 8, 10, 15]

for task in task_list:
    for n in num_pulses:
        print('Training network on ', task,' task, ', n, ' pulses...')
        save_fn = task + '_' + str(n) + '.pkl'
        updates = {'trial_type': task, 'save_fn': save_fn, 'num_pulses': n}
        update_parameters(updates)
        try_model(gpu_id)
