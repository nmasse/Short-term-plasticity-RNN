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


num_pulses = [2, 6, 8, 10, 12, 14, 18, 20]

for task in task_list:
    for n in num_pulses:
        for i in range(2):
            if i == 0:
                print('Training network on ', task,' task, ', n, ' pulses, with cue...')
                save_fn = task + '_' + str(n) + '_cue_on.pkl'
                updates = {'trial_type': task, 'save_fn': save_fn, 'num_pulses': n, 'order_cue': True}
                update_parameters(updates)
                try_model(gpu_id)
            elif i == 1:
                print('Training network on ', task,' task, ', n, ' pulses, without cue...')
                save_fn = task + '_' + str(n) + '_cue_off.pkl'
                updates = {'trial_type': task, 'save_fn': save_fn, 'num_pulses': n, 'order_cue': False}
                update_parameters(updates)
                try_model(gpu_id)
