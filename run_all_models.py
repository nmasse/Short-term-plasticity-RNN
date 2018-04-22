import numpy as np
from parameters import *
import model
import sys

#task_list = ['DMS+DMRS+DMC']
task_list = ['DMS']


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


for j in range(20):
    for task in task_list:
        print('Training network on ', task,' task, network model number ', j)

        save_fn = task + '_' + str(j) + '.pkl'
        updates = {'trial_type': task, 'save_fn': save_fn}
        update_parameters(updates)
        try_model(gpu_id)
