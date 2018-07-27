import numpy as np
from parameters import *
import model
import sys

task_list = ['DMSvar']
rule_list = [2, 1, 0]


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


for j in range(1):
    for task in task_list:
        for rule in rule_list:
            print('Training network on ', task,' task, network model number ', j)

            save_fn = task + '_' + str(j) + '_rule_' + str(rule) + '.pkl'
            updates = {'trial_type':task, 'save_fn':save_fn, 'rule':rule}
            update_parameters(updates)
            try_model(gpu_id)
