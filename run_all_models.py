import numpy as np
from parameters import *
import model
import sys

task_list = ['DMRS45']

    #j = sys.argv[1]
for j in range(0,999):
    for task in task_list:
        print('Training network on ', task,' task, network model number ', j)
        save_fn = task + '_' + str(j) + '.pkl'
        updates = {'trial_type': task, 'save_fn': save_fn, 'decoding_test_mode': False}
        update_parameters(updates)
        model.train_and_analyze()
