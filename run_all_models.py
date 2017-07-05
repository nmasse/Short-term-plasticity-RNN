import numpy as np
from parameters import *
import model

task_list = ['DMS', 'DMRS45', 'DMRS90', 'DMRS180', 'DMC', 'DMS+DMRS', 'ABCA', 'ABBA', 'dualDMS']
models_per_task = 25

for task in task_list:
    for j in range(models_per_task):
        save_fn = task + '_' + str(j) + '.pkl'
        updates = {'trial_type': task, 'save_fn': save_fn}
        update_parameters(updates)
        model.train_and_analyze()
