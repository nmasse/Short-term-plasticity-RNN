import numpy as np
from parameters import *
import model
import sys

task_list = ['ABCA']

for task in task_list:
    j = sys.argv[1]
    print('Training network on ', task,' task, network model number ', j)

    update_parameters(updates)
    model.train_and_analyze()
