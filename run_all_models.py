import numpy as np
from parameters import *
import model
import sys

task_list = ['DMS']


for task in task_list:
    for j in range(1):
        print('Training network on ', task,' task, network model number ', j)
        save_fn = task + str(j) + '.pkl'
        updates = {'trial_type': task, 'save_fn': save_fn}
        update_parameters(updates)

        # Keep the try-except clauses to ensure proper GPU memory release
        if par['gpu']:
            try:
                # GPU designated by first argument (must be integer 0-3)
                try:
                    print('Selecting GPU ',  sys.argv[1])
                    assert(int(sys.argv[1]) in [0,1,2,3])
                except AssertionError:
                    quit('Error: Select a valid GPU number.')

                # Run model
                model.train_and_analyze(sys.argv[1])
            except KeyboardInterrupt:
                quit('Quit by KeyboardInterrupt')
        else:
            model.train_and_analyze(0)


# Command for observing python processes:
# ps -A | grep python
