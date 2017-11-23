import numpy as np
from parameters import *
import model
import sys

<<<<<<< HEAD
task_list = ['DMRS45']

    #j = sys.argv[1]
for j in range(0,999):
    for task in task_list:
        print('Training network on ', task,' task, network model number ', j)
        save_fn = task + '_' + str(j) + '.pkl'
        updates = {'trial_type': task, 'save_fn': save_fn, 'decoding_test_mode': False}
        update_parameters(updates)
        model.train_and_analyze()
=======
task_list = ['DMS']

for task in task_list:
    for j in range(25):
        print('Training network on ', task,' task, network model number ', j)
        save_fn = task + '_' + str(j) + '.pkl'
        updates = {'trial_type': task, 'save_fn': save_fn}
        update_parameters(updates)

        # Keep the try-except clauses to ensure proper GPU memory release
        try:
            # GPU designated by first argument (must be integer 0-3)
            try:
                assert(sys.argv[1] in [0,1,2,3])
            except AssertionError:
                quit('Error: Select a valid GPU number.')

            # Run model
            model.train_and_analyze(sys.argv[1])
        except KeyboardInterrupt:
            quit('Quit by KeyboardInterrupt')


# Command for observing python processes:
# ps -A | grep python
>>>>>>> dd36fb338da4e1e77a5622fa83c1f425bddec500
