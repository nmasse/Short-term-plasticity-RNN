import numpy as np
from parameters import *
import model
import sys

<<<<<<< HEAD
=======

>>>>>>> fa4ae00f736b97432c6358f7a1a9c3a10ea02d9a
task_list = ['DMS']
delay_time = 2500

for task in task_list:
    for j in range(1,25,2):
        print('Training network on ', task,' task, network model number ', j)
        save_fn = task + '_delay2500_' + str(j) + '.pkl'
        updates = {'trial_type': task, 'save_fn': save_fn, 'delay_time': delay_time}
        update_parameters(updates)

        # Keep the try-except clauses to ensure proper GPU memory release
        try:
            # GPU designated by first argument (must be integer 0-3)
            try:
<<<<<<< HEAD
                print('Selecting GPU ',  sys.argv[1])
=======
                print('Selecting GPU ', sys.argv[1])
>>>>>>> 555cf8ede043a254cd4e86027ad08d329458e7f8
                assert(int(sys.argv[1]) in [0,1,2,3])
            except AssertionError:
                quit('Error: Select a valid GPU number.')

            # Run model
            model.train_and_analyze(int(sys.argv[1]))
        except KeyboardInterrupt:
            quit('Quit by KeyboardInterrupt')


# Command for observing python processes:
# ps -A | grep python
<<<<<<< HEAD
=======

>>>>>>> fa4ae00f736b97432c6358f7a1a9c3a10ea02d9a
