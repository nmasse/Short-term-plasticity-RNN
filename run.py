from parameters import *
import model
import psutil
import sys
import os

def script():

    # Ignore "use compiled version of TensorFlow" errors
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

    # Allow for varied processor use (if on Windows)
    if os.name == 'nt':
        p = psutil.Process(os.getpid())
        p.cpu_affinity(par['processor_affinity'])
        print('Running with PID', os.getpid(), "on processor(s)", \
                str(p.cpu_affinity()) + ".", "\n")

    # Run model without massive error logs if kill early
    try:
        model.main()
    except KeyboardInterrupt:
        print('\nQuit by KeyboardInterrupt.\n')
        quit()

script()
