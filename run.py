from parameters import *
import model
import psutil
import sys
import os

def script():
    try:
        model.main()
    except KeyboardInterrupt:
        print('\nQuit by KeyboardInterrupt.\n')
        quit()

script()
