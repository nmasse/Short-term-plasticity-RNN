from parameters import *
import model
import psutil
import sys
import os

def script():
    try:
        model.main()
    except KeyboardInterrupt:
        quit()

script()
