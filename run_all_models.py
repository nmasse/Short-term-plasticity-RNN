import numpy as np
from parameters import *
import model
import sys

model.train_and_analyze(str(sys.argv[1]))
