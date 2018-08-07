import numpy as np
from parameters import *
import model
import sys
from analysis import *

task = "chunking"

file_list = ['chunking_3_cue_off.pkl']

for file in file_list:
    print('Analyzing network...')
    save_fn = 'analysis_' + file
    analyze_model_from_file(file, savefile = save_fn, analysis = False)
