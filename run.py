from parameters import *
import realtime_tracker as rt
import model
import threading
import psutil
import sys
import os

def switch(iteration, prev_switch_iteration, savename):
    if iteration == (prev_switch_iteration + 10):
        if par['allowed_categories'] == [0]:
            par['allowed_categories'] = [1]
            print("Switching to category 1.\n")
            with open(savename, 'a') as f:
                f.write('Switching to category 1.\n')
            return iteration
        elif par['allowed_categories'] == [1]:
            par['allowed_categories'] = [0]
            print("Switching to category 0.\n")
            with open(savename, 'a') as f:
                f.write('Switching to category 0.\n')
            return iteration
        else:
            print("ERROR: Bad category.")
            quit()
    else:
        return prev_switch_iteration

def script():
    par['df_num'] = '0004'



    if par['use_GUI']:
        t_GUI = threading.Thread(target=rt.main)
    else:
        t_GUI = threading.Thread(target=None)
    t_main = threading.Thread(target=model.main(switch))
    t_GUI.start()
    t_main.start()

script()
