from parameters import *
import hud
import model
import threading
import psutil
import sys
import os

def run_model():
    if par['use_HUD']:
        t_HUD = threading.Thread(target=hud.main, args=(switch,))
        t_HUD.start()
    else:
        model.main()


def switch(iteration, savename):

    if (iteration+1)%par['switch_rule_iteration'] == 0:
        new_allowed_rule = (par['allowed_rules'][0]+1)%par['num_rules']
        par['allowed_rules'] = [allowed_rule]





def script():
    par['df_num'] = '0004'

    run_model()


script()
