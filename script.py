import os
import tensorflow as tf
import numpy as np
import stimulus
import importlib
import pickle
import model_den as model
import parameters_den

importlib.reload(stimulus)
importlib.reload(parameters_den)
importlib.reload(model)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

p = parameters_den.Parameters()
p.params['synapse_config'] = None
p.params['save_dir'] = 'C:/Users/Gregory/Desktop/dendritics/savedir/'
p.params['save_fn'] = 'DMS_stp_delay_' + str(0) + '_' + str(0) + '.pkl'
p.params['ckpt_save_fn'] = 'model_' + str(0) + '.ckpt'
p.params['ckpt_load_fn'] = 'model_' + str(0) + '.ckpt'
p.params['possible_rules']=[0]
p.params['load_previous_model'] = False
tf.reset_default_graph()
rnn = None

params = p.return_params()
model.main(params)
