import numpy as np
import matplotlib
import scipy.stats as stats
import math
import cmath
import time
import json
import pickle
import model_saver
from parameters import *

# Need to update based on the number of category rules we allowed
global num_category_rule
num_category_rule = 2

# Receiver operating characteristic (ROC) curve
# Translated to python from Nick's Matlab code
# returns a value from 0 to 1, with 0.5 representing complete overlap
def calculate_roc(xlist, ylist):
    roc = 0
    unique_vals = np.unique(xlist)

    # Calculate ROC value
    for val in unique_vals:
        p1 = np.mean(xlist==val)
        p2 = np.mean(ylist>val) + 0.5*np.mean(ylist==val)
        roc = roc + p1*p2

    return roc


# Provide either JSON filename or category rule, target direction, and hidden_state data
# Calculates and returns ROC curve vale
def roc_analysis(time_stamps=[], cat_rule=[], dir_cat=[], target_dir=[], neuron=None, dendrites=None, exc=None, inh=None):

    # Reshaping the dendrite or hidden_state matrix + pre-allocate roc matrix
    # dendrite = (num_batches x time x n_hidden x den_per_unit x batch_train_size)
    #          => (n_hidden x dend_per_unit x time x trials)
    # neuron = (num_batches x time x n_hidden x batch_train_size)
    #        => (n_hidden x time x trials)
    if par['neuron']:
        num_batches, num_time, num_hidden, batch_size = neuron.shape            # need to double check
        neuron = np.stack(neuron, axis=2)
        neuron = np.stack(neuron, axis=1)
        neuron = np.reshape(neuron, (num_hidden, num_time, batch_size*num_batches))
        roc_neuron = np.ones([num_hidden, len(time_stamps), num_category_rule, num_category_rule]) * 0.5
    if par['dendrites']:
        dendrites = np.stack(dendrites, axis=4)
        dendrites = np.transpose(dendrites,(1,2,0,3,4))
        num_hidden, num_dend, num_time, batch_size, num_batches = dendrites.shape
        dendrites = np.reshape(dendrites, (num_hidden, num_dend, num_time, batch_size*num_batches))
        roc_dend = np.ones([num_hidden, num_dend, len(time_stamps), num_category_rule, num_category_rule]) * 0.5
    if par['exc']:
        exc = np.stack(exc, axis=4)
        exc = np.transpose(exc,(1,2,0,3,4))
        exc = np.reshape(exc, (num_hidden, num_dend, num_time, batch_size*num_batches))
        roc_exc = np.ones([num_hidden, num_dend, len(time_stamps), num_category_rule, num_category_rule]) * 0.5
    if par['inh']:
        inh = np.stack(inh, axis=4)
        inh = np.transpose(inh,(1,2,0,3,4))
        inh = np.reshape(inh, (num_hidden, num_dend, num_time, batch_size*num_batches))
        roc_inh = np.ones([num_hidden, num_dend, len(time_stamps), num_category_rule, num_category_rule]) * 0.5



    # Get roc value
    # roc = (num_hidden x num_dend x num_time_stamps x num_category_rule x num_category_rule)
    t1=time.time()
    for c1 in range(num_category_rule):
        for c2 in range(num_category_rule):
            ind0 = np.where((dir_cat[c2]==0) * (cat_rule == c1))[0]
            if len(ind0!=0):
                ind1 = np.where((dir_cat[c2]==1) * (cat_rule == c1))[0]
                for n in range(num_hidden):
                    for t in range(len(time_stamps)):
                        if par['neuron']:
                            roc_neuron[n, t, c1, c2]  = calculate_roc(neuron[n, time_stamps[t], ind0], neuron[n, time_stamps[t], ind1])
                        if any([par['dendrites'], par['exc'], par['inh']]):
                            for d in range(num_dend):
                                if par['dendrites']:
                                    roc_dend[n, d, t, c1, c2]  = calculate_roc(dendrites[n, d, time_stamps[t], ind0], dendrites[n, d, time_stamps[t], ind1])
                                if par['exc']:
                                    roc_exc[n, d, t, c1, c2]  = calculate_roc(exc[n, d, time_stamps[t], ind0], exc[n, d, time_stamps[t], ind1])
                                if par['inh']:
                                    roc_inh[n, d, t, c1, c2]  = calculate_roc(inh[n, d, time_stamps[t], ind0], inh[n, d, time_stamps[t], ind1])



    # roc's = (num_hidden x num_dend x num_time_stamps x num_category_rule x num_category_rule)
    # roc_rectified = (num_category_rule x num_category_rule x num_time_stamps x num_hidden x num_dend)
    # percentile contains 90th percentile value of the roc_rectified
    print('ROC DENDRITES ', time.time()-t1)
    if par['neuron']:
        roc_neuron_rect = np.reshape(np.absolute(0.5-roc_neuron), (num_hidden*len(time_stamps), num_category_rule, num_category_rule))
        percentile_neuron = np.zeros([num_category_rule, num_category_rule])
        percentile_neuron = np.percentile(roc_rectified[:,:,:], 90, axis=0)
    if par['dendrites']:
        roc_dend_rect = np.reshape(np.absolute(0.5-roc_dend), (num_hidden*num_dend*len(time_stamps),num_category_rule,num_category_rule))
        percentile_dend = np.zeros([num_category_rule, num_category_rule])
        percentile_dend = np.percentile(roc_rectified[:,:,:], 90, axis=0)
    if par['exc']:
        roc_dend_exc = np.reshape(np.absolute(0.5-roc_exc), (num_hidden*num_dend*len(time_stamps),num_category_rule,num_category_rule))
        percentile_exc = np.zeros([num_category_rule, num_category_rule])
        percentile_exc = np.percentile(roc_rectified[:,:,:], 90, axis=0)
    if par['inh']:
        roc_dend_inh = np.reshape(np.absolute(0.5-roc_inh), (num_hidden*num_dend*len(time_stamps),num_category_rule,num_category_rule))
        percentile_inh = np.zeros([num_category_rule, num_category_rule])
        percentile_inh = np.percentile(roc_rectified[:,:,:], 90, axis=0)

    roc = {'neurons': (roc_neuron, percentile_neuron),
            'dendrites': (roc_dend, percentile_dend), 
            'dendrite_exc': (roc_exc, percentile_exc), 
            'dendrite_inh': (roc_inh, percentile_inh)
            }

    return roc



# Calculate and return ANOVA values based on sample inputs (directions, images, etc.)
def anova_analysis():
    return None



# TO-DO
# Move pre-processing into the analysis
# Set category rule as a flag
def analysis(filename=None, data=None):
# def roc_analysis(filename=None, cat_rule=[], target_dir=[], hidden=[], dendrites=[], exc=[], inh=[]):

    # Get hidden_state value from parameters if filename is not provided
    if filename is None:
        print("ROC: Gathering data from model_results")
        data = data
        time_stamps = par['time_stamps']
    else:
        print("ROC: Gathering data from JSON file")
        data = model_saver.json_load(savedir=filename)        
        time_stamps = np.array(data['params']['time_stamps'])

    category_rule = np.array(data['category_rule'])
    attended_dir = np.array(data['attended_sample_dir'])
    hidden_state = np.array(data['hidden_state'])
    dendrites = np.array(data['dendrite_state'])
    exc = np.array(data['dendrite_exc_input'])
    inh = np.array(data['dendrite_inh_input'])
    time_stamps = time_stamps//par['dt']



    # Pre-processing






    # Need update based on the category rules we are using
    # Rule 0: up == True, down == False
    # Rule 1: left == True, right == False
    trials = len(category_rule)
    direction_cat = np.zeros([num_category_rule, trials])
    for ind in range(trials):
        direction_cat[0, ind] = attended_dir[ind] in [0,1,6,7]
        direction_cat[1, ind] = attended_dir[ind] in [4,5,6,7]


    # Get roc values
    result = {'roc': None, 'ANOVA': None}
    result['roc'] = roc_analysis(time_stamps=time_stamps, cat_rule=category_rule, dir_cat=direction_cat, target_dir=attended_dir, \
                                   neuron=hidden_state, dendrites=dendrites, exc=exc, inh=inh)
    result['ANOVA'] = anova_analysis()

   
    # Save analysis result
    with open('.\savedir\dend_analysis_%s.pkl' % time.strftime('%H%M%S-%Y%m%d'), 'wb') as f:
        pickle.dump(result, f)
    print("ROC: Done pickling\n")

    return result


# GET ROC CALCULATION DONE FOR: hidden_state_hist, dendrites_hist, dendrites_inputs_exc_hist, dendrites_inputs_inh_hist
# roc_analysis(filename="savedir/model_data.json", time_stamps=[1100, 1200])
