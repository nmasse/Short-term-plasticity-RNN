import numpy as np
import json
import model_saver
import pickle
import matplotlib
import math
import cmath
import time
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
def roc_neuron(filename=None, time_stamps=[], cat_rule=[], dir_cat=[], target_dir=[], hidden=[]):

    # Parameters
    category_rule = cat_rule
    attended_dir = target_dir
    hidden_state = hidden
    time_stamps = time_stamps
    direction_cat = dir_cat

    # Dimensions
    trials = len(category_rule)
    num_time = len(hidden_state[0])
    num_hidden = len(hidden_state[0][0])
    batch_size = len(hidden_state[0][0][0])

    # Reshaping the hidden_state matrix
    # hidden_state = (num_batches x time x n_hidden x batch_train_size)
    # hidden = (n_hidden x time x trials)
    hidden_state = np.stack(hidden_state, axis=2)
    hidden_state = np.stack(hidden_state, axis=1)
    hidden_state = np.reshape(hidden_state, (num_hidden, num_time, trials))


    # Get calculate time stamp based on dt
    for i in range(len(time_stamps)):
        time_stamps[i] = time_stamps[i]//par['dt']

    # Get roc value
    # roc = (num_hidden x num_time_stamps x num_category_rule x num_category_rule)
    t1=time.time()
    roc = np.ones([num_hidden, len(time_stamps), num_category_rule, num_category_rule]) * 0.5
    for c1 in range(num_category_rule):
        for c2 in range(num_category_rule):
            ind0 = np.where((direction_cat[c2]==0) * (category_rule == c1))[0]
            if len(ind0!=0):
                ind1 = np.where((direction_cat[c2]==1) * (category_rule == c1))[0]
                for n in range(num_hidden):
                    for t in range(len(time_stamps)):
                        roc[n, t, c1, c2]  = calculate_roc(hidden_state[n, time_stamps[t], ind0], hidden_state[n, time_stamps[t], ind1])
    print('ROC NEURON ', time.time()-t1)


    # roc = (num_hidden x num_time_stamps x num_category_rule x num_category_rule)
    # roc_rectified = (num_category_rule x num_category_rule x num_time_stamps x num_hidden)
    # percentile = (num_category_rule x num_category_rule)
    roc_rectified = np.stack(roc, axis=3)
    roc_rectified = np.stack(roc_rectified, axis=2)
    roc_rectified = np.absolute(roc_rectified-0.5)

    percentile = np.zeros([num_category_rule, num_category_rule])
    for i in range(num_category_rule):
        for j in range(num_category_rule):
            percentile[i,j] = np.percentile(roc_rectified[i,j], 90)

    return roc, percentile


def roc_dendrites(filename=None, time_stamps=[], cat_rule=[], dir_cat=[], target_dir=[], dendrites=[]):

    # Parameters
    """
    category_rule = cat_rule
    attended_dir = target_dir
    dendrites = dendrites
    time_stamps = time_stamps
    direction_cat = dir_cat
    """



    # Reshaping the dendrite matrix
    # dendrite = (num_batches x time x n_hidden x den_per_unit x batch_train_size)
    # dendrite = (n_hidden x dend_per_unit x time x trials)
    dendrites = np.stack(dendrites, axis=4)
    dendrites = np.transpose(dendrites,(1,2,0,3,4))
    num_hidden, num_dend, num_time, batch_size, num_batches = dendrites.shape
    dendrites = np.reshape(dendrites, (num_hidden,num_dend, num_time, trials)))

    # Get calculate time stamp based on dt
    time_stamps = time_stamps//par['dt']

    # Get roc value
    # roc = (num_hidden x num_dend x num_time_stamps x num_category_rule x num_category_rule)
    t1=time.time()
    roc = np.ones([num_hidden, num_dend, len(time_stamps), num_category_rule, num_category_rule]) * 0.5
    for c1 in range(num_category_rule):
        for c2 in range(num_category_rule):
            ind0 = np.where((dir_cat[c2]==0) * (cat_rule == c1))[0]
            if len(ind0!=0):
                ind1 = np.where((dir_cat[c2]==1) * (cat_rule == c1))[0]
                for n in range(num_hidden):
                    for d in range(num_dend):
                        for t in range(len(time_stamps)):
                            roc[n, d, t, c1, c2]  = calculate_roc(dendrites[n, d, time_stamps[t], ind0], dendrites[n, d, time_stamps[t], ind1])

    print('ROC DENDRITES ', time.time()-t1)
    # roc = (num_hidden x num_dend x num_time_stamps x num_category_rule x num_category_rule)
    # roc_rectified = (num_category_rule x num_category_rule x num_time_stamps x num_hidden x num_dend)
    roc_rectified = np.reshape(np.abs(0.5-roc),(num_hidden*num_dend*num_time_stamps,num_category_rule,num_category_rule))
    # percentile = (num_category_rule x num_category_rule)
    roc_rectified = np.stack(roc, axis=4)
    roc_rectified = np.stack(roc_rectified, axis=4)
    roc_rectified = np.stack(roc_rectified, axis=2)


    roc_rectified = np.absolute(roc_rectified-0.5)
    print('ROC DENDRITES STACKED ', time.time()-t1)

    percentile = np.zeros([num_category_rule, num_category_rule])
    for i in range(num_category_rule):
        for j in range(num_category_rule):
            percentile[i,j] = np.percentile(roc_rectified[:,i,j], 90)

    return roc, percentile

def roc_analysis(filename=None, time_stamps=[], cat_rule=[], target_dir=[], hidden=[], dendrites=[], exc=[], inh=[]):

    # Get hidden_state value from parameters if filename is not provided
    if filename is None:
        print("ROC: Gathering data from model_results")
        category_rule = np.array(cat_rule)
        attended_dir = np.array(target_dir)
        hidden_state = np.array(hidden)
        dendrites = np.array(dendrites)
        exc = np.array(exc)
        inh = np.array(inh)
        time_stamps = np.array(time_stamps)
    else:
        print("ROC: Gathering data from JSON file")
        data = model_saver.json_load(savedir=filename)
        category_rule = np.array(data['category_rule'])
        attended_dir = np.array(data['attended_sample_dir'])
        hidden_state = np.array(data['hidden_state'])
        dendrites = np.array(data['dendrite_state'])
        exc = np.array(data['dendrite_exc_input'])
        inh = np.array(data['dendrite_inh_input'])
        time_stamps = np.array(time_stamps)

    # Need update based on the category rules we are using
    # Rule 0: up == True, down == False
    # Rule 1: left == True, right == False
    trials = len(category_rule)
    direction_cat = np.zeros([num_category_rule, trials])
    for ind in range(trials):
        direction_cat[0, ind] = attended_dir[ind] in [0,1,6,7]
        direction_cat[1, ind] = attended_dir[ind] in [4,5,6,7]


    # Get calculate time stamp based on dt
    for i in range(len(time_stamps)):
        time_stamps[i] = time_stamps[i]//par['dt']

    print(time_stamps)
    # Get roc values
    result = {'neurons': None, 'dendrites': None, 'dendrite_exc': None, 'dendrite_inh': None}
    result['neurons'] = roc_neuron(time_stamps=time_stamps, cat_rule=category_rule, \
                                   dir_cat=direction_cat, target_dir=attended_dir, hidden=hidden_state)
    result['dendrites'] = roc_dendrites(time_stamps=time_stamps, cat_rule=category_rule, \
                                   dir_cat=direction_cat, target_dir=attended_dir, dendrites=dendrites)
    result['dendrite_exc'] = roc_dendrites(time_stamps=time_stamps, cat_rule=category_rule, \
                                   dir_cat=direction_cat, target_dir=attended_dir, dendrites=exc)
    result['dendrite_inh'] = roc_dendrites(time_stamps=time_stamps, cat_rule=category_rule, \
                                   dir_cat=direction_cat, target_dir=attended_dir, dendrites=inh)


    # Save analysis result
    with open('.\savedir\dend_analysis_%s.pkl' % time.strftime('%H%M%S-%Y%m%d'), 'wb') as f:
        pickle.dump(result, f)
    print("ROC: Done pickling\n")

    return result


# GET ROC CALCULATION DONE FOR: hidden_state_hist, dendrites_hist, dendrites_inputs_exc_hist, dendrites_inputs_inh_hist
# roc_analysis(filename="savedir/model_data.json", time_stamps=[1100, 1200])
