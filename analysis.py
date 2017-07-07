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

np.set_printoptions(threshold=np.nan)
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
def roc_analysis(time_stamps=[], cat_rule=[], dir_cat=[], target_dir=[], neuron=[], dendrites=[], exc=[], inh=[]):
    
    # Pre-allocating ROC

    roc_neuron, percentile_neuron, roc_dend, percentile_dend, roc_exc, percentile_exc, roc_inh, percentile_inh = [],[],[],[],[],[],[],[]
    if par['neuron_analysis']:
        num_hidden, num_time, batch_size = neuron.shape
        roc_neuron = np.ones([num_hidden, len(time_stamps), batch_size, par['num_category_rule'], par['num_category_rule']]) * 0.5
    if par['dendrites_analysis']:
        num_hidden, num_dend, num_time, batch_size = dendrites.shape
        roc_dend = np.ones([num_hidden, num_dend, len(time_stamps), batch_size, par['num_category_rule'], par['num_category_rule']]) * 0.5
    if par['exc_analysis']:
        num_hidden, num_dend, num_time, batch_size = exc.shape
        roc_exc = np.ones([num_hidden, num_dend, len(time_stamps), batch_size, par['num_category_rule'], par['num_category_rule']]) * 0.5
    if par['inh_analysis']:
        num_hidden, num_dend, num_time, batch_size = inh.shape
        roc_inh = np.ones([num_hidden, num_dend, len(time_stamps), batch_size, par['num_category_rule'], par['num_category_rule']]) * 0.5


    # Need update based on the category rules we are using
    # Rule 0: up == True, down == False
    # Rule 1: left == True, right == False
    direction_cat = np.zeros([par['num_category_rule'], batch_size])
    for ind in range(batch_size):
        direction_cat[0, ind] = dir_cat[ind] in [0,1,6,7]
        direction_cat[1, ind] = dir_cat[ind] in [4,5,6,7]


    # Get roc value
    # roc = (num_hidden x num_dend x num_time_stamps x num_category_rule x num_category_rule)
    t1=time.time()
    for c1 in range(par['num_category_rule']):
        for c2 in range(par['num_category_rule']):
            ind0 = np.where((direction_cat[c2]==0) * (cat_rule == c1))[0]
            if len(ind0!=0):
                ind1 = np.where((direction_cat[c2]==1) * (cat_rule == c1))[0]
                for n in range(num_hidden):
                    for t in range(len(time_stamps)):
                        if par['neuron_analysis']:
                            roc_neuron[n, t, c1, c2]  = calculate_roc(neuron[n, time_stamps[t], ind0], neuron[n, time_stamps[t], ind1])
                        if any([par['dendrites_analysis'], par['exc_analysis'], par['inh_analysis']]):
                            for d in range(num_dend):
                                if par['dendrites_analysis']:
                                    roc_dend[n, d, t, c1, c2]  = calculate_roc(dendrites[n, d, time_stamps[t], ind0], dendrites[n, d, time_stamps[t], ind1])
                                if par['exc_analysis']:
                                    roc_exc[n, d, t, c1, c2]  = calculate_roc(exc[n, d, time_stamps[t], ind0], exc[n, d, time_stamps[t], ind1])
                                if par['inh_analysis']:
                                    roc_inh[n, d, t, c1, c2]  = calculate_roc(inh[n, d, time_stamps[t], ind0], inh[n, d, time_stamps[t], ind1])


    # roc's = (num_hidden x num_dend x num_time_stamps x num_category_rule x num_category_rule)
    # roc_rectified = (num_category_rule x num_category_rule x num_time_stamps x num_hidden x num_dend)
    # percentile contains 90th percentile value of the roc_rectified
    print('ROC DENDRITES ', time.time()-t1)
    if par['neuron_analysis']:
        print(roc_neuron.shape)
        roc_neuron_rect = np.reshape(np.absolute(0.5-roc_neuron), (num_hidden*len(time_stamps)*batch_size, par['num_category_rule'], par['num_category_rule']))
        percentile_neuron = np.zeros([par['num_category_rule'], par['num_category_rule']])
        percentile_neuron = np.percentile(roc_neuron_rect[:,:,:], 90, axis=0)
    if par['dendrites_analysis']:
        roc_dend_rect = np.reshape(np.absolute(0.5-roc_dend), (num_hidden*num_dend*len(time_stamps)*batch_size,par['num_category_rule'],par['num_category_rule']))
        percentile_dend = np.zeros([par['num_category_rule'], par['num_category_rule']])
        percentile_dend = np.percentile(roc_dend_rect[:,:,:], 90, axis=0)
    if par['exc_analysis']:
        roc_dend_exc = np.reshape(np.absolute(0.5-roc_exc), (num_hidden*num_dend*len(time_stamps)*batch_size,par['num_category_rule'],par['num_category_rule']))
        percentile_exc = np.zeros([par['num_category_rule'], par['num_category_rule']])
        percentile_exc = np.percentile(roc_dend_exc[:,:,:], 90, axis=0)
    if par['inh_analysis']:
        roc_dend_inh = np.reshape(np.absolute(0.5-roc_inh), (num_hidden*num_dend*len(time_stamps)*batch_size,par['num_category_rule'],par['num_category_rule']))
        percentile_inh = np.zeros([par['num_category_rule'], par['num_category_rule']])
        percentile_inh = np.percentile(roc_dend_inh[:,:,:], 90, axis=0)

    roc = {'neurons': (roc_neuron, percentile_neuron),
            'dendrites': (roc_dend, percentile_dend), 
            'dendrite_exc': (roc_exc, percentile_exc), 
            'dendrite_inh': (roc_inh, percentile_inh)
            }

    return roc


# Calculate and return ANOVA values based on sample inputs (directions, images, etc.)
def anova_analysis(time_stamps=[], info=[], neuron=[], dendrites=[], exc=[], inh=[]):

    # Pre-allocating ANOVA
    anova_neuron, anova_dend, anova_exc, anova_inh, p_neuron, p_dend, p_exc, p_inh = [], [], [], [], [], [], [], []
    if par['neuron_analysis']:
        num_hidden, num_time, batch_size = neuron.shape
        anova_neuron = np.ones([num_hidden, len(time_stamps)], np.dtype((np.int32, (2,))))
    if par['dendrites_analysis']:
        num_hidden, num_dend, num_time, batch_size = dendrites.shape
        anova_dend = np.ones([num_hidden, num_dend, len(time_stamps)], np.dtype((np.int32, (2,))))
    if par['exc_analysis']:
        num_hidden, num_dend, num_time, batch_size = exc.shape
        anova_exc = np.ones([num_hidden, num_dend, len(time_stamps)], np.dtype((np.int32, (2,))))
    if par['inh_analysis']:
        num_hidden, num_dend, num_time, batch_size = inh.shape
        anova_inh = np.ones([num_hidden, num_dend, len(time_stamps)], np.dtype((np.int32, (2,))))


    # ind contains indexes for each directions
    ind = []
    for val in np.unique(info):
        ind.append(np.where(info==val)[0])
    ind = np.array(ind)
    print("index")
    print(ind)

    # avona = (num_hidden x (num_dend) x num_time_stamps)
    for n in range(num_hidden):
        for t in range(len(time_stamps)):
            if par['neuron_analysis']:
                if np.var(neuron[n, t, :]) > 1e-12:
                    temp = []
                    for i in ind:
                        temp.append(neuron[n,t,i])
                    anova_neuron[n, t] = stats.f_oneway(*temp)
            if any([par['dendrites_analysis'], par['exc_analysis'], par['inh_analysis']]):
                for d in range(num_dend):
                    if par['dendrites_analysis']:
                        if np.var(dendrites[n,d,t,:]) > 1e-12:
                            print(np.var(dendrites[n,d,t,:]))
                            temp = []
                            for i in ind:
                                temp.append(dendrites[n,d,t,i])
                                print("dendrites", n, d, t)
                                print(dendrites[n,d,t,i])
                            print("temp")
                            print(temp)
                            print("anova")
                            print(stats.f_oneway(*temp))
                            anova_dend[n, d, t] = stats.f_oneway(*temp)
                    if par['exc_analysis']:
                        if np.var(exc[n,d,t,:]) > 1e-12:
                            print(np.var(exc[n,d,t,:]))
                            temp = []
                            for i in ind:
                                temp.append(exc[n,d,t,i])
                                print("exc", n, d, t)
                                print(exc[n,d,t,i])
                            print("temp")
                            print(temp)
                            print("anova")
                            print(stats.f_oneway(*temp))
                            anova_exc[n, d, t] = stats.f_oneway(*temp)
                    if par['inh_analysis']:
                        if np.var(inh[n,d,t,:]) > 1e-12:
                            print(np.var(inh[n,d,t,:]))
                            temp = []
                            for i in ind:
                                temp.append(inh[n,d,t,i])
                                print("inh", n, d, t)
                                print(inh[n,d,t,i])
                            print("temp")
                            print(temp)
                            print("anova")
                            print(stats.f_oneway(*temp))
                            anova_inh[n, d, t] = stats.f_oneway(*temp)


    # Get mean of p-values that are < 0.01
    p_neuron = np.mean(anova_neuron < 0.01)
    p_dend = np.mean(anova_dend < 0.01)
    p_exc = np.mean(anova_exc < 0.01)
    p_inh = np.mean(anova_inh < 0.01)


    anova = {'neurons': (anova_neuron, p_neuron),
            'dendrites': (anova_dend, p_dend),
            'dendrite_exc': (anova_exc, p_exc),
            'dendrite_inh': (anova_inh, p_inh)
            }

    return anova


# Depending on parameters, returns analysis results for ROC, ANOVA, etc.
def get_analysis(filename=None, neuron=[], dendrites=[], exc=[], inh=[], data=[], info=[]):

    # Get hidden_state value from parameters if filename is not provided
    if filename is None:
        print("ROC: Gathering data from model_results")
        data = data
        trial_info = info
        time_stamps = np.array(par['time_stamps'])
    else:
        print("ROC: Gathering data from JSON file")
        data = model_saver.json_load(savedir=filename)        
        time_stamps = np.array(data['params']['time_stamps'])

    category_rule = np.array(trial_info['rule_index'])
    attended_dir = np.array(trial_info['sample_index'])
    neuron = np.array(neuron)
    dendrites = np.array(dendrites)
    exc = np.array(exc)
    inh = np.array(inh)
    time_stamps = time_stamps//par['dt']

    # Just a note:
    # Attended_dir = (1000 x 4) or (1000)
    # Category_rule = (2 x 1000) or (1000)
    #   0 = location rule
    #   1 = category rule
    print(attended_dir.shape)
    print(category_rule.shape)

    print("exc:", exc.shape)
    print("inh:", inh.shape)


    # Reshaping the dendrite or hidden_state matrix + pre-allocate roc matrix
    # dendrite = (time x n_hidden x den_per_unit x batch_train_size)
    #          => (n_hidden x dend_per_unit x time x trials)
    # neuron = (time x n_hidden x batch_train_size)
    #        => (n_hidden x time x trials)
    print(neuron.shape)
    if par['neuron_analysis']:
        num_time, num_hidden, batch_size = neuron.shape
        neuron = np.stack(neuron, axis=1)
    if par['dendrites_analysis']:
        dendrites = np.stack(dendrites, axis=2)
        num_hidden, num_dend, num_time, num_batches = dendrites.shape
    if par['exc_analysis']:
        exc = np.stack(exc, axis=2)
        num_hidden, num_dend, num_time, num_batches = dendrites.shape
    if par['inh_analysis']:
        inh = np.stack(inh, axis=2)
        num_hidden, num_dend, num_time, num_batches = dendrites.shape


    # Get analysis results
    result = {'roc': [], 'ANOVA': []}
    if par['roc']:
        result['roc'] = roc_analysis(time_stamps=time_stamps, cat_rule=category_rule, dir_cat=attended_dir, target_dir=attended_dir, \
                                   neuron=neuron, dendrites=dendrites, exc=exc, inh=inh)
    if par['anova']:
        result['ANOVA'] = anova_analysis(time_stamps=time_stamps, info=attended_dir, neuron=neuron, dendrites=dendrites, exc=exc, inh=inh)

   
    # Save analysis result
    # with open('.\savedir\dend_analysis_%s.pkl' % time.strftime('%H%M%S-%Y%m%d'), 'wb') as f:
    with open('./savedir/dend_analysis_%s.pkl' % time.strftime('%H%M%S-%Y%m%d'), 'wb') as f:
        pickle.dump(result, f)
    print("ROC: Done pickling\n")

    return result


# GET ROC CALCULATION DONE FOR: hidden_state_hist, dendrites_hist, dendrites_inputs_exc_hist, dendrites_inputs_inh_hist
# roc_analysis(filename="savedir/model_data.json", time_stamps=[1100, 1200])
