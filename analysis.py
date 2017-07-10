import numpy as np
import matplotlib
import scipy.stats as stats
import time
import json
import pickle
import model_saver
from parameters import *

np.set_printoptions(threshold=np.nan)
# Receiver operating characteristic (ROC) curve
# Translated to python from Nick's Matlab code
# returns a value from 0 to 1, with 0.5 representing complete overlap
def calculate_roc(xlist, ylist, fast_calc = False):

    if fast_calc:
        # return the t-statistic instead of the ROC
        sd = np.sqrt(np.var(xlist)/2 + np.var(ylist)/2)
        d = np.mean(xlist) - np.mean(ylist)
        tstat = d/sd
        if np.isnan(tstat):
            tstat = 0
        return tstat

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
"""
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
            'dendrite_inh': (roc_inh, percentile_inh)}

    return roc

"""
def roc_analysis(trial_info, activity_hist):

    num_rules = trial_info['rule_index'].shape[1]
    num_samples = trial_info['sample_index'].shape[1]
    time_pts = np.array(par['time_pts'])//par['dt']
    roc = {}

    n_trials = par['num_batches']*par['batch_train_size']
    sample_cat = np.zeros((n_trials, par['num_receptive_fields'], par['num_categorizations']))

    """"
    Determine the category membership of stimuli in all receptive fields
    """
    t1=time.time()
    for rf in range(par['num_receptive_fields']):
        ind_rule0 = np.where(trial_info['sample_index'][:,rf]>=par['num_motion_dirs']//2)[0]
        ind_rule1 = np.where((trial_info['sample_index'][:,rf]>=par['num_motion_dirs']//4)* \
            (trial_info['sample_index'][:,rf]<3*par['num_motion_dirs']//4))[0]
        sample_cat[ind_rule0, rf, 0] = 1
        sample_cat[ind_rule1, rf, 1] = 1

    for var in par['anova_vars']:
        if activity_hist[var] == []:
            # skip this variable if no data is available
            continue
        if var.count('dend') > 0:
            dims = [par['n_hidden']*par['den_per_unit'], len(par['time_pts']), par['num_receptive_fields'], \
                par['num_categorizations'], par['num_categorizations']]
            dendrite = True
        else:
            dims = [par['n_hidden'], len(par['time_pts']), par['num_receptive_fields'], \
                par['num_categorizations'], par['num_categorizations']]
            dendrite = False
        # create variables if they're not currently in the anova dictionary
        if var not in roc.keys():
            # TODO: Change this if using t-stat
            roc[var] = 0.5*np.ones((dims), dtype = np.float32)

        for rf in range(par['num_receptive_fields']):
            for rule in range(par['num_categorizations']):
                for cat in range(par['num_categorizations']):
                    ind0 = np.where((trial_info['rule_index']==rule)*(sample_cat[:,rf,cat]==0))
                    ind1 = np.where((trial_info['rule_index']==rule)*(sample_cat[:,rf,cat]==1))
                    for n in range(dims[0]):
                        for t in range(dims[1]):
                            if dendrite:
                                roc[var][n,t,rf,rule,cat] = calculate_roc(activity_hist[var][time_pts[t],n%par['n_hidden'], \
                                    n//par['n_hidden'],ind0], activity_hist[var][time_pts[t],n%par['n_hidden'], n//par['n_hidden'],ind1], fast_calc = True)
                            else:
                                roc[var][n,t,rf,rule,cat] = calculate_roc(activity_hist[var][time_pts[t],n,ind0], \
                                    activity_hist[var][time_pts[t],n,ind1], fast_calc = True)


    # reshape dendritic variables, assumuming the neccessary dendritic values were present (activity_hist[var] not empty)
    for var in par['roc_vars']:
        if var.count('dend') > 0 and not activity_hist[var] == []:
            roc[var] = np.reshape(roc[var],(par['n_hidden'],par['den_per_unit'], \
                len(par['time_pts']),par['num_receptive_fields'], par['num_categorizations'], par['num_categorizations']), order='F')

    return roc



# Calculate and return ANOVA values based on sample inputs (directions, images, etc.)
def anova_analysis(trial_info, activity_hist):

    # REMOVE hard-coding of 1 at end
    trial_info['sample_index'] = np.reshape(trial_info['sample_index'],(par['num_batches']*par['batch_train_size'], 4), order='F')
    trial_info['rule_index'] = np.reshape(trial_info['rule_index'],(par['num_batches']*par['batch_train_size'], 2), order='F')

    num_rules = trial_info['rule_index'].shape[1]
    num_samples = trial_info['sample_index'].shape[1]
    time_pts = np.array(par['time_pts'])//par['dt']
    anova = {}

    """
    Loop through rules, samples, neurons, etc. and calculate the ANOVAs
    """
    for r in range(num_rules):
        for s in range(num_samples):

            # find the trial indices for current stimulus and rule cue
            trial_index = []
            for val in np.unique(trial_info['sample_index'][:,s]):
                trial_index.append(np.where((trial_info['rule_index'][:,r]==r)*(trial_info['sample_index'][:,s]==val))[0])

            for var in par['anova_vars']:
                if activity_hist[var] == []:
                    # skip this variable if no data is available
                    continue
                if var.count('dend') > 0:
                    dims = [par['n_hidden']*par['den_per_unit'], len(par['time_pts']), num_rules, num_samples]
                    dendrite = True
                else:
                    dims = [par['n_hidden'], len(par['time_pts']), num_rules, num_samples]
                    dendrite = False
                # create variables if they're not currently in the anova dictionary
                if (var + '_pval') not in anova.keys():
                    anova[var + '_pval'] = np.ones((dims), dtype = np.float32)
                    anova[var + '_fval'] = np.zeros((dims), dtype = np.float32)
                for n in range(dims[0]):
                    for t in range(dims[1]):
                        x = []
                        for trial_list in trial_index:
                            # need to index into activity_hist differently depending on whether it
                            # refers to a dendritic or neuronal value
                            if dendrite:
                                x.append(activity_hist[var][time_pts[t],n%par['n_hidden'],n//par['n_hidden'],trial_list])
                            else:
                                x.append(activity_hist[var][time_pts[t],n,trial_list])
                        f, p = stats.f_oneway(*x)
                        if not np.isnan(p):
                            anova[var + '_pval'][n,t,r,s] = p
                            anova[var + '_fval'][n,t,r,s] = f

    # reshape dendritic variables, assumuming the neccessary dendritic values were present (activity_hist[var] not empty)
    for var in par['anova_vars']:
        if var.count('dend') > 0 and not activity_hist[var] == []:
            anova[var + '_pval'] = np.reshape(anova[var + '_pval'],(par['n_hidden'],par['den_per_unit'], \
                len(par['time_pts']), num_rules, num_samples), order='F')
            anova[var + '_fval'] = np.reshape(anova[var + '_pval'],(par['n_hidden'],par['den_per_unit'], \
                len(par['time_pts']), num_rules, num_samples), order='F')

    return anova


# Depending on parameters, returns analysis results for ROC, ANOVA, etc.
def get_analysis(trial_info=[], activity_hist=[], filename=None):

    # Get hidden_state value from parameters if filename is not provided
    if filename is not None:
        print("ROC: Gathering data from JSON file")
        print("ROC: NEEDS FIXING")
        data = model_saver.json_load(savedir=filename)
        time_stamps = np.array(data['params']['time_stamps'])

    # Get analysis results
    result = {'roc': [], 'anova': []}

    if par['roc_vars'] is not None:
        result['roc'] = roc_analysis(trial_info, activity_hist)
    if par['anova_vars'] is not None:
        result['anova'] = anova_analysis(trial_info, activity_hist)


    # Save analysis result
    # with open('.\savedir\dend_analysis_%s.pkl' % time.strftime('%H%M%S-%Y%m%d'), 'wb') as f:
    """
    with open('./savedir/dend_analysis_%s.pkl' % time.strftime('%H%M%S-%Y%m%d'), 'wb') as f:
        pickle.dump(result, f)
    print("ROC: Done pickling\n")
    """

    return result
