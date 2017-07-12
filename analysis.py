import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import json
import pickle
import model_saver
from parameters import *

global iteration
iteration = 0

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



def roc_analysis(trial_info, activity_hist):

    time_pts = np.array(par['time_pts'])//par['dt']
    roc = {}

    n_trials = par['num_test_batches']*par['batch_train_size']
    sample_cat = np.zeros((n_trials, par['num_RFs'], par['num_rules']))

    """"
    Determine the category membership of stimuli in all receptive fields
    """
    for rf in range(par['num_RFs']):
        ind_rule0 = np.where(trial_info['sample_index'][:,rf]>=par['num_unique_samples']//2)[0]
        ind_rule1 = np.where((trial_info['sample_index'][:,rf]>=par['num_unique_samples']//4)* \
            (trial_info['sample_index'][:,rf]<3*par['num_samples']//4))[0]
        sample_cat[ind_rule0, rf, 0] = 1
        sample_cat[ind_rule1, rf, 1] = 1

    for var in par['anova_vars']:
        if activity_hist[var] == []:
            # skip this variable if no data is available
            continue
        if var.count('dend') > 0:
            dims = [par['n_hidden']*par['den_per_unit'], len(par['time_pts']), par['num_RFs'], \
                par['num_rules'], par['num_rules']]
            dendrite = True
        else:
            dims = [par['n_hidden'], len(par['time_pts']), par['num_RFs'], \
                par['num_rules'], par['num_rules']]
            dendrite = False
        # create variables if they're not currently in the anova dictionary
        if var not in roc.keys():
            # TODO: Change this if using t-stat or ROC
            roc[var] = np.zeros((dims), dtype = np.float32)

        for rf in range(par['num_RFs']):
            for rule in range(par['num_rules']):
                for cat in range(par['num_rules']):
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
                len(par['time_pts']),par['num_RFs'], par['num_rules'], par['num_rules']), order='F')

    return roc



# Calculate and return ANOVA values based on sample inputs (directions, images, etc.)
def anova_analysis(trial_info, activity_hist):

    time_pts = np.array(par['time_pts'])//par['dt']
    anova = {}

    """
    Loop through rules, samples, neurons, etc. and calculate the ANOVAs
    """
    for rf in range(par['num_RFs']):
        bool_attend = trial_info['location_index'] == rf
        for r in range(par['num_rules']):
            bool_rule = trial_info['rule_index'] == r
            # find the trial indices for current stimulus and rule cue
            trial_index_attend = []
            trial_index_not_attend = []
            for s in range(par['num_unique_samples']):
                trial_index_attend.append(np.where(bool_rule*bool_attend*(trial_info['sample_index'][:,rf]==s))[0])
                trial_index_not_attend.append(np.where(bool_rule*np.logical_not(bool_attend)*(trial_info['sample_index'][:,rf]==s))[0])

            for var in par['anova_vars']:
                if activity_hist[var] == []:
                    # skip this variable if no data is available
                    continue
                if var.count('dend') > 0:
                    dims = [par['n_hidden']*par['den_per_unit'], par['num_RFs'],par['num_rules'], len(par['time_pts'])]
                    dendrite = True
                else:
                    dims = [par['n_hidden'], par['num_RFs'], par['num_rules'], len(par['time_pts'])]
                    dendrite = False
                # create variables if they're not currently in the anova dictionary
                if (var + '_pval') not in anova.keys():
                    anova[var + '_attn_pval'] = np.ones((dims), dtype = np.float32)
                    anova[var + '_attn_fval'] = np.zeros((dims), dtype = np.float32)
                    anova[var + '_no_attn_pval'] = np.ones((dims), dtype = np.float32)
                    anova[var + '_no_attn_fval'] = np.zeros((dims), dtype = np.float32)
                for n in range(dims[0]):
                    for t in range(len(par['time_pts'])):
                        attend_vals = []
                        not_attend_vals = []
                        for trial_list in trial_index_attend:
                            if dendrite:
                                attend_vals.append(activity_hist[var][time_pts[t],n%par['n_hidden'],n//par['n_hidden'],trial_list])
                            else:
                                attend_vals.append(activity_hist[var][time_pts[t],n,trial_list])
                        for trial_list in trial_index_not_attend:
                            if dendrite:
                                not_attend_vals.append(activity_hist[var][time_pts[t],n%par['n_hidden'],n//par['n_hidden'],trial_list])
                            else:
                                not_attend_vals.append(activity_hist[var][time_pts[t],n,trial_list])

                        f_attend, p_attend = stats.f_oneway(*attend_vals)
                        f_not_attend, p_not_attend = stats.f_oneway(*attend_vals)

                        if not np.isnan(p_attend):
                            anova[var + '_attn_pval'][n,rf,r,t] = p_attend
                            anova[var + '_attn_fval'][n,rf,r,t] = f_attend
                            anova[var + '_no_attn_pval'][n,rf,r,t] = p_not_attend
                            anova[var + '_no_attn_fval'][n,rf,r,t] = f_not_attend

    # reshape dendritic variables, assumuming the neccessary dendritic values were present (activity_hist[var] not empty)
    for var in par['anova_vars']:
        if var.count('dend') > 0 and not activity_hist[var] == []:
                for k,v in anova.items():
                    anova[k] = np.reshape(v, (par['n_hidden'],par['den_per_unit'], \
                        len(par['time_pts']),  par['num_RFs'], par['num_rules']), order='F')

    return anova



# calclulate mean activity
def tuning_analysis(trial_info, activity_hist):

    # Variables
    time_pts = np.array(par['time_pts'])//par['dt']
    tuning = {}

    # Calculate mean firing rate for attended RF and unattended RF
    for var in par['tuning_vars']:
        if activity_hist[var] == []:
            # skip this variable if no data is available
            continue
        if var.count('dend') > 0:
            dims = [par['n_hidden']*par['den_per_unit'], par['num_RFs'],par['num_rules'], len(par['time_pts'])]
            dendrite = True
        else:
            dims = [par['n_hidden'], par['num_RFs'], par['num_rules'], len(par['time_pts'])]
            dendrite = False
        # create variables if they're not currently in the anova dictionary
        if (var + '_pval') not in tuning.keys():
            tuning[var + '_attn'] = np.zeros((dims), dtype = np.float32)
            tuning[var + '_no_attn'] = np.zeros((dims), dtype = np.float32)

        for rf in range(par['num_RFs']):
            bool_attend = trial_info['location_index'] == rf
            for r in range(par['num_rules']):
                bool_rule = trial_info['rule_index'] == r
                for s in range(par['num_unique_samples']):
                    trial_ind_attend = np.where(bool_rule*bool_attend*(trial_info['sample_index'][:,rf]==s))[0]
                    trial_ind_not_attend = np.where(bool_rule*np.logical_not(bool_attend)*(trial_info['sample_index'][:,rf]==s))[0]

                    for n in range(dims[0]):
                        for t in range(len(par['time_pts'])):

                            if dendrite:
                                tuning[var + '_attn'][n,rf,r,t] = np.mean(activity_hist[var][time_pts[t], \
                                    n%par['n_hidden'],n//par['n_hidden'],trial_ind_attend])
                                tuning[var + '_no_attn'][n,rf,r,t] = np.mean(activity_hist[var][time_pts[t],n%par['n_hidden'], \
                                    n//par['n_hidden'],trial_ind_not_attend])
                            else:
                                tuning[var + '_attn'][n,rf,r,t] = np.mean(activity_hist[var][time_pts[t],n,trial_ind_attend])
                                tuning[var + '_no_attn'][n,rf,r,t] = np.mean(activity_hist[var][time_pts[t],n,trial_ind_not_attend])

    # reshape dendritic variables, assumuming the neccessary dendritic values were present (activity_hist[var] not empty)
    for var in par['tuning_vars']:
        if var.count('dend') > 0 and not activity_hist[var] == []:
                for k,v in tuning.items():
                    tuning[k] = np.reshape(v, (par['n_hidden'],par['den_per_unit'], \
                        len(par['time_pts']),  par['num_RFs'], par['num_rules']), order='F')

    return tuning



# Just plotting a lot of things...
def plot(trial_info, activity_hist):
    global iteration
    print("Plotting with iteration: ", iteration)

    time_pts = np.array(par['time_pts'])//par['dt']
    num_trials = par['num_test_batches']*par['batch_train_size']

    for var in par['anova_vars']:
        for t in time_pts:
            for i in range(par['n_hidden']):
                if trial_info['rule_index'][1,0] == 0:
                    x = np.array([0,1,2,3,4,5,6,7])
                elif trial_info['rule_index'][1,0] == 1:
                    x = np.array([2,3,4,5,0,1,6,7])

                attended_dir = trial_info['sample_index'][range(num_trials), trial_info['rule_index'][0]]

                mean, std, roc = [], [], []
                for val in x:
                    data = activity_hist['state_hist'][t, i, np.where(attended_dir==val)]
                    roc = np.append(roc, data)
                    mean = np.append(mean, np.mean(data))
                    std = np.append(std, np.std(data))

                plt.errorbar(x, mean, std)
                t_stat = calculate_roc(roc[0:(int)(num_trials/2)], roc[(int)(num_trials/2):num_trials],fast_calc=True)
                plt.title('neuron'+str(i)+"_t_stat_"+str(t_stat))
                plt.savefig('./analysis/neuron_'+str(i)+"_time_"+str(t*par['dt'])+"_iter_"+str(iteration)+'.jpg')
                plt.clf()

    iteration = iteration + 1



# Depending on parameters, returns analysis results for ROC, ANOVA, etc.
def get_analysis(trial_info=[], activity_hist=[], filename=None):

    # Get hidden_state value from parameters if filename is not provided
    if filename is not None:
        print("ROC: Gathering data from JSON file")
        print("ROC: NEEDS FIXING")
        data = model_saver.json_load(savedir=filename)
        time_stamps = np.array(data['params']['time_stamps'])

    # Get analysis results
    result = {'roc': [], 'anova': [], 'tuning': []}

    if par['roc_vars'] is not None:
        result['roc'] = roc_analysis(trial_info, activity_hist)
    if par['anova_vars'] is not None:
        result['anova'] = anova_analysis(trial_info, activity_hist)
    if par['tuning_vars'] is not None:
        result['tuning'] = tuning_analysis(trial_info, activity_hist)

    #plot(trial_info, activity_hist)

    # Save analysis result
    # with open('.\savedir\dend_analysis_%s.pkl' % time.strftime('%H%M%S-%Y%m%d'), 'wb') as f:
    """
    with open('./savedir/dend_analysis_%s.pkl' % time.strftime('%H%M%S-%Y%m%d'), 'wb') as f:
        pickle.dump(result, f)
    print("ROC: Done pickling\n")
    """

    return result
