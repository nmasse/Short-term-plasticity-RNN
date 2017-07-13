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
        if sd == 0:
            return 0
        d = np.mean(xlist) - np.mean(ylist)
        tstat = d/sd

        return tstat

    roc = 0
    unique_vals = np.unique(xlist)

    # Calculate ROC value
    for val in unique_vals:
        p1 = np.mean(xlist==val)
        p2 = np.mean(ylist>val) + 0.5*np.mean(ylist==val)
        roc = roc + p1*p2

    return roc



def roc_analysis(test_data):

    time_pts = np.array(par['time_pts'])//par['dt']
    roc = {}

    n_trials = par['num_test_batches']*par['batch_train_size']
    sample_cat = np.zeros((n_trials, par['num_RFs'], par['num_rules']))

    """"
    Determine the category membership of stimuli in all receptive fields
    """
    for rf in range(par['num_RFs']):
        ind_rule0 = np.where(test_data['sample_index'][:,rf]>=par['num_unique_samples']//2)[0]
        ind_rule1 = np.where((test_data['sample_index'][:,rf]>=par['num_unique_samples']//4)* \
            (test_data['sample_index'][:,rf]<3*par['num_samples']//4))[0]
        sample_cat[ind_rule0, rf, 0] = 1
        sample_cat[ind_rule1, rf, 1] = 1

    for var in par['roc_vars']:
        if not var in test_data.keys():
            # skip this variable if no data is available
            continue
        if var.count('dend') > 0:
            dims = [par['n_hidden'], par['den_per_unit'], par['num_RFs'],par['num_rules'],par['num_rules'], len(par['time_pts'])]
        else:
            dims = [par['n_hidden'], par['num_RFs'], par['num_rules'],par['num_rules'], len(par['time_pts'])]
        # create variables if they're not currently in the anova dictionary
        roc[var + '_attn'] = np.zeros((dims), dtype = np.float32)
        roc[var + '_no_attn'] = np.zeros((dims), dtype = np.float32)

        for rf in range(par['num_RFs']):
            bool_attend = test_data['location_index'][:,0] == rf
            for rule in range(par['num_rules']):
                bool_attn_rule = bool_attend*(test_data['rule_index'][:,0] == rule)
                bool_no_attn_rule = np.logical_not(bool_attend)*(test_data['rule_index'][:,0] == rule)
                for cat in range(par['num_rules']):
                    ind0_attn = np.where(bool_attn_rule*(sample_cat[:,rf,cat]==0))
                    ind1_attn = np.where(bool_attn_rule*(sample_cat[:,rf,cat]==1))
                    ind0_no_attn = np.where(bool_no_attn_rule*(sample_cat[:,rf,cat]==0))
                    ind1_no_attn = np.where(bool_no_attn_rule*(sample_cat[:,rf,cat]==1))
                    for t in range(len(par['time_pts'])):
                        for n in range(par['n_hidden']):
                            if var.count('dend') > 0:
                                for d in range(par['den_per_unit']):
                                    roc[var + '_attn'][n,d,rf,rule,cat,t] = calculate_roc(test_data[var][time_pts[t], \
                                        n%par['n_hidden'], n//par['n_hidden'],ind0_attn], test_data[var][time_pts[t], \
                                        n%par['n_hidden'], n//par['n_hidden'],ind1_attn], fast_calc = True)
                                    roc[var + '_no_attn'][n,d,rf,rule,cat,t] = calculate_roc(test_data[var][time_pts[t], \
                                        n%par['n_hidden'], n//par['n_hidden'],ind0_no_attn], test_data[var][time_pts[t], \
                                        n%par['n_hidden'], n//par['n_hidden'],ind1_no_attn], fast_calc = True)
                            else:
                                roc[var + '_attn'][n,rf,rule,cat,t] = calculate_roc(test_data[var][time_pts[t], \
                                    n,ind0_attn], test_data[var][time_pts[t], n,ind1_attn], fast_calc = True)
                                roc[var + '_no_attn'][n,rf,rule,cat,t] = calculate_roc(test_data[var][time_pts[t], \
                                    n,ind0_no_attn], test_data[var][time_pts[t], n,ind1_no_attn], fast_calc = True)

    return roc



# Calculate and return ANOVA values based on sample inputs (directions, images, etc.)
def anova_analysis(test_data):
    time_pts = np.array(par['time_pts'])//par['dt']
    anova = {}

    """
    Loop through rules, samples, neurons, etc. and calculate the ANOVAs
    """
    for var in par['anova_vars']:
        if not var in test_data.keys():
            # skip this variable if no data is available
            continue
        if var.count('dend') > 0:
            dims = [par['n_hidden'], par['den_per_unit'], par['num_RFs'],par['num_rules'], len(par['time_pts'])]
        else:
            dims = [par['n_hidden'], par['num_RFs'], par['num_rules'], len(par['time_pts'])]
        # create variables if they're not currently in the anova dictionary
        anova[var + '_attn_pval'] = np.ones((dims), dtype = np.float32)
        anova[var + '_attn_fval'] = np.zeros((dims), dtype = np.float32)
        anova[var + '_no_attn_pval'] = np.ones((dims), dtype = np.float32)
        anova[var + '_no_attn_fval'] = np.zeros((dims), dtype = np.float32)


    for rf in range(par['num_RFs']):
        bool_attend = test_data['location_index'][:,0] == rf
        for r in range(par['num_rules']):
            bool_attn_rule = bool_attend*(test_data['rule_index'][:,0] == r)
            bool_no_attn_rule = np.logical_not(bool_attend)*(test_data['rule_index'][:,0] == r)

            # find the trial indices for current stimulus and rule cue
            trial_index_attend = []
            trial_index_not_attend = []
            for s in range(par['num_unique_samples']):
                bool_sample = test_data['sample_index'][:,rf] == s
                trial_index_attend.append(np.where(bool_attn_rule*bool_sample)[0])
                m = len(trial_index_attend)
                trial_index_not_attend.append(np.where(bool_no_attn_rule*bool_sample)[0][:m])

            for var in par['anova_vars']:
                for t in range(len(par['time_pts'])):
                    for n in range(par['n_hidden']):
                        if var.count('dend') > 0:
                            for d in range(par['den_per_unit']):
                                if np.var(test_data[var][time_pts[t],n,d,:]) < 1e-12:
                                    continue
                                attend_vals = []
                                not_attend_vals = []

                                for trial_list in trial_index_attend:
                                    attend_vals.append(test_data[var][time_pts[t],n,d,trial_list])
                                for trial_list in trial_index_not_attend:
                                    not_attend_vals.append(test_data[var][time_pts[t],n,d,trial_list])

                                f_attend, p_attend = stats.f_oneway(*attend_vals)
                                f_not_attend, p_not_attend = stats.f_oneway(*not_attend_vals)
                                if not np.isnan(p_attend):
                                    anova[var + '_attn_pval'][n,d,rf,r,t] = p_attend
                                    anova[var + '_attn_fval'][n,d,rf,r,t] = f_attend
                                    anova[var + '_no_attn_pval'][n,d,rf,r,t] = p_not_attend
                                    anova[var + '_no_attn_fval'][n,d,rf,r,t] = f_not_attend
                        else:
                            if np.var(test_data[var][time_pts[t],n,:]) < 1e-12:
                                continue
                            attend_vals = []
                            not_attend_vals = []

                            for trial_list in trial_index_attend:
                                attend_vals.append(test_data[var][time_pts[t],n,trial_list])
                            for trial_list in trial_index_not_attend:
                                not_attend_vals.append(test_data[var][time_pts[t],n,trial_list])

                            f_attend, p_attend = stats.f_oneway(*attend_vals)
                            f_not_attend, p_not_attend = stats.f_oneway(*not_attend_vals)
                            if not np.isnan(p_attend):
                                anova[var + '_attn_pval'][n,rf,r,t] = p_attend
                                anova[var + '_attn_fval'][n,rf,r,t] = f_attend
                                anova[var + '_no_attn_pval'][n,rf,r,t] = p_not_attend
                                anova[var + '_no_attn_fval'][n,rf,r,t] = f_not_attend

    return anova



# calclulate mean activity
def tuning_analysis(test_data):

    # Variables
    time_pts = np.array(par['time_pts'])//par['dt']
    tuning = {}

    # Initialize variables
    for var in par['tuning_vars']:
        if not var in test_data.keys():
            # skip this variable if no data is available
            continue
        if var.count('dend') > 0:
            dims = [par['n_hidden'],par['den_per_unit'], par['num_RFs'],par['num_rules'], len(par['time_pts'])]
        else:
            dims = [par['n_hidden'], par['num_RFs'], par['num_rules'], len(par['time_pts'])]
        tuning[var + '_attn'] = np.zeros((dims), dtype = np.float32)
        tuning[var + '_no_attn'] = np.zeros((dims), dtype = np.float32)

    for rf in range(par['num_RFs']):
        bool_attend = test_data['location_index'][:,0] == rf
        for r in range(par['num_rules']):
            bool_attn_rule = bool_attend*(test_data['rule_index'][:,0] == r)
            bool_no_attn_rule = np.logical_not(bool_attend)*(test_data['rule_index'][:,0] == r)
            for s in range(par['num_unique_samples']):
                bool_sample = test_data['sample_index'][:,rf]==s
                trial_ind_attend = np.where(bool_sample*bool_attn_rule)[0]
                trial_ind_not_attend = np.where(bool_sample*bool_no_attn_rule)[0]

                for t in range(len(par['time_pts'])):
                    for var in par['tuning_vars']:
                        if var.count('dend') > 0:
                            tuning[var + '_attn'][:,:,rf,r,t] = np.mean(test_data[var][time_pts[t], \
                                :, :, trial_ind_attend], axis=0, keepdims=True)
                            tuning[var + '_no_attn'][:,:,rf,r,t] = np.mean(test_data[var][time_pts[t], \
                                :, :,trial_ind_not_attend], axis=0, keepdims=True)
                        else:
                            tuning[var + '_attn'][:,rf,r,t] = np.mean(test_data[var][time_pts[t], \
                                :, trial_ind_attend], axis=0, keepdims=True)
                            tuning[var + '_no_attn'][:,rf,r,t] = np.mean(test_data[var][time_pts[t], \
                                :, trial_ind_not_attend], axis=0, keepdims=True)

    return tuning


def get_perf(test_data):
    """
    Calculate task accuracy by comparing the actual network output to the
    desired output, but only examines time points when test stimulus is on
    (in another words, when y[0,:,:] is not 0)
    """

    rule_accuracy   = np.zeros([par['num_rules']])
    rule_counters   = np.zeros([par['num_rules']])
    accuracy        = np.zeros([par['num_test_batches'], par['batch_train_size']])

    # Put axes in order [batch x trial x time steps x outputs]
    y_raw           = np.transpose(test_data['y'], [0,3,1,2])
    y_hat_raw       = np.transpose(test_data['y_hat'], [0,3,2,1])
    mask_raw        = np.transpose(test_data['train_mask'], [0,2,1])

    for b in range(par['num_test_batches']):
        for n in range(par['batch_train_size']):
            # First, isolate a single trial
            # The axes are now [time steps x outputs]
            y       = y_raw[b,n]
            y_hat   = y_hat_raw[b,n]
            m       = mask_raw[b,n]

            # In corporate the fixation period into the mask
            # When the 0th output neuron is 0, there is no fixation, and so
            # is not incorporated into the mask as a 0
            m *= y_hat[:,0]==0

            # Now both y and y_hat are of shape [time steps]
            y     = np.argmax(y, axis=1)
            y_hat = np.argmax(y_hat, axis=1)

            # y and y_hat are now compared before being multiplied by the mask,
            # so as to only look at desired parts of the output.
            accuracy[b,n] = np.sum(np.multiply(np.float32(y == y_hat), m))/np.sum(m)

            # Now that we know the accuracy for this trial, determine the rule
            # used for this trial.  The accuracy is then added to the proper
            # rule accuracy.
            rule = test_data['rule_index'][b*par['batch_train_size']+n,0]
            rule_accuracy[rule] += accuracy[b,n]
            rule_counters[rule] += 1

    # Normalize the rule accuracies by the number of occurences of that rule
    for r in range(par['num_rules']):
        rule_accuracy[r] = rule_accuracy[r]/rule_counters[r]

    return np.mean(accuracy), rule_accuracy


# Just plotting a lot of things...
def plot(test_data):
    global iteration
    print("Plotting with iteration: ", iteration)

    time_pts = np.array(par['time_pts'])//par['dt']
    num_trials = par['num_test_batches']*par['batch_train_size']

    for var in par['anova_vars']:
        for t in time_pts:
            for i in range(par['n_hidden']):
                if test_data['rule_index'][1,0] == 0:
                    x = np.array([0,1,2,3,4,5,6,7])
                elif test_data['rule_index'][1,0] == 1:
                    x = np.array([2,3,4,5,0,1,6,7])

                attended_dir = test_data['sample_index'][range(num_trials), test_data['rule_index'][0]]

                mean, std, roc = [], [], []
                for val in x:
                    data = test_data['state_hist'][t, i, np.where(attended_dir==val)]
                    roc = np.append(roc, data)
                    mean = np.append(mean, np.mean(data))
                    std = np.append(std, np.std(data))

                plt.errorbar(x, mean, std)
                t_stat = calculate_roc(roc[0:(int)(num_trials/2)], roc[(int)(num_trials/2):num_trials],fast_calc=True)
                plt.title('neuron'+str(i)+"_t_stat_"+str(t_stat))
                plt.savefig('./analysis/neuron_'+str(i)+"_time_"+str(t*par['dt'])+"_iter_"+str(iteration)+'.jpg')
                plt.clf()

    iteration = iteration + 1


def get_analysis(test_data={}, filename=None):
    """
    Depending on parameters, returns analysis results for ROC, ANOVA, etc.
    """

    # Get hidden_state value from parameters if filename is not provided
    if filename is not None:
        print("ROC: Gathering data from JSON file")
        print("ROC: NEEDS FIXING")
        data = model_saver.json_load(savedir=filename)
        time_stamps = np.array(data['params']['time_stamps'])

    # Get analysis results
    result = {'roc': [], 'anova': [], 'tuning': [], 'accuracy' : [], \
              'rule_accuracy' : []}

    # Analyze the network output and get the accuracy values
    result['accuracy'], result['rule_accuracy'] = get_perf(test_data)

    # Get ROC, ANOVA, and Tuning results as requested
    if par['roc_vars'] is not None:
        t1 = time.time()
        result['roc'] = roc_analysis(test_data)
        print('ROC time ', time.time()-t1)
    if par['anova_vars'] is not None:
        t1 = time.time()
        result['anova'] = anova_analysis(test_data)
        print('ANOVA time ', time.time()-t1)
    if par['tuning_vars'] is not None:
        t1 = time.time()
        result['tuning'] = tuning_analysis(test_data)
        print('TUNING time ', time.time()-t1)

    #plot(test_data)

    # Save analysis result
    # with open('.\savedir\dend_analysis_%s.pkl' % time.strftime('%H%M%S-%Y%m%d'), 'wb') as f:
    """
    with open('./savedir/dend_analysis_%s.pkl' % time.strftime('%H%M%S-%Y%m%d'), 'wb') as f:
        pickle.dump(result, f)
    print("ROC: Done pickling\n")
    """

    return result
