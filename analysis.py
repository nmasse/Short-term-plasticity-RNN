"""
Functions used to save model data and to perform analysis
"""

import numpy as np
from parameters import *
from sklearn import svm
import time
import pickle
import stimulus
import os
import copy
import matplotlib.pyplot as plt
from itertools import product
from scipy import signal
from scipy.optimize import curve_fit

data_dir = './savedir/'

neuron_groups = []
neuron_groups.append(range(0,par['num_exc_units'],2))
neuron_groups.append(range(1,par['num_exc_units'],2))
neuron_groups.append(range(par['num_exc_units'],par['num_exc_units']+par['num_inh_units'],2))
neuron_groups.append(range(par['num_exc_units']+1,par['num_exc_units']+par['num_inh_units'],2))
neuron_groups.append(range(par['n_hidden']))

def run_multiple():

    task_list = ['DMS']

    update_params = {
        'decode_stability':         False,
        'decoding_reps':            100,
        'simulation_reps':          100,
        'analyze_tuning':           True,
        'calculate_resp_matrix':    True,
        'suppress_analysis':        False,
        'decode_test':              False,
        'decode_rule':              False,
        'decode_match':             False,
        'svm_normalize':            True}



    for t in task_list:
        for j in range(20):
            fn = data_dir + t + str(j) + '.pkl'
            print('Analyzing ', fn)
            analyze_model_from_file(fn, savefile = fn, update_params = update_params)


def analyze_model_from_file(filename, savefile = None, update_params = {}):

    """ The first section loads the model weights and simulates the network on
        several different task conditions, saving the network activity and output """

    results = pickle.load(open(filename, 'rb'))

    if savefile is None:
        results['parameters']['save_fn'] = 'test.pkl'
    else:
        results['parameters']['save_fn'] = savefile

    update_parameters(results['parameters'])
    update_parameters(update_params)

    stim = stimulus.Stimulus()

    # generate trials with match probability at 50%
    trial_info = stim.generate_trial(test_mode = True)
    input_data = np.squeeze(np.split(trial_info['neural_input'], par['num_time_steps'], axis=0))

    h_init = results['weights']['h']

    y, h, syn_x, syn_u = run_model(input_data, h_init, \
        results['parameters']['syn_x_init'], results['parameters']['syn_u_init'], results['weights'])

    # generate trials with random sample and test stimuli, used for decoding
    trial_info_decode = stim.generate_trial(test_mode = True)
    input_data = np.squeeze(np.split(trial_info_decode['neural_input'], par['num_time_steps'], axis=0))
    _, h_decode, syn_x_decode, syn_u_decode = run_model(input_data, h_init, \
        results['parameters']['syn_x_init'], results['parameters']['syn_u_init'], results['weights'])

    # generate trials using DMS task, only used for measuring how neuronal and synaptic representations evolve in
    # a standardized way, used for figure correlating persistent activity and manipulation
    update_parameters({'trial_type': 'DMS'})
    trial_info_dms = stim.generate_trial(test_mode = True)
    input_data = np.squeeze(np.split(trial_info_dms['neural_input'], trial_info_dms['neural_input'].shape[0], axis=0))
    _, h_dms, syn_x_dms, syn_u_dms = run_model(input_data, h_init, \
        results['parameters']['syn_x_init'], results['parameters']['syn_u_init'], results['weights'])
    update_parameters(results['parameters']) # reset trial type to original value
    update_parameters(update_params)

    """ The next section performs various analysis """

    # calculate task accuracy
    results['task_accuracy'],_,_ = get_perf(trial_info['desired_output'], y, trial_info['train_mask'])
    results['task_accuracy_per_rule'] = []
    for r in np.unique(trial_info['rule']):
        ind = np.where(trial_info['rule'] == r)[0]
        acc, _, _ = get_perf(trial_info['desired_output'][:,ind,:], y[:,ind,:], trial_info['train_mask'][:, ind])
        results['task_accuracy_per_rule'].append(acc)

    print('Task accuracy',results['task_accuracy'])

    if par['calculate_resp_matrix']:
        print('calculate response matrix...')
        resp_matrix_results = calculate_response_matrix(trial_info_decode, results['weights'])
        for key, val in resp_matrix_results.items():
            if np.var(val) > 0:
                results[key] = val


    # Decode the sample direction from neuronal activity and synaptic efficacies using support vector machines
    trial_time = np.arange(0,h.shape[0]*par['dt'], par['dt'])
    trial_time_dms = np.arange(0,h_dms.shape[0]*par['dt'], par['dt'])
    if par['decoding_reps'] > 0:
        print('decoding activity...')
        decoding_results = calculate_svms(h_decode, syn_x_decode, syn_u_decode, trial_info_decode, trial_time, \
            num_reps = par['decoding_reps'], num_reps_stability = 10, decode_test = par['decode_test'], \
            decode_rule = par['decode_rule'], decode_match = par['decode_match'])
        for key, val in decoding_results.items():
            if np.var(val) > 0:
                results[key] = val

        if par['trial_type'] in ['DMS', 'DMC', 'DMRS90', 'DMRS90ccw', 'DMRS45', 'DMRS180', 'location_DMS']:
            for key, val in decoding_results.items():
                if np.var(val) > 0:
                    results[key + '_dms'] = val

        else:
            # Calculate decoding for a DMS trial, used to correlate persistent activity and manipulation
            update_parameters({'trial_type': 'DMS'})
            decoding_results = calculate_svms(h_dms, syn_x_dms, syn_u_dms, trial_info_dms, trial_time_dms, \
                num_reps = par['decoding_reps'], num_reps_stability = 0, decode_test = par['decode_test'], decode_rule = par['decode_rule'])
            for key, val in decoding_results.items():
                if np.var(val) > 0:
                    results[key + '_dms'] = val
            update_parameters(results['parameters'])
            update_parameters(update_params)



    # Calculate neuronal and synaptic sample motion tuning
    if par['analyze_tuning']:
        print('calculate tuning...')

        tuning_results = calculate_tuning(h_decode, syn_x_decode, syn_u_decode, \
            trial_info_decode, trial_time, results['weights'], calculate_test = par['decode_test'])
        for key, val in tuning_results.items():
            if np.var(val) > 0:
                results[key] = val

        # Calculate tuning for a DMS trial, used to correlate persistent activity and manipulation
        if par['trial_type'] in ['DMS', 'DMC', 'DMRS90', 'DMRS90ccw','DMRS45', 'DMRS180', 'location_DMS']:
            for key, val in tuning_results.items():
                if np.var(val) > 0:
                    results[key + '_dms'] = val
        else:
            update_parameters({'trial_type': 'DMS'})
            tuning_results = calculate_tuning(h_dms, syn_x_dms, syn_u_dms, \
                trial_info_dms, trial_time_dms, results['weights'], calculate_test = False)
            for key, val in tuning_results.items():
                if np.var(val) > 0:
                    results[key + '_dms'] = val
            update_parameters(results['parameters'])
            update_parameters(update_params)


    # Calculate mean sample traces
    results['h_sample_mean'] = np.zeros((results['parameters']['num_time_steps'], results['parameters']['n_hidden'], \
        results['parameters']['num_motion_dirs']), dtype = np.float32)
    for i in range(results['parameters']['num_motion_dirs']):
        ind = np.where(trial_info_decode['sample'] == i)[0]
        results['h_sample_mean'][:,:,i] = np.mean(h_decode[:,ind,:], axis = 1)

    # Calculate the neuronal and synaptic contributions towards solving the task
    if par['simulation_reps'] > 0:
        print('simulating network...')
        simulation_results = simulate_network(trial_info, h, syn_x, \
            syn_u, results['weights'], num_reps = par['simulation_reps'])
        for key, val in simulation_results.items():
            if np.var(val) > 0:
                results[key] = val

    # Save the analysis results
    pickle.dump(results, open(savefile, 'wb') )
    print('Analysis results saved in ', savefile)



def calculate_svms(h, syn_x, syn_u, trial_info, trial_time, num_reps = 20, num_reps_stability = 5, \
    decode_test = False, decode_rule = False, decode_match = False, decode_neuronal_groups = False):

    """ Calculates neuronal and synaptic decoding accuracies uisng support vector machines """

    lin_clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr', shrinking=False, tol=1e-3)

    num_time_steps = len(trial_time)
    decoding_results = {}

    # The synaptic efficacy is the product of syn_x and syn_u. Will decode sample
    # direction from this value
    syn_efficacy = syn_x*syn_u

    if par['trial_type'] == 'DMC':

        # Will also calculate the category decoding accuracies, assuming the first half of
        # the sample direction belong to category 1, and the second half belong to category 2
        num_motion_dirs = len(np.unique(trial_info['sample']))
        sample = np.floor(trial_info['sample']/(num_motion_dirs/2)*np.ones_like(trial_info['sample']))
        test = np.floor(trial_info['test']/(num_motion_dirs/2)*np.ones_like(trial_info['sample']))
        rule = trial_info['rule']
        match = np.array(trial_info['match'])
    elif par['trial_type'] == 'dualDMS':
        sample = trial_info['sample']
        rule = trial_info['rule'][:,0] + 2*trial_info['rule'][:,1]
        par['num_rules'] = 4
        match = np.array(trial_info['match'])
    elif par['trial_type'] == 'locationDMS':
        sample = trial_info['sample'][:, 0]
        test = trial_info['test']
        rule = trial_info['rule']
        match = np.array(trial_info['match'])
    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        sample = trial_info['sample']
        rule = trial_info['rule']
        test = np.array(trial_info['test'][:,0])
        match = np.array(trial_info['match'][:,0])
    elif par['trial_type'] == 'DMS+DMC':
        # rule 0 is DMS, rule 1 is DMC
        ind_rule = np.where(trial_info['rule']==1)[0]
        num_motion_dirs = len(np.unique(trial_info['sample']))
        sample = np.array(trial_info['sample'])
        test = np.array(trial_info['test'])
        # change DMC sample motion directions into categories
        sample[ind_rule] = np.floor(trial_info['sample'][ind_rule]/(num_motion_dirs/2)*np.ones_like(trial_info['sample'][ind_rule]))
        test[ind_rule] = np.floor(trial_info['test'][ind_rule]/(num_motion_dirs/2)*np.ones_like(trial_info['sample'][ind_rule]))
        rule = trial_info['rule']
        match = np.array(trial_info['match'])
    else:
        sample = np.array(trial_info['sample'])
        rule = np.array(trial_info['rule'])
        match = np.array(trial_info['match'])

    if trial_info['test'].ndim == 2:
        test = trial_info['test'][:,0]
    else:
        test = np.array(trial_info['test'])

    if len(np.unique(np.array(trial_info['rule']))) == 1 and decode_rule:
        print('Only one unique rule; setting decode rule to False')
        decode_rule = False


    decoding_results['neuronal_sample_decoding'], decoding_results['synaptic_sample_decoding'], \
        decoding_results['neuronal_sample_decoding_stability'], decoding_results['synaptic_sample_decoding_stability'] = \
        svm_wraper(lin_clf, h, syn_efficacy, sample, rule, num_reps, num_reps_stability, trial_time)

    to = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
    print('Neuronal and synaptic delay period decoding', \
        np.mean(decoding_results['neuronal_sample_decoding'][0,0,:,to-10:to]), \
        np.mean(decoding_results['synaptic_sample_decoding'][0,0,:,to-10:to]))

    if decode_test:
        decoding_results['neuronal_test_decoding'], decoding_results['synaptic_test_decoding'] ,_ ,_ = \
            svm_wraper(lin_clf, h, syn_efficacy, test, rule, num_reps, 0, trial_time)

    if decode_match:
        decoding_results['neuronal_match_decoding'], decoding_results['synaptic_match_decoding'] ,_ ,_ = \
            svm_wraper(lin_clf, h, syn_efficacy, match, rule, num_reps, 0, trial_time)

    if decode_rule:
        decoding_results['neuronal_rule_decoding'], decoding_results['synaptic_rule_decoding'] ,_ ,_ = \
            svm_wraper(lin_clf, h, syn_efficacy, trial_info['rule'], np.zeros_like(rule), num_reps, 0, trial_time)


    if decode_neuronal_groups:

        decoding_results['neuronal_sample_decoding_group'] = []
        decoding_results['synaptic_sample_decoding_group'] = []
        decoding_results['neuronal_test_decoding_group'] = []
        decoding_results['synaptic_test_decoding_group'] = []
        decoding_results['neuronal_match_decoding_group'] = []
        decoding_results['synaptic_match_decoding_group'] = []

        for i in range(4):
            neuronal_decoding, synaptic_decoding, _, _ = \
                svm_wraper(lin_clf, h[neuron_groups[i],:,:], syn_efficacy[neuron_groups[i],:,:], sample, rule, 20, 0, trial_time)
            decoding_results['neuronal_sample_decoding_group'].append(neuronal_decoding)
            decoding_results['synaptic_sample_decoding_group'].append(synaptic_decoding)

            neuronal_decoding, synaptic_decoding, _, _ = \
                svm_wraper(lin_clf, h[neuron_groups[i],:,:], syn_efficacy[neuron_groups[i],:,:], test, rule, 20, 0, trial_time)
            decoding_results['neuronal_test_decoding_group'].append(neuronal_decoding)
            decoding_results['synaptic_test_decoding_group'].append(synaptic_decoding)

            neuronal_decoding, synaptic_decoding, _, _ = \
                svm_wraper(lin_clf, h[neuron_groups[i],:,:], syn_efficacy[neuron_groups[i],:,:], match, rule, 20, 0, trial_time)
            decoding_results['neuronal_match_decoding_group'].append(neuronal_decoding)
            decoding_results['synaptic_match_decoding_group'].append(synaptic_decoding)


    return decoding_results



def svm_wraper_simple(lin_clf, h, syn_eff, stimulus, rule, num_reps, num_reps_stability, trial_time):

    train_pct = 0.75
    num_time_steps, num_trials, _ = h.shape
    num_rules = len(np.unique(rule))

    # 4 refers to four data_type, normalized neural activity and synaptic efficacy, and unnormalized neural activity and synaptic efficacy
    score = np.zeros((4, num_rules, par['num_receptive_fields'], num_reps, num_time_steps), dtype = np.float32)
    score_dynamic = np.zeros((4, num_rules, par['num_receptive_fields'], num_reps_stability, num_time_steps, num_time_steps), dtype = np.float32)

    # number of reps used to calculate encoding stability should not be larger than number of normal deocding reps
    num_reps_stability = np.minimum(num_reps_stability, num_reps)

    for r in range(num_rules):
        ind_rule = np.where(rule==r)[0]
        for rf in range(par['num_receptive_fields']):
            if par['trial_type'] == 'dualDMS':
                labels = np.array(stimulus[:,rf])
            else:
                labels = np.array(stimulus)

            for rep in range(num_reps):

                q = np.random.permutation(num_trials)
                ind_train = q[:round(train_pct*num_trials)]
                ind_test = q[round(train_pct*num_trials):]

                for data_type in [0]:
                    if data_type == 0:
                        z = normalize_values(h, ind_train)
                    elif data_type == 1:
                        z = normalize_values(syn_eff, ind_train)
                    elif data_type == 2:
                        z = np.array(h)
                    elif data_type == 3:
                        z = np.array(syn_eff)

                    for t in range(num_time_steps):
                        lin_clf.fit(z[:,t,ind_train].T, labels[ind_train])
                        predicted_sample = lin_clf.predict(z[:,t,ind_test].T)
                        score[data_type, r, rf, rep, t] = np.mean( labels[ind_test]==predicted_sample)

                        if rep < num_reps_stability and par['decode_stability']:
                            for t1 in range(num_time_steps):
                                predicted_sample = lin_clf.predict(z[:,t1,ind_test].T)
                                score_dynamic[data_type, r, rf, rep, t, t1] = np.mean(labels[ind_test]==predicted_sample)

        return score, score_dynamic


def svm_wraper(lin_clf, h, syn_eff, conds, rule, num_reps, num_reps_stability, trial_time):

    """ Wraper function used to decode sample/test or rule information
        from hidden activity (h) and synaptic efficacies (syn_eff) """

    train_pct = 0.75
    num_time_steps, num_trials, _ = h.shape
    num_rules = len(np.unique(rule))

    score_h = np.zeros((num_rules, par['num_receptive_fields'], num_reps, num_time_steps), dtype = np.float32)
    score_syn_eff = np.zeros((num_rules, par['num_receptive_fields'], num_reps, num_time_steps), dtype = np.float32)
    score_h_stability = np.zeros((num_rules, par['num_receptive_fields'], num_reps_stability, num_time_steps, num_time_steps), dtype = np.float32)
    score_syn_eff_stability = np.zeros((num_rules, par['num_receptive_fields'], num_reps_stability, num_time_steps, num_time_steps), dtype = np.float32)

    # number of reps used to calculate encoding stability should not be larger than number of normal deocding reps
    num_reps_stability = np.minimum(num_reps_stability, num_reps)

    for r in range(num_rules):
        ind_rule = np.where(rule==r)[0]
        for n in range(par['num_receptive_fields']):
            if par['trial_type'] == 'dualDMS':
                current_conds = np.array(conds[:,n])
            else:
                current_conds = np.array(conds)

            num_conds = len(np.unique(conds[ind_rule]))
            if num_conds <= 2:
                trials_per_cond = 100
            else:
                trials_per_cond = 25

            equal_train_ind = np.zeros((num_conds*trials_per_cond), dtype = np.uint16)
            equal_test_ind = np.zeros((num_conds*trials_per_cond), dtype = np.uint16)

            cond_ind = []
            for c in range(num_conds):
                cond_ind.append(ind_rule[np.where(current_conds[ind_rule] == c)[0]])
                if len(cond_ind[c]) < 4:
                    print('Not enough trials for this condition!')
                    print('Setting cond_ind to [0,1,2,3]')
                    cond_ind[c] = [0,1,2,3]

            for rep in range(num_reps):
                for c in range(num_conds):
                    u = range(c*trials_per_cond, (c+1)*trials_per_cond)
                    q = np.random.permutation(len(cond_ind[c]))
                    i = int(np.round(len(cond_ind[c])*train_pct))
                    train_ind = cond_ind[c][q[:i]]
                    test_ind = cond_ind[c][q[i:]]

                    q = np.random.randint(len(train_ind), size = trials_per_cond)
                    equal_train_ind[u] =  train_ind[q]
                    q = np.random.randint(len(test_ind), size = trials_per_cond)
                    equal_test_ind[u] =  test_ind[q]

                score_h[r,n,rep,:] = calc_svm(lin_clf, h, current_conds, current_conds, equal_train_ind, equal_test_ind)
                score_syn_eff[r,n,rep,:] = calc_svm(lin_clf, syn_eff, current_conds, current_conds, equal_train_ind, equal_test_ind)

                if par['decode_stability'] and rep < num_reps_stability:
                    score_h_stability[r,n,rep,:,:] = calc_svm_stability(lin_clf, h,  current_conds, current_conds, equal_train_ind, equal_test_ind)
                    score_syn_eff_stability[r,n,rep,:,:] = calc_svm_stability(lin_clf, syn_eff,  current_conds, current_conds, equal_train_ind, equal_test_ind)

    return score_h, score_syn_eff, score_h_stability, score_syn_eff_stability


def calc_svm_stability(lin_clf, y, train_conds, test_conds, train_ind, test_ind):

    n_test_inds = len(test_ind)
    score = np.zeros((par['num_time_steps'], par['num_time_steps']))

    #y = normalize_values(y, train_ind)

    t0 = time.time()
    for t in range(par['dead_time']//par['dt'], par['num_time_steps']):
        lin_clf.fit(y[t,train_ind,:], train_conds[train_ind])
        for t1 in range(par['dead_time']//par['dt'],par['num_time_steps']):
            dec = lin_clf.predict(y[t1,test_ind,:])
            score[t, t1] = np.mean(test_conds[test_ind] == dec)

    return score

def calc_svm(lin_clf, y, train_conds, test_conds, train_ind, test_ind):

    n_test_inds = len(test_ind)
    score = np.zeros((par['num_time_steps']))

    y = normalize_values(y, train_ind)

    for t in range(par['dead_time']//par['dt'], par['num_time_steps']):
        lin_clf.fit(y[t,train_ind,:], train_conds[train_ind])
        dec = lin_clf.predict(y[t,test_ind,:])
        score[t] = np.mean(test_conds[test_ind]==dec)

    return score

def normalize_values(z, train_ind):

    if not par['svm_normalize']:
        return z

    # normalize values between 0 and 1
    for t, n in product(range(z.shape[0]), range(z.shape[2])): # loop across time, neurons
        m1 = z[t,train_ind,n].min()
        m2 = z[t,train_ind,n].max()
        z[t,:,n] -= m1
        if m2 > m1:
            z[t,:,n] /= (m2-m1)

    return z



def calculate_response_matrix(trial_info, network_weights):

    test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']

    resp_matrix_results = {
        'resp_no_suppresion' : np.zeros((par['n_hidden'], par['num_motion_dirs'], par['num_motion_dirs']), dtype = np.float32),
        'resp_suppresion' : np.zeros((par['n_hidden'], par['n_hidden'], par['num_motion_dirs'], par['num_motion_dirs']), dtype = np.float32)}

    x = np.split(trial_info['neural_input'],par['num_time_steps'],axis=0)

    _, h, _, _ = run_model(x, network_weights['h'], par['syn_x_init'], par['syn_u_init'], network_weights)
    resp_matrix_results['resp_no_suppresion'] = average_test_response(h, trial_info, test_onset)

    for n in range(par['n_hidden']):
        suppress_activity = np.ones((par['num_time_steps'], par['n_hidden']))
        suppress_activity[test_onset:, n] = 0 # suppress activity starting from test onset

        suppress_activity = np.split(suppress_activity, par['num_time_steps'], axis=0)
        _, h,_,_ = run_model(x, network_weights['h'], par['syn_x_init'], par['syn_u_init'], \
            network_weights, suppress_activity = suppress_activity)
        resp_matrix_results['resp_suppresion'][n,:,:,:] = average_test_response(h, trial_info, test_onset)


    return resp_matrix_results

def average_test_response(h, trial_info, test_onset):

    resp = np.zeros((par['n_hidden'], par['num_motion_dirs'], par['num_motion_dirs']), dtype = np.float32)
    h_test = np.mean(h[test_onset:, :, : ], axis=0)
    for i in range(par['num_motion_dirs']):
        for j in range(par['num_motion_dirs']):
            ind = np.where((trial_info['sample']==i)*(trial_info['test']==j))[0]
            resp[:, i, j] = np.mean(h_test[ind, :], axis = 0)

    return resp


def simulate_network(trial_info, h, syn_x, syn_u, network_weights, num_reps = 20):

    epsilon = 1e-3

    # Simulation will start from the start of the test period until the end of trial
    if par['trial_type'] == 'dualDMS':
        test_onset = [(par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+par['test_time'])//par['dt']]
    elif  par['trial_type'] in ['ABBA','ABCA']:
        test_onset = [(par['dead_time']+par['fix_time']+par['sample_time']+i*par['ABBA_delay'])//par['dt'] for i in range(1,2)]
    elif  par['trial_type'] in ['DMS', 'DMC', 'DMRS90', 'DMRS90ccw']:
        test_onset = []
        test_onset.append((par['dead_time']+par['fix_time']+par['sample_time'])//par['dt'])
        test_onset.append((par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt'])


    num_test_periods = len(test_onset)

    suppression_time_range = [range(test_onset[0]-200//par['dt'], test_onset[0])]

    syn_efficacy = syn_x*syn_u
    test = np.array(trial_info['test'])
    sample = np.array(trial_info['sample'])
    if test.ndim == 2:
        test = test[:, 0]
    elif test.ndim == 3:
        test = test[:, 0, 0]
    test_dir = np.ones((len(test), 3))
    test_dir[:,1] = np.cos(2*np.pi*test/par['num_motion_dirs'])
    test_dir[:,2] = np.sin(2*np.pi*test//par['num_motion_dirs'])

    trial_length, batch_train_size, n_hidden = h.shape
    num_grp_reps = 5

    simulation_results = {
        'simulation_accuracy'            : np.zeros((par['num_rules'], num_test_periods, num_reps)),
        'accuracy_neural_shuffled'      : np.zeros((par['num_rules'], num_test_periods, num_reps)),
        'accuracy_syn_shuffled'         : np.zeros((par['num_rules'], num_test_periods, num_reps)),
        'accuracy_suppression'          : np.zeros((par['num_rules'], len(suppression_time_range), len(neuron_groups), num_grp_reps)),
        'accuracy_neural_shuffled_grp'  : np.zeros((par['num_rules'], num_test_periods, len(neuron_groups), num_grp_reps)),
        'accuracy_syn_shuffled_grp'     : np.zeros((par['num_rules'], num_test_periods, len(neuron_groups), num_grp_reps)),
        'synaptic_pev_test_suppression' : np.zeros((par['num_rules'], num_test_periods, len(neuron_groups), n_hidden, trial_length)),
        'synaptic_pref_dir_test_suppression': np.zeros((par['num_rules'], num_test_periods, len(neuron_groups), n_hidden, trial_length))}


    mask = np.array(trial_info['train_mask'])
    if par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        t0 = (par['dead_time']+par['fix_time']+par['sample_time'] + 2*par['ABBA_delay'])//par['dt']
        mask[:t0,:] = 0
        t0 = (par['dead_time']+par['fix_time']+par['sample_time'] + 4*par['ABBA_delay'])//par['dt']
        mask[t0:,:] = 0


    for r, t in product(range(par['num_rules']), range(num_test_periods)):

        test_length = trial_length - test_onset[t]
        trial_ind = np.where(trial_info['rule']==r)[0]
        train_mask = mask[test_onset[t]:,trial_ind]
        x = np.split(trial_info['neural_input'][test_onset[t]:,trial_ind, :],test_length,axis=0)
        desired_output = trial_info['desired_output'][test_onset[t]:,trial_ind, :]

        for n in range(num_reps):

            # Calculating behavioral accuracy without shuffling
            hidden_init = np.copy(h[test_onset[t]-1,trial_ind,:])
            syn_x_init = np.copy(syn_x[test_onset[t]-1,trial_ind,:])
            syn_u_init = np.copy(syn_u[test_onset[t]-1,trial_ind,:])
            y, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
            simulation_results['simulation_accuracy'][r,t,n] ,_ ,_ = get_perf(desired_output, y, train_mask)

            # Keep the synaptic values fixed, permute the neural activity
            ind_shuffle = np.random.permutation(len(trial_ind))
            hidden_init = np.copy(hidden_init[ind_shuffle, :])
            y, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
            simulation_results['accuracy_neural_shuffled'][r,t,n] ,_ ,_ = get_perf(desired_output, y, train_mask)


            # Keep the hidden values fixed, permute synaptic values
            hidden_init = np.copy(h[test_onset[t]-1,trial_ind, :])
            syn_x_init = np.copy(syn_x_init[ind_shuffle, :])
            syn_u_init = np.copy(syn_u_init[ind_shuffle, :])
            y, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
            simulation_results['accuracy_syn_shuffled'][r,t,n] ,_ ,_ = get_perf(desired_output, y, train_mask)

        for n in range(num_grp_reps): # Neuron group shuffling

            for g in range(len(neuron_groups)):

                # reset everything
                hidden_init = np.copy(h[test_onset[t]-1,trial_ind,:])
                syn_x_init = np.copy(syn_x[test_onset[t]-1,trial_ind,:])
                syn_u_init = np.copy(syn_u[test_onset[t]-1,trial_ind,:])

                # shuffle neuronal activity
                ind_shuffle = np.random.permutation(len(trial_ind))
                for neuron_num in neuron_groups[g]:
                    hidden_init[:, neuron_num] = hidden_init[ind_shuffle, neuron_num]
                y, _, syn_x_hist, syn_u_hist = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
                simulation_results['accuracy_neural_shuffled_grp'][r,t,g,n] ,_ ,_ \
                    = get_perf(desired_output, y, train_mask)

                if par['trial_type'] in ['ABBA','ABCA']:
                    syn_efficacy = syn_x_hist*syn_u_hist
                    for hidden_num in range(par['n_hidden']):
                        for t1 in range(test_length):
                            weights = np.linalg.lstsq(test_dir[trial_ind,:], syn_efficacy[t1,trial_ind,hidden_num],rcond=None)
                            weights = np.reshape(weights[0],(3,1))
                            pred_err = syn_efficacy[t1,trial_ind,hidden_num] - np.dot(test_dir[trial_ind,:], weights).T
                            mse = np.mean(pred_err**2)
                            response_var = np.var(syn_efficacy[t1,trial_ind,hidden_num])
                            simulation_results['synaptic_pev_test_shuffled'][r,t,g,n, hidden_num,t1+test_onset[t]] = 1 - mse/(response_var + epsilon)
                            simulation_results['synaptic_pref_dir_test_shuffled'][r,t,g,n,hidden_num,t1+test_onset[t]] = np.arctan2(weights[2,0],weights[1,0])

                # reset neuronal activity, shuffle synaptic activity
                hidden_init = h[test_onset[t]-1,trial_ind,:]
                for neuron_num in neuron_groups[g]:
                    syn_x_init[:,neuron_num] = syn_x_init[ind_shuffle,neuron_num]
                    syn_u_init[:,neuron_num] = syn_u_init[ind_shuffle,neuron_num]
                y, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
                simulation_results['accuracy_syn_shuffled_grp'][r,t,g,n] ,_ ,_ = get_perf(desired_output, y, train_mask)

                if par['trial_type'] in ['ABBA','ABCA']:
                    syn_efficacy = syn_x_hist*syn_u_hist
                    for hidden_num in range(par['n_hidden']):
                        for t1 in range(test_length):
                            weights = np.linalg.lstsq(test_dir[trial_ind,:], syn_efficacy[t1,trial_ind,hidden_num],rcond=None)
                            weights = np.reshape(weights[0],(3,1))
                            pred_err = syn_efficacy[t1,trial_ind,hidden_num] - np.dot(test_dir[trial_ind,:], weights).T
                            mse = np.mean(pred_err**2)
                            response_var = np.var(syn_efficacy[hidden_num,t1,trial_ind])
                            simulation_results['synaptic_pev_test_shuffled'][r,t,g,n, hidden_num,t1+test_onset[t]] = 1 - mse/(response_var + epsilon)
                            simulation_results['synaptic_pref_dir_test_shuffled'][r,t,g,n,hidden_num,t1+test_onset[t]] = np.arctan2(weights[2,0],weights[1,0])


    if par['suppress_analysis'] and False:
        """
        if par['trial_type'] == 'ABBA' or  par['trial_type'] == 'ABCA':
            test_onset_sup = (par['fix_time']+par['sample_time']+par['ABBA_delay'])//par['dt']
        elif par['trial_type'] == 'DMS' or par['trial_type'] == 'DMC' or \
            par['trial_type'] == 'DMRS90' or par['trial_type'] == 'DMRS180':
            test_onset_sup = (par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
        """
        x = np.split(trial_info['neural_input'][:,trial_ind,:],trial_length,axis=0)
        desired_output = trial_info['desired_output'][:,trial_ind,:]
        train_mask = np.copy(mask[:,trial_ind])
        """
        train_mask = trial_info['train_mask'][:,trial_ind]
        if par['trial_type'] == 'ABBA' or  par['trial_type'] == 'ABCA':
            train_mask[test_onset_sup + par['ABBA_delay']//par['dt']:, :] = 0
        """

        syn_x_init = np.copy(syn_x[0,trial_ind,:])
        syn_u_init = np.copy(syn_u[0,trial_ind,:])
        hidden_init = np.copy(h[0,trial_ind,:])

        y, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
        acc, acc_non_match, acc_match = get_perf(desired_output, y, train_mask)
        simulation_results['accuracy_no_suppression'] = np.array([acc, acc_non_match, acc_match])

        for k, k1 in product(range(len(suppression_time_range)), range(len(neuron_groups))):

            suppress_activity = np.ones((trial_length,par['n_hidden']))
            for m1, m2 in product(neuron_groups[k1], suppression_time_range[k]):
                suppress_activity[m2,m1] = 0

            suppress_activity = np.split(suppress_activity, trial_length, axis=0)
            syn_x_init = np.array(syn_x[0,trial_ind,:])
            syn_u_init = np.array(syn_u[0,trial_ind,:])
            hidden_init = np.array(h[0,trial_ind,:])

            y, _, syn_x_sim, syn_u_sim = run_model(x, hidden_init, syn_x_init, \
                syn_u_init, network_weights, suppress_activity = suppress_activity)
            acc, acc_non_match, acc_match = get_perf(desired_output, y, train_mask)
            simulation_results['accuracy_suppression'][r,k,k1,:] = np.array([acc, acc_non_match, acc_match])

            syn_efficacy = syn_x_sim*syn_u_sim
            for hidden_num in range(par['n_hidden']):
                for t1 in range(syn_x_sim.shape[1]):
                    weights = np.linalg.lstsq(test_dir[trial_ind,:], syn_efficacy[t1,trial_ind,t1,trial_ind,hidden_num])
                    weights = np.reshape(weights[0],(3,1))
                    pred_err = syn_efficacy[t1,trial_ind,t1,trial_ind,hidden_num] - np.dot(test_dir[trial_ind,:], weights).T
                    mse = np.mean(pred_err**2)
                    response_var = np.var(syn_efficacy[hidden_num,t1,trial_ind])
                    simulation_results['synaptic_pev_test_suppression'][r,k,k1, hidden_num,t1] = 1 - mse/(response_var+1e-9)
                    simulation_results['synaptic_pref_dir_test_suppression'][r,k,k1,hidden_num,t1] = np.arctan2(weights[2,0],weights[1,0])


    return simulation_results

def calculate_tuning(h, syn_x, syn_u, trial_info, trial_time, network_weights, calculate_test = False):

    """ Calculates neuronal and synaptic sample motion direction tuning """

    epsilon = 1e-9

    num_test_stimuli = 1 # only analyze the first test stimulus
    mask = np.array(trial_info['train_mask'])

    if par['trial_type'] == 'dualDMS':
        sample = trial_info['sample']
        test = trial_info['test'][:,:,:num_test_stimuli]
        rule = trial_info['rule'][:,0] + 2*trial_info['rule'][:,1]
        par['num_rules'] = 4
        par['num_receptive_fields'] = 2
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
        suppression_time_range = [range(test_onset-50//par['dt'], test_onset)]

    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        test = trial_info['test'][:,:num_test_stimuli]
        rule = np.array(trial_info['rule'])
        sample = np.reshape(np.array(trial_info['sample']),(par['batch_size'], 1))
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['ABBA_delay'])//par['dt']
        suppression_time_range = [range(test_onset-200//par['dt'], test_onset)]
        # onyl want to examine accuracy on 2nd pulse
        t0 = (par['dead_time']+par['fix_time']+par['sample_time'] + 2*par['ABBA_delay'])//par['dt']
        mask[:t0,:] = 0
        t0 = (par['dead_time']+par['fix_time']+par['sample_time'] + 4*par['ABBA_delay'])//par['dt']
        mask[t0:,:] = 0

    elif par['trial_type'] == 'location_DMS':
        par['num_receptive_fields'] = 1
        test = np.reshape(np.array(trial_info['test']),(par['batch_size'], 1))
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
        sample = np.reshape(np.array(trial_info['sample']),(par['batch_size'], 1))
        rule = np.array(trial_info['rule'])
        match = np.array(trial_info['match'])
        suppression_time_range = [range(test_onset-200//par['dt'], test_onset)]
    else:
        rule = np.array(trial_info['rule'])
        match = np.array(trial_info['match'])
        sample = np.reshape(np.array(trial_info['sample']),(par['batch_size'], 1))
        test = np.reshape(np.array(trial_info['test']),(par['batch_size'], 1))
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
        suppression_time_range = [range(test_onset-50//par['dt'], test_onset)]

    num_time_steps = len(trial_time)

    tuning_results = {
        'neuronal_pref_dir'     : np.zeros((par['n_hidden'],  par['num_rules'], par['num_receptive_fields'], num_time_steps), dtype=np.float32),
        'synaptic_pref_dir'     : np.zeros((par['n_hidden'],  par['num_rules'], par['num_receptive_fields'], num_time_steps), dtype=np.float32),
        'neuronal_pev'          : np.zeros((par['n_hidden'],  par['num_rules'], par['num_receptive_fields'], num_time_steps), dtype=np.float32),
        'synaptic_pev'          : np.zeros((par['n_hidden'],  par['num_rules'], par['num_receptive_fields'], num_time_steps), dtype=np.float32),
        'neuronal_pev_test'     : np.zeros((par['n_hidden'],  par['num_rules'], par['num_receptive_fields'], num_time_steps), dtype=np.float32),
        'synaptic_pev_test'     : np.zeros((par['n_hidden'],  par['num_rules'], par['num_receptive_fields'], num_time_steps), dtype=np.float32),
        'neuronal_pev_match'    : np.zeros((par['n_hidden'],  par['num_rules'], par['num_receptive_fields'], num_time_steps), dtype=np.float32),
        'synaptic_pev_match'    : np.zeros((par['n_hidden'],  par['num_rules'], par['num_receptive_fields'], num_time_steps), dtype=np.float32),
        'neuronal_pref_dir_test': np.zeros((par['n_hidden'],  par['num_rules'], par['num_receptive_fields'], num_time_steps), dtype=np.float32),
        'synaptic_pref_dir_test': np.zeros((par['n_hidden'],  par['num_rules'], par['num_receptive_fields'],  num_time_steps), dtype=np.float32),
        'acc_neuronal_suppression': np.zeros((par['num_rules'], len(suppression_time_range), len(neuron_groups), 3)),
        'neuronal_sample_tuning': np.zeros((par['n_hidden'],  par['num_rules'], par['num_motion_dirs'], par['num_receptive_fields'], num_time_steps), dtype=np.float32),
        'synaptic_sample_tuning': np.zeros((par['n_hidden'],  par['num_rules'], par['num_motion_dirs'], par['num_receptive_fields'], num_time_steps), dtype=np.float32),
        'synaptic_pev_test_suppression'    : np.zeros((par['num_rules'], len(suppression_time_range), len(neuron_groups), par['n_hidden'], num_time_steps)),
        'synaptic_pref_dir_test_suppression': np.zeros((par['num_rules'], len(suppression_time_range), len(neuron_groups), par['n_hidden'], num_time_steps))}


    syn_efficacy = syn_x*syn_u

    sample_dir = np.ones((par['batch_size'], 3, par['num_receptive_fields']))
    for rf in range(par['num_receptive_fields']):
        sample_dir[:,1, rf] = np.cos(2*np.pi*sample[:,rf]/par['num_motion_dirs'])
        sample_dir[:,2, rf] = np.sin(2*np.pi*sample[:,rf]/par['num_motion_dirs'])

    test_dir = np.ones((par['batch_size'], 3, par['num_receptive_fields']))
    for rf in range(par['num_receptive_fields']):
        test_dir[:,1, rf] = np.reshape(np.cos(2*np.pi*test[:, rf]/par['num_motion_dirs']), (par['batch_size']))
        test_dir[:,2, rf] = np.reshape(np.sin(2*np.pi*test[:, rf]/par['num_motion_dirs']), (par['batch_size']))

    for r in range(par['num_rules']):
        trial_ind = np.where((rule==r))[0]
        for n in range(par['n_hidden']):
            for t in range(num_time_steps):

                # Mean sample response
                for md in range(par['num_motion_dirs']):
                    for rf in range(par['num_receptive_fields']):

                        ind_motion_dir = np.where((rule==r)*(sample[:,rf]==md))[0]
                        tuning_results['neuronal_sample_tuning'][n,r,md,rf,t] = np.mean(h[t,ind_motion_dir,n])
                        tuning_results['synaptic_sample_tuning'][n,r,md,rf,t] = np.mean(syn_efficacy[t,ind_motion_dir,n])

                for rf in range(par['num_receptive_fields']):
                    # Neuronal sample tuning

                    weights = np.linalg.lstsq(sample_dir[trial_ind,:,rf], h[t,trial_ind,n], rcond=None)
                    weights = np.reshape(weights[0],(3,1))
                    pred_err = h[t,trial_ind,n] - np.dot(sample_dir[trial_ind,:,rf], weights).T
                    mse = np.mean(pred_err**2)
                    response_var = np.var(h[t,trial_ind,n])
                    if response_var > epsilon:
                        tuning_results['neuronal_pev'][n,r,rf,t] = 1 - mse/(response_var + epsilon)
                        tuning_results['neuronal_pref_dir'][n,r,rf,t] = np.arctan2(weights[2,0],weights[1,0])

                    if calculate_test:
                        weights = np.linalg.lstsq(test_dir[trial_ind,:,rf], h[t,trial_ind,n], rcond=None)
                        weights = np.reshape(weights[0],(3,1))
                        pred_err = h[t,trial_ind,n] - np.dot(test_dir[trial_ind,:,rf], weights).T
                        mse = np.mean(pred_err**2)
                        response_var = np.var(h[t,trial_ind,n])
                        if response_var > epsilon:
                            tuning_results['neuronal_pev_test'][n,r,rf,t] = 1 - mse/(response_var + epsilon)
                            tuning_results['neuronal_pref_dir_test'][n,r,rf,t] = np.arctan2(weights[2,0],weights[1,0])


                    # Synaptic sample tuning
                    weights = np.linalg.lstsq(sample_dir[trial_ind,:,rf], syn_efficacy[t,trial_ind,n], rcond=None)
                    weights = np.reshape(weights[0],(3,1))
                    pred_err = syn_efficacy[t,trial_ind,n] - np.dot(sample_dir[trial_ind,:,rf], weights).T
                    mse = np.mean(pred_err**2)
                    response_var = np.var(syn_efficacy[t,trial_ind,n])
                    tuning_results['synaptic_pev'][n,r,rf,t] = 1 - mse/(response_var + epsilon)
                    tuning_results['synaptic_pref_dir'][n,r,rf,t] = np.arctan2(weights[2,0],weights[1,0])

                    if calculate_test:
                        weights = np.linalg.lstsq(test_dir[trial_ind,:,rf], syn_efficacy[t,trial_ind,n], rcond=None)
                        weights = np.reshape(weights[0],(3,1))
                        pred_err = syn_efficacy[t,trial_ind,n] - np.dot(test_dir[trial_ind,:,rf], weights).T
                        mse = np.mean(pred_err**2)
                        response_var = np.var(syn_efficacy[t,trial_ind,n])
                        tuning_results['synaptic_pev_test'][n,r,rf,t] = 1 - mse/(response_var + epsilon)
                        tuning_results['synaptic_pref_dir_test'][n,r,rf,t] = np.arctan2(weights[2,0],weights[1,0])

        if par['suppress_analysis']:

            x = np.split(trial_info['neural_input'][:,trial_ind,:],num_time_steps,axis=0)
            y = trial_info['desired_output'][:,trial_ind,:]
            train_mask = np.array(mask[:,trial_ind])
            syn_x_init = np.array(syn_x[0,trial_ind,:])
            syn_u_init = np.array(syn_u[0,trial_ind,:])
            hidden_init = np.array(h[0,trial_ind,:])

            y_hat, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
            acc, acc_non_match, acc_match = get_perf(y, y_hat, train_mask)
            tuning_results['accuracy_no_suppression'] = np.array([acc, acc_non_match, acc_match])

            for k in range(len(suppression_time_range)):
                for k1 in range(len(neuron_groups)):

                    suppress_activity = np.ones((num_time_steps, par['n_hidden']))
                    for m1, m2 in product(neuron_groups[k1], suppression_time_range[k]):
                        suppress_activity[m2, m1] = 0

                    suppress_activity = np.split(suppress_activity, num_time_steps, axis=0)

                    y_hat, _, syn_x_sim, syn_u_sim = run_model(x, hidden_init, syn_x_init, \
                        syn_u_init, network_weights, suppress_activity = suppress_activity)
                    acc, acc_non_match, acc_match = get_perf(y, y_hat, train_mask)
                    tuning_results['acc_neuronal_suppression'][r,k,k1,:] = np.array([acc, acc_non_match, acc_match])

                    syn_efficacy = syn_x_sim*syn_u_sim
                    for hidden_num in range(par['n_hidden']):
                        for t1 in range(num_time_steps):
                            weights = np.linalg.lstsq(test_dir[trial_ind,:,0], syn_efficacy[t1,trial_ind,hidden_num], rcond=None)
                            weights = np.reshape(weights[0],(3,1))
                            pred_err = syn_efficacy[t1,trial_ind,hidden_num] - np.dot(test_dir[trial_ind,:,0], weights).T
                            mse = np.mean(pred_err**2)
                            response_var = np.var(syn_efficacy[t1,trial_ind,hidden_num])
                            tuning_results['synaptic_pev_test_suppression'][r,k,k1, hidden_num,t1] = 1 - mse/(response_var+1e-9)
                            tuning_results['synaptic_pref_dir_test_suppression'][r,k,k1,hidden_num,t1] = np.arctan2(weights[2,0],weights[1,0])

    return tuning_results


def run_model(x, h_init_org, syn_x_init_org, syn_u_init_org, weights, suppress_activity = None):

    """ Simulate the RNN """

    # copying data to ensure nothing gets changed upstream
    h_init = copy.copy(h_init_org)
    syn_x_init = copy.copy(syn_x_init_org)
    syn_u_init = copy.copy(syn_u_init_org)

    network_weights = {k:v for k,v in weights.items()}

    if par['EI']:
        network_weights['w_rnn'] = par['EI_matrix'] @ np.maximum(0,network_weights['w_rnn'])
        network_weights['w_in'] = np.maximum(0,network_weights['w_in'])
        network_weights['w_out'] = np.maximum(0,network_weights['w_out'])

    h, syn_x, syn_u = \
        rnn_cell_loop(x, h_init, syn_x_init, syn_u_init, network_weights, suppress_activity)

    # Network output
    y = [h0 @ network_weights['w_out'] + weights['b_out'] for h0 in h]

    syn_x   = np.stack(syn_x)
    syn_u   = np.stack(syn_u)
    h       = np.stack(h)
    y       = np.stack(y)

    return y, h, syn_x, syn_u


def rnn_cell_loop(x_unstacked, h, syn_x, syn_u, weights, suppress_activity):

    h_hist = []
    syn_x_hist = []
    syn_u_hist = []

    # Loop through the neural inputs to the RNN
    for t, rnn_input in enumerate(x_unstacked):

        if suppress_activity is not None:
            h, syn_x, syn_u = rnn_cell(np.squeeze(rnn_input), h, syn_x, syn_u, weights, suppress_activity[t])
        else:
            h, syn_x, syn_u = rnn_cell(np.squeeze(rnn_input), h, syn_x, syn_u, weights, 1)
        print('h ', h.shape, ' syn_x ', syn_x.shape, ' syn_u ', syn_u.shape)

        h_hist.append(h)
        syn_x_hist.append(syn_x)
        syn_u_hist.append(syn_u)

    return h_hist, syn_x_hist, syn_u_hist

def rnn_cell(rnn_input, h, syn_x, syn_u, weights, suppress_activity):

    # Update the synaptic plasticity paramaters
    if par['synapse_config'] is not None:
        # implement both synaptic short term facilitation and depression
        syn_x_new = syn_x + (par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h)*par['dynamic_synapse']
        syn_u_new = syn_u + (par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h)*par['dynamic_synapse']
        syn_x_new = np.minimum(1, np.maximum(0, syn_x_new))
        syn_u_new = np.minimum(1, np.maximum(0, syn_u_new))
        h_post = syn_u_new*syn_x_new*h

    else:
        # no synaptic plasticity
        h_post = h


    # Update the hidden state
    h = np.maximum(0, h*(1-par['alpha_neuron'])
                   + par['alpha_neuron']*(rnn_input @ weights['w_in']
                   + h_post @ weights['w_rnn'] + weights['b_rnn'])
                   + np.random.normal(0, par['noise_rnn'],size = h.shape))

    h *= suppress_activity

    if par['synapse_config'] is None:
        syn_x_new = np.ones_like(h)
        syn_u_new = np.ones_like(h)

    return h, syn_x_new, syn_u_new


def get_perf(target, output, mask):

    """ Calculate task accuracy by comparing the actual network output to the desired output
        only examine time points when test stimulus is on, e.g. when y[:,:,0] = 0 """

    mask_full = np.float32(mask > 0)
    mask_test = mask_full*(target[:,:,0]==0)
    mask_non_match = mask_full*(target[:,:,1]==1)
    mask_match = mask_full*(target[:,:,2]==1)
    target_max = np.argmax(target, axis = 2)
    output_max = np.argmax(output, axis = 2)
    accuracy = np.sum(np.float32(target_max == output_max)*mask_test)/np.sum(mask_test)

    accuracy_non_match = np.sum(np.float32(target_max == output_max)*np.squeeze(mask_non_match))/np.sum(mask_non_match)
    accuracy_match = np.sum(np.float32(target_max == output_max)*np.squeeze(mask_match))/np.sum(mask_match)

    return accuracy, accuracy_non_match, accuracy_match
