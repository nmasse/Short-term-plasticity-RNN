"""
Functions used to save model data and to perform analysis
"""

import numpy as np
from parameters import *
from sklearn import svm
import time
import pickle

def analyze_model(trial_info, y_hat, h, syn_x, syn_u, model_performance, weights,simulation = True, \
        tuning = True, decoding = True, load_previous_file = False, save_raw_data = False):

    """
    Converts neuronal and synaptic values, stored in lists, into 3D arrays
    Creating new variable since h, syn_x, and syn_u are class members of model.py,
    and will get mofiied by functions within analysis.py
    """
    syn_x_stacked = np.stack(syn_x, axis=1)
    syn_u_stacked = np.stack(syn_u, axis=1)
    h_stacked = np.stack(h, axis=1)
    trial_time = np.arange(0,h_stacked.shape[1]*par['dt'], par['dt'])

    save_fn = par['save_dir'] + par['save_fn']
    if load_previous_file:
        results = pickle.load(open(save_fn, 'rb'))
    else:
        results = {
            'model_performance': model_performance,
            'parameters': par,
            'weights': weights,
            'trial_time': trial_time}

    if save_raw_data:
        results['h'] = h
        results['syn_x'] = np.array(syn_x)
        results['syn_u'] = np.array(syn_u)
        results['y_hat'] = np.array(y_hat)
        results['trial_info'] = trial_info


    """
    Calculate the neuronal and synaptic contributions towards solving the task
    """
    if simulation:
        print('simulating network...')
        simulation_results = simulate_network(trial_info, h_stacked, syn_x_stacked, \
            syn_u_stacked, weights, num_reps = par['simulation_reps'])
        for key, val in simulation_results.items():
            results[key] = val


    """
    Calculate neuronal and synaptic sample motion tuning
    """
    if tuning:
        print('calculate tuning...')
        tuning_results = calculate_tuning(h_stacked, syn_x_stacked, syn_u_stacked, \
            trial_info, trial_time, weights, calculate_test = par['decode_test'])
        for key, val in tuning_results.items():
            results[key] = val

    """
    Decode the sample direction from neuronal activity and synaptic efficacies
    using support vector machines
    """
    if decoding:
        print('decoding activity...')
        decoding_results = calculate_svms(h_stacked, syn_x_stacked, syn_u_stacked, trial_info, trial_time, \
            num_reps = par['decoding_reps'], decode_test = par['decode_test'], decode_rule = par['decode_rule'], \
            decode_sample_vs_test = par['decode_sample_vs_test'])
        for key, val in decoding_results.items():
            results[key] = val

    pickle.dump(results, open(save_fn, 'wb') )
    print('Analysis results saved in ', save_fn)

def calculate_svms(h, syn_x, syn_u, trial_info, trial_time, num_reps = 20, \
    decode_test = False, decode_rule = False, decode_sample_vs_test = False):

    """
    Calculates neuronal and synaptic decoding accuracies uisng support vector machines
    sample is the index of the sample motion direction for each trial_length
    rule is the rule index for each trial_length
    """

    lin_clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr', shrinking=False, tol=1e-4)

    num_time_steps = len(trial_time)
    decoding_results = {}

    """
    The synaptic efficacy is the product of syn_x and syn_u, will decode sample
    direction from this value
    """
    syn_efficacy = syn_x*syn_u

    if par['trial_type'] == 'DMC':
        """
        Will also calculate the category decoding accuracies, assuming the first half of
        the sample direction belong to category 1, and the second half belong to category 2
        """
        num_motion_dirs = len(np.unique(trial_info['sample']))
        sample = np.floor(trial_info['sample']/(num_motion_dirs/2)*np.ones_like(trial_info['sample']))
        test = np.floor(trial_info['test']/(num_motion_dirs/2)*np.ones_like(trial_info['sample']))
        rule = trial_info['rule']
    elif par['trial_type'] == 'dualDMS':
        sample = trial_info['sample']
        rule = trial_info['rule'][:,0] + 2*trial_info['rule'][:,1]
        par['num_rules'] = 4
    else:
        sample = trial_info['sample']
        rule = trial_info['rule']

    if trial_info['test'].ndim == 2:
        test = trial_info['test'][:,0]
    else:
        test = np.array(trial_info['test'])


    # number of unique samples
    N = len(np.unique(sample))
    print('sample decoding...num_reps = ', num_reps)
    decoding_results['neuronal_sample_decoding'], decoding_results['synaptic_sample_decoding'] = \
        svm_wraper(lin_clf, h, syn_efficacy, sample, rule, num_reps, N, trial_time)

    if decode_sample_vs_test:
        print('sample vs. test decoding...')
        decoding_results['neuronal_sample_test_decoding'], decoding_results['synaptic_sample_test_decoding'] = \
            svm_wraper_sample_vs_test(lin_clf, h, syn_efficacy, trial_info['sample'], trial_info['test'], num_reps, N, trial_time)

    if decode_test:
        print('test decoding...')
        decoding_results['neuronal_test_decoding'], decoding_results['synaptic_test_decoding'] = \
            svm_wraper(lin_clf, h, syn_efficacy, test, rule, num_reps, N, trial_time)

    if decode_rule:
        print('rule decoding...')
        decoding_results['neuronal_rule_decoding'], decoding_results['synaptic_rule_decoding'] = \
            svm_wraper(lin_clf, h, syn_efficacy, trial_info['rule'], np.zeros_like(sample), num_reps, 2, trial_time)

    return decoding_results

def svm_wraper_sample_vs_test(lin_clf, h, syn_eff, sample, test, num_reps, num_conds, trial_time):

    _, num_time_steps, num_trials = h.shape
    trials_per_cond = 25
    score_h = np.zeros((num_reps, num_time_steps), dtype = np.float32)
    score_syn_eff = np.zeros((num_reps, num_time_steps), dtype = np.float32)

    trial_ind = np.zeros((num_conds*trials_per_cond), dtype = np.uint16)

    cond_ind = []
    for c in range(num_conds):
        cond_ind.append(np.where(sample == c)[0])
        if len(cond_ind[c]) < 4:
            print('Not enough trials for this condition!')
            print('Setting cond_ind to [0,1,2,3]')
            cond_ind[c] = [0,1,2,3]

    for rep in range(num_reps):
        for c in range(num_conds):
            u = range(c*trials_per_cond, (c+1)*trials_per_cond)
            q = np.random.randint(len(cond_ind[c]), size = trials_per_cond)
            trial_ind[u] =  cond_ind[c][q]

        for t in range(num_time_steps):
            if trial_time[t] <= par['dead_time']:
                # no need to analyze activity during dead time
                continue

            score_h[rep,t] = calc_svm(lin_clf, h[:,t,:].T, sample, test, trial_ind, trial_ind)
            score_syn_eff[rep,t] = calc_svm(lin_clf, syn_eff[:,t,:].T, sample, test, trial_ind, trial_ind)


    return score_h, score_syn_eff



def svm_wraper(lin_clf, h, syn_eff, conds, rule, num_reps, num_conds, trial_time):

    """
    Wraper function used to decode sample/test or rule information
    from hidden activity (h) and synaptic efficacies (syn_eff)
    """
    train_pct = 0.75
    trials_per_cond = 25
    _, num_time_steps, num_trials = h.shape
    num_rules = len(np.unique(rule))

    score_h = np.zeros((num_rules, par['num_receptive_fields'], num_reps, num_time_steps), dtype = np.float32)
    score_syn_eff = np.zeros((num_rules, par['num_receptive_fields'], num_reps, num_time_steps), dtype = np.float32)

    equal_train_ind = np.zeros((num_conds*trials_per_cond), dtype = np.uint16)
    equal_test_ind = np.zeros((num_conds*trials_per_cond), dtype = np.uint16)

    for r in range(num_rules):
        ind_rule = np.where(rule==r)[0]
        for n in range(par['num_receptive_fields']):
            if par['trial_type'] == 'dualDMS':
                current_conds = conds[:,n]
            else:
                current_conds = np.array(conds)

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

                for t in range(num_time_steps):
                    if trial_time[t] <= par['dead_time']:
                        # no need to analyze activity during dead time
                        continue

                    score_h[r,n,rep,t] = calc_svm(lin_clf, h[:,t,:].T, current_conds, current_conds, equal_train_ind, equal_test_ind)
                    score_syn_eff[r,n,rep,t] = calc_svm(lin_clf, syn_eff[:,t,:].T, current_conds, current_conds, equal_train_ind, equal_test_ind)


    return score_h, score_syn_eff


def calc_svm(lin_clf, y, train_conds, test_conds, train_ind, test_ind):

    n_test_inds = len(test_ind)
    # normalize values between 0 and 1
    for i in range(y.shape[1]):
        m1 = y[train_ind,i].min()
        m2 = y[train_ind,i].max()
        y[:,i] -= m1
        if m2>m1:
            if par['svm_normalize']:
                y[:,i] /=(m2-m1)

    lin_clf.fit(y[train_ind,:], train_conds[train_ind])
    dec = lin_clf.predict(y[test_ind,:])
    score = 0
    for i in range(n_test_inds):
        if test_conds[test_ind[i]]==dec[i]:
            score += 1/n_test_inds

    return score


def lesion_weights(trial_info, h, syn_x, syn_u, weights):

    N = weights['w_rnn'].shape[0]
    num_reps = 5
    accuracy_rnn_start = np.ones((N,N), dtype=np.float32)
    accuracy_rnn_test = np.ones((N,N), dtype=np.float32)
    accuracy_out = np.ones((3,N), dtype=np.float32)
    trial_time = np.arange(0,h.shape[1]*par['dt'], par['dt'])

    neuronal_decoding = np.zeros((N,N,par['num_rules'], num_reps, len(trial_time)))
    neuronal_pref_dir = np.zeros((N,N,par['n_hidden'],  par['num_rules'],  len(trial_time)))
    neuronal_pev = np.zeros((N,N,par['n_hidden'],  par['num_rules'],  len(trial_time)))


    # network inputs/outputs
    _, trial_length, batch_train_size = h.shape
    x = np.split(trial_info['neural_input'],trial_length,axis=1)
    y = trial_info['desired_output']
    train_mask = trial_info['train_mask']


    test_onset = 1
    hidden_init = h[:,test_onset-1,:]
    syn_x_init = syn_x[:,test_onset-1,:]
    syn_u_init = syn_u[:,test_onset-1,:]

    test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
    hidden_init_test = h[:,test_onset-1,:]
    syn_x_init_test = syn_x[:,test_onset-1,:]
    syn_u_init_test = syn_u[:,test_onset-1,:]

    x_test = np.split(trial_info['neural_input'][:,test_onset:,:],trial_length-test_onset,axis=1)
    y_test = trial_info['desired_output'][:,test_onset:,:]
    train_mask_test = trial_info['train_mask'][test_onset:,:]


    # create new dict of weights
    weights_new = {}
    for k,v in weights.items():
        weights_new[k] = v

    for n1 in range(3):
        for n2 in range(N):

            if weights['w_out'][n1,n2] <= 0:
                continue

            # lesion weights
            q = np.ones((3,N))
            q[n1,n2] = 0
            weights_new['w_out'] = weights['w_out']*q

            # simulate network
            y_hat, hidden_state_hist = run_model(x_test, y_test, hidden_init_test, syn_x_init_test, syn_u_init_test, weights_new)
            accuracy_out[n1,n2] = get_perf(y_test, y_hat, train_mask_test)

    for n1 in range(N):
        for n2 in range(N):

            if weights['w_rnn'][n1,n2] <= 0:
                continue

            # lesion weights
            q = np.ones((N,N))
            q[n1,n2] = 0
            weights_new['w_rnn'] = weights['w_rnn']*q

            # simulate network
            y_hat, hidden_state_hist = run_model(x, y, hidden_init, syn_x, syn_u, weights_new)
            accuracy_rnn_start[n1,n2] = get_perf(y, y_hat, train_mask)

            y_hat, hidden_state_hist = run_model(x_test, y_test, hidden_init_test, syn_x_init_test, syn_u_init_test, weights_new)
            accuracy_rnn_test[n1,n2] = get_perf(y_test, y_hat, train_mask_test)

            if accuracy_rnn_start[n1,n2] < -1:

                h_stacked = np.stack(hidden_state_hist, axis=1)

                neuronal_decoding[n1,n2,:,:,:], _ = calculate_svms(h_stacked, syn_x, syn_u, trial_info['sample'], \
                    trial_info['rule'], trial_info['match'], trial_time, num_reps = num_reps)

                neuronal_pref_dir[n1,n2,:,:], neuronal_pev[n1,n2,:,:], _, _ = calculate_sample_tuning(h_stacked, \
                    syn_x, syn_u, trial_info['sample'], trial_info['rule'], trial_info['match'], trial_time)

            """
            y_hat_test = run_model(x_test, y_test, hidden_init_test, syn_x_init_test, syn_u_init_test, weights_new)
            accuracy_test[n1,n2] = get_perf(y_test, y_hat_test, train_mask_test)
            """

    return accuracy_rnn_start, accuracy_rnn_test, accuracy_out




def simulate_network(trial_info, h, syn_x, syn_u, network_weights, num_reps = 20):

    """
    Simulation will start from the start of the test period until the end of trial
    """
    if par['trial_type'] == 'dualDMS':
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+par['test_time'])//par['dt']
    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA' :
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+5*par['ABBA_delay'])//par['dt']
    else:
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']

    simulation_results = {
        'accuracy'                  : np.zeros((par['num_rules'], num_reps)),
        'accuracy_neural_shuffled'  : np.zeros((par['num_rules'], num_reps)),
        'accuracy_syn_shuffled'     : np.zeros((par['num_rules'], num_reps))}

    _, trial_length, batch_train_size = h.shape
    test_length = trial_length - test_onset

    for r in range(par['num_rules']):
        # For ABBA/ABCA trials, will only analyze trials for which the first n-1
        # test stimuli, out of n, are non-matches
        """
        if par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
            trial_ind = np.where((np.sum(trial_info['match'][:,:-1],axis=1)==0)*(trial_info['rule']==r))[0]
        else:
        """
        trial_ind = np.where(trial_info['rule']==r)[0]
        train_mask = trial_info['train_mask'][test_onset:,trial_ind]
        x = np.split(trial_info['neural_input'][:,test_onset:,trial_ind],test_length,axis=1)
        y = trial_info['desired_output'][:,test_onset:,trial_ind]

        for n in range(num_reps):

            """
            Calculating behavioral accuracy without shuffling
            """
            hidden_init = h[:,test_onset-1,trial_ind]
            syn_x_init = syn_x[:,test_onset-1,trial_ind]
            syn_u_init = syn_u[:,test_onset-1,trial_ind]
            y_hat, _, _, _ = run_model(x, y, hidden_init, syn_x_init, syn_u_init, network_weights)
            simulation_results['accuracy'][r,n] ,_ ,_ = get_perf(y, y_hat, train_mask)

            """
            Keep the synaptic values fixed, permute the neural activity
            """
            ind_shuffle = np.random.permutation(len(trial_ind))

            hidden_init = hidden_init[:,ind_shuffle]
            y_hat, _, _, _ = run_model(x, y, hidden_init, syn_x_init, syn_u_init, network_weights)
            simulation_results['accuracy_neural_shuffled'][r,n] ,_ ,_ = get_perf(y, y_hat, train_mask)

            """
            Keep the hidden values fixed, permute synaptic values
            """
            hidden_init = h[:,test_onset-1,trial_ind]
            syn_x_init = syn_x_init[:,ind_shuffle]
            syn_u_init = syn_u_init[:,ind_shuffle]
            y_hat, _, _, _ = run_model(x, y, hidden_init, syn_x_init, syn_u_init, network_weights)
            simulation_results['accuracy_syn_shuffled'][r,n] ,_ ,_ = get_perf(y, y_hat, train_mask)

        if par['trial_type'] == 'ABCA' or  par['trial_type'] == 'ABBA':

            simulation_results['ABBA_test2_acc_shuffled'] = np.zeros((7,18,3))
            _, trial_length, batch_train_size = h.shape
            trial_ind = range(batch_train_size)

            if par['suppress_analysis']:

                trial_onset = (par['dead_time'])//par['dt']
                test_onset = (par['fix_time']+par['sample_time']+par['ABBA_delay'])//par['dt']
                test_length = trial_length - trial_onset
                x = np.split(trial_info['neural_input'][:,trial_onset:,trial_ind],test_length,axis=1)
                y = trial_info['desired_output'][:,trial_onset:,trial_ind]
                train_mask = trial_info['train_mask'][trial_onset:,trial_ind]
                train_mask[test_onset + par['ABBA_delay']//par['dt']:, :] = 0
                syn_x_init = syn_x[:,trial_onset-1,trial_ind]
                syn_u_init = syn_u[:,trial_onset-1,trial_ind]
                hidden_init = h[:,trial_onset-1,trial_ind]

                y_hat, _, _, _ = run_model(x, y, hidden_init, syn_x_init, syn_u_init, network_weights)
                acc, acc_non_match, acc_match = get_perf(y, y_hat, train_mask)
                simulation_results['ABBA_test2_acc'] = np.array([acc, acc_non_match, acc_match])

                time_range = []
                for k in range(9):
                    time_range.append(range(test_onset-k*5, test_onset))
                for k in range(9):
                    time_range.append(range(test_onset-k*5+20, test_onset+20))

                for k in range(18):
                    for k1 in range(7):

                        suppress_activity = np.ones((par['n_hidden'], test_length))
                        if k1 == 0:
                            for m2 in time_range[k]:
                                suppress_activity[:,m2] = 0
                        elif k1 == 1:
                            for m1 in range(par['num_exc_units']):
                                for m2 in time_range[k]:
                                    suppress_activity[m1,m2] = 0

                        elif k1 == 2:
                            for m1 in range(par['num_exc_units'], par['n_hidden']):
                                for m2 in time_range[k]:
                                    suppress_activity[m1,m2] = 0

                        elif k1 == 3:
                            for m1 in range(0, par['num_exc_units'], 2):
                                for m2 in time_range[k]:
                                    suppress_activity[m1,m2] = 0

                        elif k1 == 4:
                            for m1 in range(1, par['num_exc_units'], 2):
                                for m2 in time_range[k]:
                                    suppress_activity[m1,m2] = 0

                        elif k1 == 5:
                            for m1 in range(par['num_exc_units'], par['n_hidden'], 2):
                                for m2 in time_range[k]:
                                    suppress_activity[m1,m2] = 0

                        elif k1 == 6:
                            for m1 in range(par['num_exc_units']+1, par['n_hidden'], 2):
                                for m2 in time_range[k]:
                                    suppress_activity[m1,m2] = 0

                        suppress_activity = np.split(suppress_activity, test_length, axis=1)

                        y_hat, _, syn_x_sim, syn_u_sim = run_model(x, y, hidden_init, syn_x_init, \
                            syn_u_init, network_weights, suppress_activity = suppress_activity)
                        acc, acc_non_match, acc_match = get_perf(y, y_hat, train_mask)
                        simulation_results['ABBA_test2_acc_shuffled'][k1,k,:] = np.array([acc, acc_non_match, acc_match])


    return simulation_results

def calculate_tuning(h, syn_x, syn_u, trial_info, trial_time, network_weights, calculate_test = False):

    """
    Calculates neuronal and synaptic sample motion direction tuning
    """
    num_time_steps = len(trial_time)
    tuning_results = {
        'neuronal_pref_dir'     : np.zeros((par['n_hidden'],  par['num_rules'], num_time_steps), dtype=np.float32),
        'synaptic_pref_dir'     : np.zeros((par['n_hidden'],  par['num_rules'], num_time_steps), dtype=np.float32),
        'neuronal_pev'          : np.zeros((par['n_hidden'],  par['num_rules'], num_time_steps), dtype=np.float32),
        'synaptic_pev'          : np.zeros((par['n_hidden'],  par['num_rules'], num_time_steps), dtype=np.float32),
        'neuronal_pev_test'     : np.zeros((par['n_hidden'],  par['num_rules'], num_time_steps), dtype=np.float32),
        'synaptic_pev_test'     : np.zeros((par['n_hidden'],  par['num_rules'], num_time_steps), dtype=np.float32),
        'neuronal_pref_dir_test': np.zeros((par['n_hidden'],  par['num_rules'], num_time_steps), dtype=np.float32),
        'synaptic_pref_dir_test': np.zeros((par['n_hidden'],  par['num_rules'], num_time_steps), dtype=np.float32),
        'neuronal_sample_tuning': np.zeros((par['n_hidden'],  par['num_rules'], par['num_motion_dirs'], num_time_steps), dtype=np.float32),
        'synaptic_sample_tuning': np.zeros((par['n_hidden'],  par['num_rules'], par['num_motion_dirs'], num_time_steps), dtype=np.float32),
        'synaptic_pev_test_shuffled' : np.zeros((7,18,par['n_hidden'],  par['num_rules'], num_time_steps), dtype=np.float32),
        'synaptic_pref_dir_test_shuffled' : np.zeros((7,18,par['n_hidden'],  par['num_rules'], num_time_steps), dtype=np.float32)}



    """
    The synaptic efficacy is the product of syn_x and syn_u, will decode sample
    direction from this value
    """
    syn_efficacy = syn_x*syn_u

    sample = np.array(trial_info['sample'])
    test = np.array(trial_info['test'])
    rule = np.array(trial_info['rule'])
    if sample.ndim == 2:
        sample = sample[:, 0]
    if test.ndim == 2:
        test = test[:, 0]
    if test.ndim == 3:
        test = test[:, 0, 0]



    # number of unique samples
    N = len(np.unique(sample))

    sample_dir = np.ones((len(sample), 3))
    sample_dir[:,1] = np.cos(2*np.pi*sample/N)
    sample_dir[:,2] = np.sin(2*np.pi*sample/N)

    if calculate_test:
        test_dir = np.ones((len(sample), 3))
        test_dir[:,1] = np.cos(2*np.pi*test/N)
        test_dir[:,2] = np.sin(2*np.pi*test/N)

    for r in range(par['num_rules']):
        ind = np.where((rule==r))[0]
        for n in range(par['n_hidden']):
            for t in range(num_time_steps):
                if trial_time[t] <= par['dead_time']:
                    # no need to analyze activity during dead time
                    continue

                # Mean sample response
                for md in range(par['num_motion_dirs']):
                    ind_motion_dir = np.where((rule==r)*(sample==md))[0]
                    tuning_results['neuronal_sample_tuning'][n,r,md,t] = np.mean(h[n,t,ind_motion_dir])
                    tuning_results['synaptic_sample_tuning'][n,r,md,t] = np.mean(syn_efficacy[n,t,ind_motion_dir])

                # Neuronal sample tuning
                weights = np.linalg.lstsq(sample_dir[ind,:], h[n,t,ind])
                weights = np.reshape(weights[0],(3,1))
                pred_err = h[n,t,ind] - np.dot(sample_dir[ind,:], weights).T
                mse = np.mean(pred_err**2)
                response_var = np.var(h[n,t,ind])
                tuning_results['neuronal_pev'][n,r,t] = 1 - mse/(response_var+1e-9)
                tuning_results['neuronal_pref_dir'][n,r,t] = np.arctan2(weights[2,0],weights[1,0])

                if calculate_test:
                    weights = np.linalg.lstsq(test_dir[ind,:], h[n,t,ind])
                    weights = np.reshape(weights[0],(3,1))
                    pred_err = h[n,t,ind] - np.dot(test_dir[ind,:], weights).T
                    mse = np.mean(pred_err**2)
                    response_var = np.var(h[n,t,ind])
                    tuning_results['neuronal_pev_test'][n,r,t] = 1 - mse/(response_var+1e-9)
                    tuning_results['neuronal_pref_dir_test'][n,r,t] = np.arctan2(weights[2,0],weights[1,0])


                # Synaptic sample tuning
                weights = np.linalg.lstsq(sample_dir[ind,:], syn_efficacy[n,t,ind])
                weights = np.reshape(weights[0],(3,1))
                pred_err = syn_efficacy[n,t,ind] - np.dot(sample_dir[ind,:], weights).T
                mse = np.mean(pred_err**2)
                response_var = np.var(syn_efficacy[n,t,ind])
                tuning_results['synaptic_pev'][n,r,t] = 1 - mse/(response_var+1e-9)
                tuning_results['synaptic_pref_dir'][n,r,t] = np.arctan2(weights[2,0],weights[1,0])

                if calculate_test:
                    weights = np.linalg.lstsq(test_dir[ind,:], syn_efficacy[n,t,ind])
                    weights = np.reshape(weights[0],(3,1))
                    pred_err = syn_efficacy[n,t,ind] - np.dot(test_dir[ind,:], weights).T
                    mse = np.mean(pred_err**2)
                    response_var = np.var(syn_efficacy[n,t,ind])
                    tuning_results['synaptic_pev_test'][n,r,t] = 1 - mse/(response_var+1e-9)
                    tuning_results['synaptic_pref_dir_test'][n,r,t] = np.arctan2(weights[2,0],weights[1,0])

    if par['trial_type'] == 'ABCA' or  par['trial_type'] == 'ABBA':


        trial_ind = np.where((rule==r))[0]
        _, trial_length, batch_train_size = h.shape


        if par['suppress_analysis']:

            trial_onset = (par['dead_time'])//par['dt']
            test_onset = (par['fix_time']+par['sample_time']+par['ABBA_delay'])//par['dt']
            test_length = trial_length - trial_onset
            x = np.split(trial_info['neural_input'][:,trial_onset:,trial_ind],test_length,axis=1)
            y = trial_info['desired_output'][:,trial_onset:,trial_ind]
            train_mask = trial_info['train_mask'][trial_onset:,trial_ind]
            train_mask[test_onset + par['ABBA_delay']//par['dt']:, :] = 0
            syn_x_init = syn_x[:,trial_onset-1,trial_ind]
            syn_u_init = syn_u[:,trial_onset-1,trial_ind]
            hidden_init = h[:,trial_onset-1,trial_ind]

            time_range = []
            for k in range(9):
                time_range.append(range(test_onset-k*5, test_onset))
            for k in range(9):
                time_range.append(range(test_onset-k*5+20, test_onset+20))

            for k in range(18):
                for k1 in range(7):

                    suppress_activity = np.ones((par['n_hidden'], test_length))
                    if k1 == 0:
                        for m2 in time_range[k]:
                            suppress_activity[:,m2] = 0
                    elif k1 == 1:
                        for m1 in range(par['num_exc_units']):
                            for m2 in time_range[k]:
                                suppress_activity[m1,m2] = 0

                    elif k1 == 2:
                        for m1 in range(par['num_exc_units'], par['n_hidden']):
                            for m2 in time_range[k]:
                                suppress_activity[m1,m2] = 0

                    elif k1 == 3:
                        for m1 in range(0, par['num_exc_units'], 2):
                            for m2 in time_range[k]:
                                suppress_activity[m1,m2] = 0

                    elif k1 == 4:
                        for m1 in range(1, par['num_exc_units'], 2):
                            for m2 in time_range[k]:
                                suppress_activity[m1,m2] = 0

                    elif k1 == 5:
                        for m1 in range(par['num_exc_units'], par['n_hidden'], 2):
                            for m2 in time_range[k]:
                                suppress_activity[m1,m2] = 0

                    elif k1 == 6:
                        for m1 in range(par['num_exc_units']+1, par['n_hidden'], 2):
                            for m2 in time_range[k]:
                                suppress_activity[m1,m2] = 0

                    suppress_activity = np.split(suppress_activity, test_length, axis=1)

                    y_hat, _, syn_x_sim, syn_u_sim = run_model(x, y, hidden_init, syn_x_init, \
                        syn_u_init, network_weights, suppress_activity = suppress_activity)

                    syn_efficacy = syn_x_sim*syn_u_sim

                    for n in range(par['n_hidden']):
                        for t in range(num_time_steps-trial_onset):

                            weights = np.linalg.lstsq(test_dir[trial_ind,:], syn_efficacy[n,t,trial_ind])
                            weights = np.reshape(weights[0],(3,1))
                            pred_err = syn_efficacy[n,t,trial_ind] - np.dot(test_dir[trial_ind,:], weights).T
                            mse = np.mean(pred_err**2)
                            response_var = np.var(syn_efficacy[n,t,ind])
                            tuning_results['synaptic_pev_test_shuffled'][k1,k,n,r,t+trial_onset] = 1 - mse/(response_var+1e-9)
                            tuning_results['synaptic_pref_dir_test_shuffled'][k1,k,n,r,t+trial_onset] = np.arctan2(weights[2,0],weights[1,0])


    return tuning_results


def run_model(x, y, hidden_init, syn_x_init, syn_u_init, weights, suppress_activity = None):

    """
    Run the reccurent network
    History of hidden state activity stored in self.hidden_state_hist
    """
    hidden_state_hist, syn_x_hist, syn_u_hist = \
        rnn_cell_loop(x, hidden_init, syn_x_init, syn_u_init, weights, suppress_activity)

    """
    Network output
    Only use excitatory projections from the RNN to the output layer
    """
    y_hat = [np.dot(np.maximum(0,weights['w_out']), h) + weights['b_out'] for h in hidden_state_hist]

    syn_x_hist = np.stack(syn_x_hist, axis=1)
    syn_u_hist = np.stack(syn_u_hist, axis=1)
    hidden_state_hist = np.stack(hidden_state_hist, axis=1)

    return y_hat, hidden_state_hist, syn_x_hist, syn_u_hist


def rnn_cell_loop(x_unstacked, h, syn_x, syn_u, weights, suppress_activity):

    hidden_state_hist = []
    syn_x_hist = []
    syn_u_hist = []

    """
    Loop through the neural inputs to the RNN, indexed in time
    """

    for t, rnn_input in enumerate(x_unstacked):
        #print(t)
        if suppress_activity is not None:
            #print('len sp', len(suppress_activity))
            h, syn_x, syn_u = rnn_cell(np.squeeze(rnn_input), h, syn_x, syn_u, weights, suppress_activity[t])
        else:
            h, syn_x, syn_u = rnn_cell(np.squeeze(rnn_input), h, syn_x, syn_u, weights, 1)
        hidden_state_hist.append(h)
        syn_x_hist.append(syn_x)
        syn_u_hist.append(syn_u)

    return hidden_state_hist, syn_x_hist, syn_u_hist

def rnn_cell(rnn_input, h, syn_x, syn_u, weights, suppress_activity):

    if par['EI']:
        # ensure excitatory neurons only have postive outgoing weights,
        # and inhibitory neurons have negative outgoing weights
        W_rnn_effective = np.dot(np.maximum(0,weights['w_rnn']), par['EI_matrix'])
    else:
        W_rnn_effective = weights['w_rnn']


    """
    Update the synaptic plasticity paramaters
    """
    if par['synapse_config'] == 'std_stf':
        # implement both synaptic short term facilitation and depression
        syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
        syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
        syn_x = np.minimum(1, np.maximum(0, syn_x))
        syn_u = np.minimum(1, np.maximum(0, syn_u))
        h_post = syn_u*syn_x*h

    elif par['synapse_config'] == 'std':
        # implement synaptic short term derpression, but no facilitation
        # we assume that syn_u remains constant at 1
        syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_x*h
        syn_x = np.minimum(1, np.maximum(0, syn_x))
        h_post = syn_x*h

    elif par['synapse_config'] == 'stf':
        # implement synaptic short term facilitation, but no depression
        # we assume that syn_x remains constant at 1
        syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
        syn_u = np.minimum(1, np.maximum(0, syn_u))
        h_post = syn_u*h

    else:
        # no synaptic plasticity
        h_post = h

    """
    Update the hidden state
    All needed rectification has already occured
    """

    h = np.maximum(0, h*(1-par['alpha_neuron'])
                   + par['alpha_neuron']*(np.dot(np.maximum(0,weights['w_in']), np.maximum(0, rnn_input))
                   + np.dot(W_rnn_effective, h_post) + weights['b_rnn'])
                   + np.random.normal(0, par['noise_rnn'],size=(par['n_hidden'], h.shape[1])))

    h *= suppress_activity

    return h, syn_x, syn_u


def get_perf(y, y_hat, mask):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when y[0,:,:] is not 0
    y is the desired output
    y_hat is the actual output
    """
    y_hat = np.stack(y_hat, axis=1)
    mask *= y[0,:,:]==0
    mask_non_match = mask*(y[1,:,:]==1)
    mask_match = mask*(y[2,:,:]==1)
    y = np.argmax(y, axis = 0)
    y_hat = np.argmax(y_hat, axis = 0)
    accuracy = np.sum(np.float32(y == y_hat)*np.squeeze(mask))/np.sum(mask)

    accuracy_non_match = np.sum(np.float32(y == y_hat)*np.squeeze(mask_non_match))/np.sum(mask_non_match)
    accuracy_match = np.sum(np.float32(y == y_hat)*np.squeeze(mask_match))/np.sum(mask_match)

    return accuracy, accuracy_non_match, accuracy_match
