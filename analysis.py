"""
Functions used to save model data and to perform analysis
"""

import numpy as np
from parameters import *
from sklearn import svm
import pickle

def analyze_model(trial_info, y_hat, h, syn_x, syn_u, model_performance, weights):

    """
    Converts neuroanl and synaptic values, stored in lists, into 3D arrays
    """
    syn_x = np.stack(syn_x, axis=1)
    syn_u = np.stack(syn_u, axis=1)
    h = np.stack(h, axis=1)

    """
    Decode the sample direction from neuronal activity and synaptic efficacies
    using support vector machhines
    """
    neuronal_decoding, synaptic_decoding = calculate_svms(h, syn_x, syn_u, trial_info['sample'], trial_info['rule'], trial_info['match'], num_reps = 20)

    """
    Determine end of delay time in order to calculate the neuronal and synaptic
    contributions to solving the task
    """
    accuracy, accuracy_neural_shuffled, accuracy_syn_shuffled = simulate_network(trial_info, h, syn_x, syn_u, weights)

    """
    Save the results
    """

    results = {
        'neuronal_decoding': neuronal_decoding,
        'synaptic_decoding': synaptic_decoding,
        'accuracy': accuracy,
        'accuracy_neural_shuffled': accuracy_neural_shuffled,
        'accuracy_syn_shuffled': accuracy_syn_shuffled,
        'model_performance': model_performance,
        'parameters': par,
        'weights': weights}

    save_fn = par['save_dir'] + par['save_fn']
    pickle.dump(results, open(save_fn, "wb" ) )
    print('Analysis results saved in ', save_fn)

def calculate_svms(h, syn_x, syn_u, sample, rule, match, num_reps = 20):

    """
    Calculates neuronal and synaptic decoding accuracies uisng support vector machines
    sample is the index of the sample motion direction for each trial_length
    rule is the rule index for each trial_length
    """

    lin_clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr', shrinking=False, tol=1e-4)

    neuronal_decoding = np.zeros((par['num_time_steps'], par['num_rules'], num_reps))
    synaptic_decoding = np.zeros((par['num_time_steps'], par['num_rules'], num_reps))

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
        num_motion_dirs = len(unique(sample))
        sample = np.floor(sample/(num_motion_dirs/2)*np.ones_like(sample))

    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        """
        For ABBA/ABCA trials, will only analyze trials for which the first n-1
        test stimuli, out of n, are non-matches
        """
        ind = np.where(np.sum(match[:,:-1],axis=1)==0)[0]
        sample = sample[ind]
        rule = rule[ind]
        h = h[:,:,ind]
        syn_efficacy = syn_efficacy[:,:,ind]

    # number of unique samples
    N = len(np.unique(sample))

    for r in range(par['num_rules']):
        ind = np.where((rule==r))[0]
        for t in range(par['num_time_steps']):
            neuronal_decoding[t,r,:] = calc_svm_equal_trials(lin_clf, h[:,t,ind].T, sample[ind], num_reps, N)
            synaptic_decoding[t,r,:] = calc_svm_equal_trials(lin_clf, syn_efficacy[:,t,ind].T, sample[ind], num_reps, N)

    return neuronal_decoding, synaptic_decoding


def calc_svm_equal_trials(lin_clf, y, conds, num_reps, num_conds):

    # normalize values between 0 and 1
    for i in range(y.shape[1]):
        m1 = y[:,i].min()
        m2 = y[:,i].max()
        y[:,i] -= m1
        if m2>m1:
            y[:,i] /=(m2-m1)

    """
    Want to ensure that all conditions have the same number of trials
    Will find the min number of trials per conditions, and remove trials above the min number
    """
    num_trials = np.zeros((num_conds))
    for i in range(num_conds):
        num_trials[i] = np.sum(conds==i)
    min_num_trials = int(np.min(num_trials))
    conds_equal = np.zeros((min_num_trials*num_conds))
    y_equal = np.zeros((min_num_trials*num_conds, y.shape[1]))
    for i in range(num_conds):
        ind = np.where(conds==i)[0]
        ind = ind[:min_num_trials]
        conds_equal[i*min_num_trials:(i+1)*min_num_trials] = i
        y_equal[i*min_num_trials:(i+1)*min_num_trials, :] = y[ind,:]

    train_pct = 0.75
    score = np.zeros((num_reps))
    for r in range(num_reps):
        q = np.random.permutation(len(conds_equal))
        i = np.int_(np.round(len(conds_equal)*train_pct))
        train_ind = q[:i]
        test_ind = q[i:]

        lin_clf.fit(y_equal[train_ind,:], conds_equal[train_ind])
        dec = lin_clf.predict(y_equal[test_ind,:])

        for i in range(len(test_ind)):
            if conds_equal[test_ind[i]]==dec[i]:
                score[r] += 1/len(test_ind)

    return score

def simulate_network(trial_info, h, syn_x, syn_u, weights, num_reps = 20):

    """
    Simulation will start from the start of the test period until the end of trial
    """
    if par['trial_type'] == 'dualDMS':
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+par['test_time'])//par['dt']
    else:
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']

    accuracy = np.zeros((par['num_rules'], num_reps))
    accuracy_neural_shuffled = np.zeros((par['num_rules'], num_reps))
    accuracy_syn_shuffled = np.zeros((par['num_rules'], num_reps))

    _, trial_length, batch_train_size = h.shape
    test_length = trial_length - test_onset


    for r in range(par['num_rules']):
        ind = np.where(trial_info['rule']==r)[0]
        train_mask = trial_info['train_mask'][test_onset:,ind]
        x = np.split(trial_info['neural_input'][:,test_onset:,ind],test_length,axis=1)
        y = trial_info['desired_output'][:,test_onset:,ind]

        for n in range(num_reps):

            """
            Calculating behavioral accuracy without shuffling
            """
            hidden_init = h[:,test_onset-1,ind]
            syn_x_init = syn_x[:,test_onset-1,ind]
            syn_u_init = syn_u[:,test_onset-1,ind]
            y_hat = run_model(x, y, hidden_init, syn_x_init, syn_u_init, weights)
            accuracy[r,n] = get_perf(y, y_hat, train_mask)

            """
            Keep the synaptic values fixed, permute the neural activity
            """
            ind_shuffle = np.random.permutation(batch_train_size)

            hidden_init = hidden_init[:,ind_shuffle]
            y_hat = run_model(x, y, hidden_init, syn_x_init, syn_u_init, weights)
            accuracy_neural_shuffled[r,n] = get_perf(y, y_hat, train_mask)

            """
            Keep the hidden values fixed, permute synaptic values
            """
            hidden_init = h[:,test_onset-1,ind]
            syn_x_init = syn_x_init[:,ind_shuffle]
            syn_u_init = syn_u_init[:,ind_shuffle]
            y_hat = run_model(x, y, hidden_init, syn_x_init, syn_u_init, weights)
            accuracy_syn_shuffled[r,n] = get_perf(y, y_hat, train_mask)

    return accuracy, accuracy_neural_shuffled, accuracy_syn_shuffled


def run_model(x, y, hidden_init, syn_x_init, syn_u_init, weights):

    """
    Run the reccurent network
    History of hidden state activity stored in self.hidden_state_hist
    """
    hidden_state_hist = rnn_cell_loop(x, hidden_init, syn_x_init, syn_u_init, weights)

    """
    Network output
    Only use excitatory projections from the RNN to the output layer
    """
    y_hat = [np.dot(np.maximum(0,weights['w_out']), h) + weights['b_out'] for h in hidden_state_hist]

    return y_hat


def rnn_cell_loop(x_unstacked, h, syn_x, syn_u, weights):

    hidden_state_hist = []

    """
    Loop through the neural inputs to the RNN, indexed in time
    """
    for rnn_input in x_unstacked:
        h, syn_x, syn_u = rnn_cell(np.squeeze(rnn_input), h, syn_x, syn_u, weights)
        hidden_state_hist.append(h)

    return hidden_state_hist

def rnn_cell(rnn_input, h, syn_x, syn_u, weights):

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
                   + np.random.normal(0, par['noise_sd'],size=(par['n_hidden'], par['batch_train_size'])))

    return h, syn_x, syn_u


def get_perf(y, y_hat, mask):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when y[0,:,:] is not 0
    """

    y_hat = np.stack(y_hat, axis=1)
    mask *= y[0,:,:]==0
    y = np.argmax(y, axis = 0)
    y_hat = np.argmax(y_hat, axis = 0)

    return np.sum(np.float32(y == y_hat)*np.squeeze(mask))/np.sum(mask)
