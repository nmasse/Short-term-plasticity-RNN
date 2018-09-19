"""
Functions used to save model data and to perform analysis
"""

import numpy as np
from parameters import *
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import time
import pickle
import stimulus
import os
import copy
from sklearn.decomposition import PCA, FactorAnalysis
from dPCA import dPCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def run_multiple():

    task_list = [ 'DMS','DMRS90','DMRS45','DMRS180','ABCA','ABBA','DMS+DMRS','dualDMS','location_DMS']
    task_list = [ 'location_DMS']
    #task_list = ['ABCA','ABBA','DMS+DMRS','dualDMS','location_DMS']
    task_list =['DMS+DMRS']

    update_params = {'decode_stability':False, 'decoding_reps':100, 'simulation_reps': 100, 'analyze_tuning':True,'calculate_resp_matrix':False,\
                    'suppress_analysis':False, 'decode_test': False,'decode_rule':False, 'decode_match':False, 'analyze_currents':False}

    #data_dir = '/media/masse/MySSDataStor1/Short-Term-Synaptic-Plasticity/savedir_resubmission/FINAL/'
    data_dir = './savedir_FINAL/'
    #filenames = os.listdir(data_dir)

    for t in task_list:
        for j in range(0,20,1):
            for sc in [5]:
                wc = 0

                fn = data_dir + t + '_wc' + str(wc) + '_sc' + str(sc) + '_tcm2_balEI1_L2_lr2_v' + str(j) + '.pkl'
                print('Analyzing ', fn)
                analyze_model_from_file(fn, savefile = fn, update_params = update_params)



def analyze_model_from_file(filename, savefile = None, update_params = {}):

    results = pickle.load(open(filename, 'rb'))

    if savefile is None:
        results['parameters']['save_fn'] = 'test.pkl'
    else:
        results['parameters']['save_fn'] = savefile

    update_parameters(results['parameters'])
    update_parameters(update_params)

    stim = stimulus.Stimulus()
    print('time step ', par['num_time_steps'])
    print('st ', par['sample_time'], 'trial tyep ', par['trial_type'], results['parameters']['num_time_steps'])

    # generate trials with match probability at 50%
    trial_info = stim.generate_trial(test_mode = False)
    print(trial_info['neural_input'].shape)
    input_data = np.squeeze(np.split(trial_info['neural_input'], trial_info['neural_input'].shape[1], axis=1))

    h_init = np.array(results['weights']['hidden_init'])
    #h_init = np.random.uniform(0, 0.5, size = (par['n_hidden'],1))
    #h_init = np.array(results['parameters']['h_init'])

    y_hat, h, syn_x, syn_u = run_model(input_data, h_init, \
        results['parameters']['syn_x_init'], results['parameters']['syn_u_init'], results['weights'])

    #print('mean activity ', np.mean(h))
    #return None
    # generate trials with random sample and test stimuli, used for decoding
    trial_info_decode = stim.generate_trial(test_mode = True)
    input_data = np.squeeze(np.split(trial_info_decode['neural_input'], trial_info_decode['neural_input'].shape[1], axis=1))
    _, h_decode, syn_x_decode, syn_u_decode = run_model(input_data, h_init, \
        results['parameters']['syn_x_init'], results['parameters']['syn_u_init'], results['weights'])


    # generate trials using DMS task, only used for measuring how neuronal and synaptic representations evolve in
    # a standardized way, used for figure correlating persistent activity and manipulation
    update_parameters({'trial_type': 'DMS'})
    #stim = []
    #stim = stimulus.Stimulus()
    trial_info_dms = stim.generate_trial(test_mode = True)
    input_data = np.squeeze(np.split(trial_info_dms['neural_input'], trial_info_dms['neural_input'].shape[1], axis=1))
    _, h_dms, syn_x_dms, syn_u_dms = run_model(input_data, h_init, \
        results['parameters']['syn_x_init'], results['parameters']['syn_u_init'], results['weights'])
    update_parameters(results['parameters']) # reset trial type to original value
    update_parameters(update_params)

    if par['save_trial_data']:
        results['trial_info'] = trial_info
        results['syn_x'] = syn_x
        results['syn_u'] = syn_u
        results['y_hat'] = y_hat
        results['h']     = h

    print('decode_stability ', par['decode_stability'])

    trial_time = np.arange(0,h.shape[1]*par['dt'], par['dt'])
    trial_time_dms = np.arange(0,h_dms.shape[1]*par['dt'], par['dt'])
    lesion = False

    """
    Calculate task accuracy
    """
    results['task_accuracy'],_,_ = get_perf(trial_info['desired_output'], y_hat, trial_info['train_mask'])
    results['task_accuracy_per_rule'] = []
    y_hat_stacked = np.stack(y_hat)
    for r in np.unique(trial_info['rule']):
        ind = np.where(trial_info['rule'] == r)[0]
        acc, _, _ = get_perf(trial_info['desired_output'][:,:,ind], y_hat_stacked[:,:,ind], trial_info['train_mask'][:, ind])
        results['task_accuracy_per_rule'].append(acc)


    if par['calculate_resp_matrix']:
        print('calculate response matrix...')
        resp_matrix_results = calculate_response_matrix(trial_info_decode, results['weights'])
        for key, val in resp_matrix_results.items():
            if np.var(val) > 0:
                results[key] = val

    """
    Decode the sample direction from neuronal activity and synaptic efficacies
    using support vector machines
    """
    if par['decoding_reps'] > 0:
        print('decoding activity...')

        decoding_results = calculate_svms(h_decode, syn_x_decode, syn_u_decode, trial_info_decode, trial_time, \
            num_reps = par['decoding_reps'], num_reps_stability = 10, decode_test = par['decode_test'], decode_rule = par['decode_rule'], \
            decode_sample_vs_test = par['decode_sample_vs_test'], decode_match = par['decode_match'])
        for key, val in decoding_results.items():
            if np.var(val) > 0:
                results[key] = val

        if par['trial_type'] in ['DMS', 'DMC', 'DMRS90', 'DMRS90ccw', 'DMRS45', 'DMRS180', 'location_DMS']:
            for key, val in decoding_results.items():
                if np.var(val) > 0:
                    results[key + '_dms'] = val

        else:
            # calculate decoding for a DMS trial,
            # used to correlate persistent activity and manipulation
            update_parameters({'trial_type': 'DMS'})
            decoding_results = calculate_svms(h_dms, syn_x_dms, syn_u_dms, trial_info_dms, trial_time_dms, \
                num_reps = par['decoding_reps'], num_reps_stability = 0, decode_test = par['decode_test'], decode_rule = par['decode_rule'], \
                decode_sample_vs_test = par['decode_sample_vs_test'])
            for key, val in decoding_results.items():
                if np.var(val) > 0:
                    results[key + '_dms'] = val
            update_parameters(results['parameters'])
            update_parameters(update_params)


    """
    Calculate neuronal and synaptic sample motion tuning
    """
    if par['analyze_tuning']:
        print('calculate tuning...')

        tuning_results = calculate_tuning(h_decode, syn_x_decode, syn_u_decode, \
            trial_info_decode, trial_time, results['weights'], calculate_test = par['decode_test'])
        for key, val in tuning_results.items():
            if np.var(val) > 0:
                results[key] = val

        # calculate tuning for a DMS trial,
        # used to correlate persistent activity and manipulation
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

    #plt.imshow(tuning_results['neuronal_pev'][:,0,0,:], aspect='auto')
    #plt.colorbar()
    #plt.show()



    """
    Calculate mean sample traces
    """
    results['h_sample_mean'] = np.zeros((results['parameters']['n_hidden'], results['parameters']['num_time_steps'], results['parameters']['num_motion_dirs']), dtype = np.float32)
    for i in range(results['parameters']['num_motion_dirs']):
        ind = np.where(trial_info_decode['sample'] == i)[0]
        results['h_sample_mean'][:,:,i] = np.mean(h_decode[:,:,ind], axis = 2)


    """
    Calculate accuracy after lesioning weights
    """
    if lesion:
        print('lesioning weights...')
        lesion_results = lesion_weights(trial_info, h, syn_x, syn_u, results['weights'], trial_time)
        for key, val in lesion_results.items():
             if np.var(val) > 0:
                 results[key] = val

    """
    Calculate the neuronal and synaptic contributions towards solving the task
    """
    if par['simulation_reps'] > 0:
        print('simulating network...')
        simulation_results = simulate_network(trial_info, h, syn_x, \
            syn_u, results['weights'], num_reps = par['simulation_reps'])
        for key, val in simulation_results.items():
            if np.var(val) > 0:
                results[key] = val

    """
    Calculate currents
    """
    if par['analyze_currents']:
        print('calculate current...')
        current_results = calculate_currents(h, syn_x, syn_u, trial_info, results['weights'])
        for key, val in current_results.items():
            results[key] = val
            #x[key] = val # added just to be able to run cut_weights in one analysis run
        pickle.dump(results, open(savefile, 'wb'))




    pickle.dump(results, open(savefile, 'wb') )
    print('Analysis results saved in ', savefile)
    print(results.keys())

    """
    print('Performing dPCA...')
    dim_results = dimension_reduction(h, syn_x, syn_u, trial_info, trial_time)
    for key, val in dim_results.items():
        results[key] = val

    pickle.dump(results, open(savefile, 'wb') )
    print('Analysis results saved in ', savefile)
    print(results.keys())
    """

def calculate_currents(h, syn_x, syn_u, trial_info, weights):

    trial_length = h.shape[1]
    current_results = {
        'match'            :  np.zeros((3, par['n_hidden'], trial_length, 2),dtype=np.float32),
        'non_match'        :  np.zeros((3, par['n_hidden'], trial_length, 2),dtype=np.float32)}

    weights['w_out'] = np.maximum(0, weights['w_out'])
    weights['w_rnn'] = np.maximum(0, weights['w_rnn'])
    w_rnn_eff = np.matmul(weights['w_rnn'], par['EI_matrix'])

    neuron_groups = []
    neuron_groups.append(range(0,par['num_exc_units'],2))
    neuron_groups.append(range(1,par['num_exc_units'],2))
    neuron_groups.append(range(par['num_exc_units'],par['num_exc_units']+par['num_inh_units'],2))
    neuron_groups.append(range(par['num_exc_units']+1,par['num_exc_units']+par['num_inh_units'],2))


    ind_match = np.where(trial_info['match']==1)[0]
    ind_non_match = np.where(trial_info['match']==0)[0]
    mean_activity_match     = np.mean(h[:, :, ind_match], axis=2)
    mean_eff_activity_match = np.mean(h[:, :, ind_match]*syn_x[:, :, ind_match]*syn_u[:, :, ind_match], axis=2)
    mean_activity_non_match     = np.mean(h[:, :, ind_non_match], axis=2)
    mean_eff_activity_non_match = np.mean(h[:, :, ind_non_match]*syn_x[:, :, ind_non_match]*syn_u[:, :, ind_non_match], axis=2)

    for n in range(par['n_hidden']):

        w_out = np.reshape(weights['w_out'][:,n], (3,1))
        w_rnn =  par['EI_matrix'][n,n]*np.reshape(weights['w_rnn'][:,n], (100,1))

        # no STP to output layer
        direct_out_match = np.matmul(w_out, np.reshape(mean_activity_match[n,:], (1,trial_length)))
        direct_out_non_match = np.matmul(w_out, np.reshape(mean_activity_non_match[n,:], (1,trial_length)))
        direct_out_eff_match = np.matmul(w_out,np.reshape(mean_activity_match[n,:], (1,trial_length)))
        direct_out_eff_non_match = np.matmul(w_out, np.reshape(mean_activity_non_match[n,:], (1,trial_length)))

        indirect_out_match = np.matmul(weights['w_out'], np.matmul(w_rnn, np.reshape(mean_activity_match[n,:], (1,trial_length))))
        indirect_out_non_match = np.matmul(weights['w_out'], np.matmul(w_rnn, np.reshape(mean_activity_non_match[n,:], (1,trial_length))))
        indirect_out_eff_match = np.matmul(weights['w_out'], np.matmul(w_rnn, np.reshape(mean_eff_activity_match[n,:], (1,trial_length))))
        indirect_out_eff_non_match = np.matmul(weights['w_out'], np.matmul(w_rnn, np.reshape(mean_eff_activity_non_match[n,:], (1,trial_length))))

        #s=np.matmul(w_rnn_eff, np.reshape(mean_activity_match[n,:], (1,trial_length)))
        #print(s.shape)
        indirect_out_match2 = np.matmul(weights['w_out'], np.matmul(w_rnn_eff, np.matmul(w_rnn, np.reshape(mean_activity_match[n,:], (1,trial_length)))))
        indirect_out_non_match2 = np.matmul(weights['w_out'], np.matmul(w_rnn_eff, np.matmul(w_rnn, np.reshape(mean_activity_non_match[n,:], (1,trial_length)))))
        indirect_out_eff_match2 = np.matmul(weights['w_out'], np.matmul(w_rnn_eff, np.matmul(w_rnn, np.reshape(mean_eff_activity_match[n,:], (1,trial_length)))))
        indirect_out_eff_non_match2 = np.matmul(weights['w_out'], np.matmul(w_rnn_eff, np.matmul(w_rnn, np.reshape(mean_eff_activity_non_match[n,:], (1,trial_length)))))


        #indirect_out_match = np.matmul(weights['w_out'], np.matmul(w_rnn, h[n,:,ind_match]))
        #indirect_out_non_match = np.matmul(weights['w_out'], np.matmul(w_rnn,h[n,:,ind_non_match]))
        #indirect_out_eff_match = np.matmul(weights['w_out'], np.matmul(w_rnn, h[n,:,ind_match]*syn_x[:,:,ind_match]*syn_u[:,:,ind_match]))
        #-indirect_out_eff_non_match = np.matmul(weights['w_out'], np.matmul(w_rnn, h[n,:,ind_non_match]*syn_x[:,:,ind_non_match]*syn_u[:,:,ind_non_match]))


        current_results['match'][:,n,:,0] = direct_out_match + indirect_out_match2+ indirect_out_match*0
        current_results['match'][:,n,:,1] = direct_out_eff_match + indirect_out_eff_match2 + indirect_out_eff_match*0
        current_results['non_match'][:,n,:,0] = direct_out_non_match + indirect_out_non_match2 + indirect_out_non_match*0
        current_results['non_match'][:,n,:,1] = direct_out_eff_non_match + indirect_out_eff_non_match2+ indirect_out_eff_non_match*0
    """
    for i in range(4):
        m = np.mean(current_results['match'][:,neuron_groups[i],:,1], axis = 0)
        print(current_results['match'][:,neuron_groups[i],:,1].shape)
        print(m.shape)
        nm = np.mean(current_results['non_match'][:,neuron_groups[i],:,1], axis = 0)
        plt.imshow(m-nm, aspect = 'auto')
        plt.colorbar()
        plt.show()


    plt.imshow(current_results['match'][:,:,235,1], aspect = 'auto')
    plt.colorbar()
    plt.show()
    """


    return current_results

def dimension_reduction(h, syn_x, syn_u, trial_info, trial_time):

    num_rules = len(np.unique(trial_info['rule']))
    for r in range(num_rules):
        ind_rule = np.where(trial_info['rule']==r)[0]
        print('len rule ind', len(ind_rule))
        for rf in range(par['num_receptive_fields']):
            h_mean = np.zeros((par['n_hidden'], par['num_time_steps'], par['num_motion_dirs']))
            for n in range(par['num_motion_dirs']):
                ind = np.where((trial_info['sample']==n)*(trial_info['rule']==r)*(trial_info['sample']==n))[0]
                h_mean[:, :, n] = np.mean(h[:,:,ind], axis = 2)

            h_mean = np.transpose(np.reshape(h_mean,(par['n_hidden'], -1)))
            """
            pca = PCA(n_components=3)
            pca.fit(h_mean)
            PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,\
                svd_solver='auto', tol=0.0, whiten=False)
            y = pca.transform(h_mean)
            y = np.reshape(y,(3, par['num_motion_dirs'], par['num_time_steps'] ),order='F')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(y[0,0,:],y[1,0,:],y[2,0,:],'b')
            ax.plot(y[0,4,:],y[1,4,:],y[2,4,:],'r')
            #ax.plot(y[0,:,2],y[1,:,2],y[2,:,2],'g')
            #ax.plot(y[0,:,3],y[1,:,3],y[2,:,3],'k')
            plt.show()

            fa = FactorAnalysis(n_components=3)
            fa.fit(h_mean)
            y = fa.transform(h_mean)
            y = np.reshape(y,(3, par['num_motion_dirs'], par['num_time_steps'] ),order='F')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(y[0,0,:],y[1,0,:],y[2,0,:],'b')
            ax.plot(y[0,4,:],y[1,4,:],y[2,4,:],'r')
            #ax.plot(y[0,:,2],y[1,:,2],y[2,:,2],'g')
            #ax.plot(y[0,:,3],y[1,:,3],y[2,:,3],'k')
            plt.show()

            print(pca.explained_variance_ratio_)
            print(y.shape)
            """

            num_trials = 128
            h_trial = np.zeros((num_trials, par['n_hidden'], par['num_time_steps'], par['num_motion_dirs']), dtype = np.float32)
            s_trial = np.zeros((num_trials, par['n_hidden'], par['num_time_steps'], par['num_motion_dirs']), dtype = np.float32)
            for n in range(par['num_motion_dirs']):
                ind = np.where((trial_info['sample']==n)*(trial_info['rule']==r)*(trial_info['sample']==n))[0]
                # in case we don't have enough trials for this condition, we'll reuse some trials
                print('Number of trials ', len(ind) , ' required: ', num_trials)
                ind = np.array([*list(ind), *list(ind)])

                for i in range(num_trials):
                    h_trial[i,:,:,n] = h[:,:,ind[i]]
                    s_trial[i,:,:,n] = syn_x[:,:,ind[i]]*syn_u[:,:,ind[i]]

            # trial-average data
            h = np.mean(h_trial,axis = 0)
            s = np.mean(s_trial,axis = 0)
            # center data
            h -= np.mean(h.reshape((par['n_hidden'],-1)),1)[:,None,None]
            s -= np.mean(s.reshape((par['n_hidden'],-1)),1)[:,None,None]

            dpca = dPCA(labels='st',regularizer='auto')
            dpca.protect = ['t']

            dpca_neuronal = dpca.fit_transform(h,h_trial)
            dpca_synaptic = dpca.fit_transform(s,s_trial)
            results = {'dpca_neuronal': dpca_neuronal, 'dpca_synaptic': dpca_synaptic}

        return results






def calculate_svms(h, syn_x, syn_u, trial_info, trial_time, num_reps = 20, num_reps_stability = 5, \
    decode_test = False, decode_rule = False, decode_sample_vs_test = False, decode_match = False, decode_neuronal_groups = False):

    """
    Calculates neuronal and synaptic decoding accuracies uisng support vector machines
    sample is the index of the sample motion direction for each trial_length
    rule is the rule index for each trial_length
    """

    lin_clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr', shrinking=False, tol=1e-3)

    #lin_clf_lda = lda(solver = 'svd', shrinkage=None)

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


    print('sample decoding...num_reps = ', num_reps)
    print('mean dyn synpase ', np.mean(par['dynamic_synapse']))


    decoding_results['neuronal_sample_decoding'], decoding_results['synaptic_sample_decoding'], \
        decoding_results['neuronal_sample_decoding_stability'], decoding_results['synaptic_sample_decoding_stability'] = \
        svm_wraper(lin_clf, h, syn_efficacy, sample, rule, num_reps, num_reps_stability, trial_time)

    to = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
    print('DELAY NEURONAL SAMPLE DECODING')
    print(np.mean(decoding_results['neuronal_sample_decoding'][0,0,:,to-10:to]))
    print(np.sum(np.mean(decoding_results['neuronal_sample_decoding'][0,0,:,to-10:to],axis=1)>0.125))

    if decode_sample_vs_test:
        print('sample vs. test decoding...')
        decoding_results['neuronal_sample_test_decoding'], decoding_results['synaptic_sample_test_decoding'] ,_ ,_ = \
            svm_wraper_sample_vs_test(lin_clf, h, syn_efficacy, trial_info['sample'], trial_info['test'], num_reps, num_reps_stability, trial_time)

    if decode_test:
        print('test decoding...')
        decoding_results['neuronal_test_decoding'], decoding_results['synaptic_test_decoding'] ,_ ,_ = \
            svm_wraper(lin_clf, h, syn_efficacy, test, rule, num_reps, 0, trial_time)

    if decode_match:
        print('match decoding...')
        print(match)
        decoding_results['neuronal_match_decoding'], decoding_results['synaptic_match_decoding'] ,_ ,_ = \
            svm_wraper(lin_clf, h, syn_efficacy, match, rule, num_reps, 0, trial_time)

    if decode_rule:
        print('rule decoding...')
        decoding_results['neuronal_rule_decoding'], decoding_results['synaptic_rule_decoding'] ,_ ,_ = \
            svm_wraper(lin_clf, h, syn_efficacy, trial_info['rule'], np.zeros_like(rule), num_reps, 0, trial_time)
    """
    plt.plot(np.mean(decoding_results['neuronal_sample_decoding'][0,0,:,:], axis = 0),'g')
    plt.plot(np.mean(decoding_results['synaptic_sample_decoding'][0,0,:,:], axis = 0),'m')
    plt.plot(np.mean(decoding_results['neuronal_test_decoding'][0,0,:,:], axis = 0),'b')
    plt.plot(np.mean(decoding_results['synaptic_test_decoding'][0,0,:,:], axis = 0),'r')
    plt.show()
    """

    if decode_neuronal_groups:

        neuron_groups = []
        neuron_groups.append(range(0,par['num_exc_units'],2))
        neuron_groups.append(range(1,par['num_exc_units'],2))
        neuron_groups.append(range(par['num_exc_units'],par['num_exc_units']+par['num_inh_units'],2))
        neuron_groups.append(range(par['num_exc_units']+1,par['num_exc_units']+par['num_inh_units'],2))

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
    _, num_time_steps, num_trials = h.shape
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

    """
    Wraper function used to decode sample/test or rule information
    from hidden activity (h) and synaptic efficacies (syn_eff)
    """
    train_pct = 0.75
    _, num_time_steps, num_trials = h.shape
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
        lin_clf.fit(y[:,t,train_ind].T, train_conds[train_ind])
        for t1 in range(par['dead_time']//par['dt'],par['num_time_steps']):
            dec = lin_clf.predict(y[:,t1,test_ind].T)
            score[t, t1] = np.mean(test_conds[test_ind] == dec)

    return score

def calc_svm(lin_clf, y, train_conds, test_conds, train_ind, test_ind):

    n_test_inds = len(test_ind)
    score = np.zeros((par['num_time_steps']))

    y = normalize_values(y, train_ind)

    for t in range(par['dead_time']//par['dt'], par['num_time_steps']):
        lin_clf.fit(y[:,t,train_ind].T, train_conds[train_ind])
        dec = lin_clf.predict(y[:,t,test_ind].T)
        score[t] = np.mean(test_conds[test_ind]==dec)

    return score

def normalize_values(z, train_ind):

    if not par['svm_normalize']:
        return z

    # normalize values between 0 and 1
    for t in range(z.shape[1]):
        for i in range(z.shape[0]):
            m1 = z[i,t,train_ind].min()
            m2 =  z[i,t,train_ind].max()
            z[i,t,:] -= m1
            if m2>m1:
                z[i,t,:] /=(m2-m1)

    return z



def lesion_weights(trial_info, h, syn_x, syn_u, network_weights, trial_time):

    lesion_results = {'lesion_accuracy_rnn': np.ones((par['num_rules'], par['n_hidden'],par['n_hidden']), dtype=np.float32),
                      'lesion_accuracy_out': np.ones((par['num_rules'], 3,par['n_hidden']), dtype=np.float32)}

    for r in range(par['num_rules']):
        trial_ind = np.where(trial_info['rule']==r)[0]
        # network inputs/outputs
        test_onset = (par['dead_time']+par['fix_time'])//par['dt']
        x = np.split(trial_info['neural_input'][:,test_onset:,trial_ind],len(trial_time)-test_onset,axis=1)
        y = np.array(trial_info['desired_output'][:,test_onset:,trial_ind])
        train_mask = np.array(trial_info['train_mask'][test_onset:,trial_ind])
        hidden_init = h[:,test_onset-1,trial_ind]
        syn_x_init = syn_x[:,test_onset-1,trial_ind]
        syn_u_init = syn_u[:,test_onset-1,trial_ind]

        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']

        hidden_init_test = h[:,test_onset-1,trial_ind]
        syn_x_init_test = syn_x[:,test_onset-1,trial_ind]
        syn_u_init_test = syn_u[:,test_onset-1,trial_ind]
        x_test = np.split(trial_info['neural_input'][:,test_onset:,trial_ind],len(trial_time)-test_onset,axis=1)
        y_test = trial_info['desired_output'][:,test_onset:,trial_ind]
        train_mask_test = trial_info['train_mask'][test_onset:,trial_ind]

        print('Lesioning output weights...')
        for n1 in range(3):
            for n2 in range(par['n_hidden']):

                if network_weights['w_out'][n1,n2] <= 0:
                    continue

                # create new dict of weights
                weights_new = {}
                for k,v in network_weights.items():
                    weights_new[k] = np.array(v+1e-32)

                # lesion weights
                q = np.ones((3,par['n_hidden']), dtype=np.float32)
                q[n1,n2] = 0
                weights_new['w_out'] *= q

                # simulate network
                y_hat, _, _, _ = run_model(x_test, hidden_init_test, syn_x_init_test, syn_u_init_test, weights_new)
                lesion_results['lesion_accuracy_out'][r,n1,n2],_,_ = get_perf(y_test, y_hat, train_mask_test)

        print('Lesioning recurrent weights...')
        for n1 in range(par['n_hidden']):
            for n2 in range(par['n_hidden']):

                if network_weights['w_rnn'][n1,n2] <= 0:
                    continue

                weights_new = {}
                for k,v in network_weights.items():
                    weights_new[k] = np.array(v+1e-32)

                # lesion weights
                q = np.ones((par['n_hidden'],par['n_hidden']), dtype=np.float32)
                q[n1,n2] = 0
                weights_new['w_rnn'] *= q

                # simulate network
                y_hat, hidden_state_hist, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, weights_new)
                lesion_results['lesion_accuracy_rnn'][r,n1,n2],_,_ = get_perf(y, y_hat, train_mask)

                #y_hat, _, _, _ = run_model(x_test, hidden_init_test, syn_x_init_test, syn_u_init_test, weights_new)
                #lesion_results['lesion_accuracy_rnn_test'][r,n1,n2],_,_ = get_perf(y_test, y_hat, train_mask_test)

                """
                if accuracy_rnn_start[n1,n2] < -1:

                    h_stacked = np.stack(hidden_state_hist, axis=1)

                    neuronal_decoding[n1,n2,:,:,:], _ = calculate_svms(h_stacked, syn_x, syn_u, trial_info['sample'], \
                        trial_info['rule'], trial_info['match'], trial_time, num_reps = num_reps)

                    neuronal_pref_dir[n1,n2,:,:], neuronal_pev[n1,n2,:,:], _, _ = calculate_sample_tuning(h_stacked, \
                        syn_x, syn_u, trial_info['sample'], trial_info['rule'], trial_info['match'], trial_time)
                """


    return lesion_results




def calculate_response_matrix(trial_info, network_weights):

    test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
    trial_length = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time'])//par['dt']

    suppression_time_range = range(test_onset,trial_length)

    resp_matrix_results = {'resp_no_suppresion' : np.zeros((par['n_hidden'], par['num_motion_dirs'], par['num_motion_dirs']), dtype = np.float32),
                           'resp_suppresion' : np.zeros((par['n_hidden'], par['n_hidden'], par['num_motion_dirs'], par['num_motion_dirs']), dtype = np.float32)}

    x = np.split(trial_info['neural_input'],trial_length,axis=1)

    _, h, _, _ = run_model(x, par['h_init'], par['syn_x_init'], par['syn_u_init'], network_weights)
    resp_matrix_results['resp_no_suppresion'] = average_test_response(h, trial_info, test_onset)

    for n in range(par['n_hidden']):
        suppress_activity = np.ones((par['n_hidden'], trial_length))
        for m in suppression_time_range:
            suppress_activity[n,m] = 0

        suppress_activity = np.split(suppress_activity, trial_length, axis=1)
        _, h,_,_ = run_model(x, par['h_init'], par['syn_x_init'], par['syn_u_init'], \
            network_weights, suppress_activity = suppress_activity)
        resp_matrix_results['resp_suppresion'][n,:,:,:] = average_test_response(h, trial_info, test_onset)


    return resp_matrix_results

def average_test_response(h, trial_info, test_onset):

    resp = np.zeros((par['n_hidden'], par['num_motion_dirs'], par['num_motion_dirs']), dtype = np.float32)
    h1 = np.mean(h[:, test_onset:, ],axis=1)
    for i in range(par['num_motion_dirs']):
        for j in range(par['num_motion_dirs']):
            ind = np.where((trial_info['sample']==i)*(trial_info['test']==j))[0]
            resp[:, i, j] = np.mean(h1[:, ind], axis = 1)

    return resp


def simulate_network(trial_info, h, syn_x, syn_u, network_weights, num_reps = 20):


    epsilon = 1e-3


    """
    Simulation will start from the start of the test period until the end of trial
    """
    if par['trial_type'] == 'dualDMS':
        test_onset = [(par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+par['test_time'])//par['dt']]
    elif  par['trial_type'] in ['ABBA','ABCA']:
        test_onset = [(par['dead_time']+par['fix_time']+par['sample_time']+i*par['ABBA_delay'])//par['dt'] for i in range(1,2)]
    elif  par['trial_type'] in ['DMS', 'DMC', 'DMRS90', 'DMRS90ccw']:
        test_onset = []
        test_onset.append((par['dead_time']+par['fix_time']+par['sample_time'])//par['dt'])
        test_onset.append((par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt'])


    num_test_periods = len(test_onset)
    """
    suppression_time_range = []
    for k in range(11):
        suppression_time_range.append(range(test_onset[-1]-k*2, test_onset[0]))
    """
    suppression_time_range = [range(test_onset[0]-200//par['dt'], test_onset[0])]

    neuron_groups = []
    if par['trial_type'] in ['DMS', 'DMC', 'DMRS90', 'DMRS90ccw','ABBA','ABCA']:
        neuron_groups.append(range(0,par['num_exc_units'],2))
        neuron_groups.append(range(1,par['num_exc_units'],2))
        neuron_groups.append(range(par['num_exc_units'],par['num_exc_units']+par['num_inh_units'],2))
        neuron_groups.append(range(par['num_exc_units']+1,par['num_exc_units']+par['num_inh_units'],2))
        neuron_groups.append(range(par['n_hidden']))

    syn_efficacy = syn_x*syn_u
    test = np.array(trial_info['test'])
    print('TTTTEST ', test.shape)
    sample = np.array(trial_info['sample'])
    if test.ndim == 2:
        test = test[:, 0]
    elif test.ndim == 3:
        test = test[:, 0, 0]
    test_dir = np.ones((len(test), 3))
    test_dir[:,1] = np.cos(2*np.pi*test/par['num_motion_dirs'])
    test_dir[:,2] = np.sin(2*np.pi*test//par['num_motion_dirs'])

    n_hidden, trial_length, batch_train_size = h.shape
    num_grp_reps = 5

    simulation_results = {
        'simulation_accuracy'            : np.zeros((par['num_rules'], num_test_periods, num_reps)),
        'accuracy_neural_shuffled'      : np.zeros((par['num_rules'], num_test_periods, num_reps)),
        'accuracy_syn_shuffled'         : np.zeros((par['num_rules'], num_test_periods, num_reps)),
        'accuracy_suppression'          : np.zeros((par['num_rules'], len(suppression_time_range), len(neuron_groups), 3)),
        'accuracy_neural_shuffled_grp'  : np.zeros((par['num_rules'], num_test_periods, len(neuron_groups), num_grp_reps)),
        'accuracy_syn_shuffled_grp'     : np.zeros((par['num_rules'], num_test_periods, len(neuron_groups), num_reps)),
        'synaptic_pev_test_suppression' : np.zeros((par['num_rules'], num_test_periods, len(neuron_groups), n_hidden, trial_length)),
        'synaptic_pref_dir_test_suppression': np.zeros((par['num_rules'], num_test_periods, len(neuron_groups), n_hidden, trial_length))}



    mask = np.array(trial_info['train_mask'])
    if par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        t0 = (par['dead_time']+par['fix_time']+par['sample_time'] + 2*par['ABBA_delay'])//par['dt']
        mask[:t0,:] = 0
        t0 = (par['dead_time']+par['fix_time']+par['sample_time'] + 4*par['ABBA_delay'])//par['dt']
        mask[t0:,:] = 0


    for r in range(par['num_rules']):
        for t in range(num_test_periods):

            test_length = trial_length - test_onset[t]
            trial_ind = np.where(trial_info['rule']==r)[0]
            train_mask = mask[test_onset[t]:,trial_ind]
            x = np.split(trial_info['neural_input'][:,test_onset[t]:,trial_ind],test_length,axis=1)
            y = trial_info['desired_output'][:,test_onset[t]:,trial_ind]

            for n in range(num_reps):

                """
                Calculating behavioral accuracy without shuffling
                """
                hidden_init = np.array(h[:,test_onset[t]-1,trial_ind])
                syn_x_init = np.array(syn_x[:,test_onset[t]-1,trial_ind])
                syn_u_init = np.array(syn_u[:,test_onset[t]-1,trial_ind])
                y_hat, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
                simulation_results['simulation_accuracy'][r,t,n] ,_ ,_ = get_perf(y, y_hat, train_mask)

                """
                Keep the synaptic values fixed, permute the neural activity
                """
                ind_shuffle = np.random.permutation(len(trial_ind))
                hidden_init = hidden_init[:,ind_shuffle]
                y_hat, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
                simulation_results['accuracy_neural_shuffled'][r,t,n] ,_ ,_ = get_perf(y, y_hat, train_mask)

                """
                Keep the hidden values fixed, permute synaptic values
                """
                hidden_init = np.array(h[:,test_onset[t]-1,trial_ind])
                syn_x_init = syn_x_init[:,ind_shuffle]
                syn_u_init = syn_u_init[:,ind_shuffle]
                y_hat, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
                simulation_results['accuracy_syn_shuffled'][r,t,n] ,_ ,_ = get_perf(y, y_hat, train_mask)

            for n in range(num_grp_reps):
                #Neuron group shuffling

                for g in range(len(neuron_groups)):

                    # reset everything
                    hidden_init = np.array(h[:,test_onset[t]-1,trial_ind])
                    syn_x_init = np.array(syn_x[:,test_onset[t]-1,trial_ind])
                    syn_u_init = np.array(syn_u[:,test_onset[t]-1,trial_ind])

                    # shuffle neuronal activity
                    ind_shuffle = np.random.permutation(len(trial_ind))
                    for neuron_num in neuron_groups[g]:
                        hidden_init[neuron_num,:] = hidden_init[neuron_num,ind_shuffle]
                    y_hat, _, syn_x_hist, syn_u_hist = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
                    simulation_results['accuracy_neural_shuffled_grp'][r,t,g,n] ,_ ,_ = get_perf(y, y_hat, train_mask)


                    if par['trial_type'] in ['ABBA','ABCA']:
                        syn_efficacy = syn_x_hist*syn_u_hist
                        for hidden_num in range(par['n_hidden']):
                            for t1 in range(test_length):
                                weights = np.linalg.lstsq(test_dir[trial_ind,:], syn_efficacy[hidden_num,t1,trial_ind])
                                weights = np.reshape(weights[0],(3,1))
                                pred_err = syn_efficacy[hidden_num,t1,trial_ind] - np.dot(test_dir[trial_ind,:], weights).T
                                mse = np.mean(pred_err**2)
                                response_var = np.var(syn_efficacy[hidden_num,t1,trial_ind])
                                simulation_results['synaptic_pev_test_shuffled'][r,t,g,n, hidden_num,t1+test_onset[t]] = 1 - mse/(response_var + epsilon)
                                simulation_results['synaptic_pref_dir_test_shuffled'][r,t,g,n,hidden_num,t1+test_onset[t]] = np.arctan2(weights[2,0],weights[1,0])

                    # reset neuronal activity, shuffle synaptic activity
                    hidden_init = h[:,test_onset[t]-1,trial_ind]
                    for neuron_num in neuron_groups[g]:
                        syn_x_init[neuron_num,:] = syn_x_init[neuron_num,ind_shuffle]
                        syn_u_init[neuron_num,:] = syn_u_init[neuron_num,ind_shuffle]
                    y_hat, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
                    simulation_results['accuracy_syn_shuffled_grp'][r,t,g,n] ,_ ,_ = get_perf(y, y_hat, train_mask)

                    if par['trial_type'] in ['ABBA','ABCA']:
                        syn_efficacy = syn_x_hist*syn_u_hist
                        for hidden_num in range(par['n_hidden']):
                            for t1 in range(test_length):
                                weights = np.linalg.lstsq(test_dir[trial_ind,:], syn_efficacy[hidden_num,t1,trial_ind])
                                weights = np.reshape(weights[0],(3,1))
                                pred_err = syn_efficacy[hidden_num,t1,trial_ind] - np.dot(test_dir[trial_ind,:], weights).T
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
            x = np.split(trial_info['neural_input'][:,:,trial_ind],trial_length,axis=1)
            y = trial_info['desired_output'][:,:,trial_ind]
            train_mask = np.array(mask[:,trial_ind])
            """
            train_mask = trial_info['train_mask'][:,trial_ind]
            if par['trial_type'] == 'ABBA' or  par['trial_type'] == 'ABCA':
                train_mask[test_onset_sup + par['ABBA_delay']//par['dt']:, :] = 0
            """

            syn_x_init = np.array(syn_x[:,0,trial_ind])
            syn_u_init = np.array(syn_u[:,0,trial_ind])
            hidden_init = np.array(h[:,0,trial_ind])

            y_hat, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
            acc, acc_non_match, acc_match = get_perf(y, y_hat, train_mask)
            simulation_results['accuracy_no_suppression'] = np.array([acc, acc_non_match, acc_match])

            for k in range(len(suppression_time_range)):
                for k1 in range(len(neuron_groups)):

                    suppress_activity = np.ones((par['n_hidden'], trial_length))
                    for m1 in neuron_groups[k1]:
                        for m2 in suppression_time_range[k]:
                            suppress_activity[m1,m2] = 0

                    suppress_activity = np.split(suppress_activity, trial_length, axis=1)
                    syn_x_init = np.array(syn_x[:,0,trial_ind])
                    syn_u_init = np.array(syn_u[:,0,trial_ind])
                    hidden_init = np.array(h[:,0,trial_ind])

                    y_hat, _, syn_x_sim, syn_u_sim = run_model(x, hidden_init, syn_x_init, \
                        syn_u_init, network_weights, suppress_activity = suppress_activity)
                    acc, acc_non_match, acc_match = get_perf(y, y_hat, train_mask)
                    simulation_results['accuracy_suppression'][r,k,k1,:] = np.array([acc, acc_non_match, acc_match])

                    syn_efficacy = syn_x_sim*syn_u_sim
                    for hidden_num in range(par['n_hidden']):
                        for t1 in range(syn_x_sim.shape[1]):
                            weights = np.linalg.lstsq(test_dir[trial_ind,:], syn_efficacy[hidden_num,t1,trial_ind])
                            weights = np.reshape(weights[0],(3,1))
                            pred_err = syn_efficacy[hidden_num,t1,trial_ind] - np.dot(test_dir[trial_ind,:], weights).T
                            mse = np.mean(pred_err**2)
                            response_var = np.var(syn_efficacy[hidden_num,t1,trial_ind])
                            simulation_results['synaptic_pev_test_suppression'][r,k,k1, hidden_num,t1] = 1 - mse/(response_var+1e-9)
                            simulation_results['synaptic_pref_dir_test_suppression'][r,k,k1,hidden_num,t1] = np.arctan2(weights[2,0],weights[1,0])


    return simulation_results

def calculate_tuning(h, syn_x, syn_u, trial_info, trial_time, network_weights, calculate_test = False):

    print('simulate_network calculate_test ', calculate_test)
    epsilon = 1e-9
    """
    Calculates neuronal and synaptic sample motion direction tuning
    """

    num_test_stimuli = 1 # only analyze the first test stimulus
    mask = np.array(trial_info['train_mask'])

    neuron_groups = []
    neuron_groups.append(range(0,par['num_exc_units'],2))
    neuron_groups.append(range(1,par['num_exc_units'],2))
    neuron_groups.append(range(par['num_exc_units'],par['num_exc_units']+par['num_inh_units'],2))
    neuron_groups.append(range(par['num_exc_units']+1,par['num_exc_units']+par['num_inh_units'],2))
    neuron_groups.append(range(par['n_hidden']))

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
        sample = np.reshape(np.array(trial_info['sample']),(par['batch_train_size'], 1))
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['ABBA_delay'])//par['dt']
        suppression_time_range = [range(test_onset-200//par['dt'], test_onset)]
        # onyl want to examine accuracy on 2nd pulse
        t0 = (par['dead_time']+par['fix_time']+par['sample_time'] + 2*par['ABBA_delay'])//par['dt']
        mask[:t0,:] = 0
        t0 = (par['dead_time']+par['fix_time']+par['sample_time'] + 4*par['ABBA_delay'])//par['dt']
        mask[t0:,:] = 0

    elif par['trial_type'] == 'location_DMS':
        par['num_receptive_fields'] = 1
        test = np.reshape(np.array(trial_info['test']),(par['batch_train_size'], 1))
        test_onset = (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
        sample = np.reshape(np.array(trial_info['sample']),(par['batch_train_size'], 1))
        rule = np.array(trial_info['rule'])
        match = np.array(trial_info['match'])
        suppression_time_range = [range(test_onset-200//par['dt'], test_onset)]
    else:
        rule = np.array(trial_info['rule'])
        match = np.array(trial_info['match'])
        sample = np.reshape(np.array(trial_info['sample']),(par['batch_train_size'], 1))
        test = np.reshape(np.array(trial_info['test']),(par['batch_train_size'], 1))
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


    """
    The synaptic efficacy is the product of syn_x and syn_u, will decode sample
    direction from this value
    """
    syn_efficacy = syn_x*syn_u


    sample_dir = np.ones((par['batch_train_size'], 3, par['num_receptive_fields']))
    for rf in range(par['num_receptive_fields']):
        sample_dir[:,1, rf] = np.cos(2*np.pi*sample[:,rf]/par['num_motion_dirs'])
        sample_dir[:,2, rf] = np.sin(2*np.pi*sample[:,rf]/par['num_motion_dirs'])

    test_dir = np.ones((par['batch_train_size'], 3, par['num_receptive_fields']))
    for rf in range(par['num_receptive_fields']):
        test_dir[:,1, rf] = np.reshape(np.cos(2*np.pi*test[:, rf]/par['num_motion_dirs']), (par['batch_train_size']))
        test_dir[:,2, rf] = np.reshape(np.sin(2*np.pi*test[:, rf]/par['num_motion_dirs']), (par['batch_train_size']))

    for r in range(par['num_rules']):
        trial_ind = np.where((rule==r))[0]
        print('RULE ', trial_ind)
        for n in range(par['n_hidden']):
            for t in range(num_time_steps):

                # Mean sample response
                for md in range(par['num_motion_dirs']):
                    for rf in range(par['num_receptive_fields']):
                        #print('rule', rule.shape)
                        #print('sample', sample.shape)
                        ind_motion_dir = np.where((rule==r)*(sample[:,rf]==md))[0]
                        #print(tuning_results['neuronal_sample_tuning'].shape)
                        tuning_results['neuronal_sample_tuning'][n,r,md,rf,t] = np.mean(h[n,t,ind_motion_dir])
                        tuning_results['synaptic_sample_tuning'][n,r,md,rf,t] = np.mean(syn_efficacy[n,t,ind_motion_dir])

                for rf in range(par['num_receptive_fields']):
                        # Neuronal sample tuning

                    weights = np.linalg.lstsq(sample_dir[trial_ind,:,rf], h[n,t,trial_ind])
                    weights = np.reshape(weights[0],(3,1))
                    pred_err = h[n,t,trial_ind] - np.dot(sample_dir[trial_ind,:,rf], weights).T
                    mse = np.mean(pred_err**2)
                    response_var = np.var(h[n,t,trial_ind])
                    if response_var > epsilon:
                        tuning_results['neuronal_pev'][n,r,rf,t] = 1 - mse/(response_var + epsilon)
                        tuning_results['neuronal_pref_dir'][n,r,rf,t] = np.arctan2(weights[2,0],weights[1,0])

                    if calculate_test:
                        weights = np.linalg.lstsq(test_dir[trial_ind,:,rf], h[n,t,trial_ind])
                        weights = np.reshape(weights[0],(3,1))
                        pred_err = h[n,t,trial_ind] - np.dot(test_dir[trial_ind,:,rf], weights).T
                        mse = np.mean(pred_err**2)
                        response_var = np.var(h[n,t,trial_ind])
                        if response_var > epsilon:
                            tuning_results['neuronal_pev_test'][n,r,rf,t] = 1 - mse/(response_var + epsilon)
                            tuning_results['neuronal_pref_dir_test'][n,r,rf,t] = np.arctan2(weights[2,0],weights[1,0])


                    # Synaptic sample tuning
                    weights = np.linalg.lstsq(sample_dir[trial_ind,:,rf], syn_efficacy[n,t,trial_ind])
                    weights = np.reshape(weights[0],(3,1))
                    pred_err = syn_efficacy[n,t,trial_ind] - np.dot(sample_dir[trial_ind,:,rf], weights).T
                    mse = np.mean(pred_err**2)
                    response_var = np.var(syn_efficacy[n,t,trial_ind])
                    tuning_results['synaptic_pev'][n,r,rf,t] = 1 - mse/(response_var + epsilon)
                    tuning_results['synaptic_pref_dir'][n,r,rf,t] = np.arctan2(weights[2,0],weights[1,0])

                    if calculate_test:
                        weights = np.linalg.lstsq(test_dir[trial_ind,:,rf], syn_efficacy[n,t,trial_ind])
                        weights = np.reshape(weights[0],(3,1))
                        pred_err = syn_efficacy[n,t,trial_ind] - np.dot(test_dir[trial_ind,:,rf], weights).T
                        mse = np.mean(pred_err**2)
                        response_var = np.var(syn_efficacy[n,t,trial_ind])
                        tuning_results['synaptic_pev_test'][n,r,rf,t] = 1 - mse/(response_var + epsilon)
                        tuning_results['synaptic_pref_dir_test'][n,r,rf,t] = np.arctan2(weights[2,0],weights[1,0])

        if par['suppress_analysis']:


            x = np.split(trial_info['neural_input'][:,:,trial_ind],num_time_steps,axis=1)
            y = trial_info['desired_output'][:,:,trial_ind]
            train_mask = np.array(mask[:,trial_ind])
            syn_x_init = np.array(syn_x[:,0,trial_ind])
            syn_u_init = np.array(syn_u[:,0,trial_ind])
            hidden_init = np.array(h[:,0,trial_ind])

            y_hat, _, _, _ = run_model(x, hidden_init, syn_x_init, syn_u_init, network_weights)
            acc, acc_non_match, acc_match = get_perf(y, y_hat, train_mask)
            tuning_results['accuracy_no_suppression'] = np.array([acc, acc_non_match, acc_match])

            for k in range(len(suppression_time_range)):
                for k1 in range(len(neuron_groups)):

                    suppress_activity = np.ones((par['n_hidden'], num_time_steps))
                    for m1 in neuron_groups[k1]:
                        for m2 in suppression_time_range[k]:
                            suppress_activity[m1,m2] = 0

                    suppress_activity = np.split(suppress_activity, num_time_steps, axis=1)

                    y_hat, _, syn_x_sim, syn_u_sim = run_model(x, hidden_init, syn_x_init, \
                        syn_u_init, network_weights, suppress_activity = suppress_activity)
                    acc, acc_non_match, acc_match = get_perf(y, y_hat, train_mask)
                    tuning_results['acc_neuronal_suppression'][r,k,k1,:] = np.array([acc, acc_non_match, acc_match])

                    syn_efficacy = syn_x_sim*syn_u_sim
                    for hidden_num in range(par['n_hidden']):
                        for t1 in range(num_time_steps):
                            weights = np.linalg.lstsq(test_dir[trial_ind,:,0], syn_efficacy[hidden_num,t1,trial_ind])
                            weights = np.reshape(weights[0],(3,1))
                            pred_err = syn_efficacy[hidden_num,t1,trial_ind] - np.dot(test_dir[trial_ind,:,0], weights).T
                            mse = np.mean(pred_err**2)
                            response_var = np.var(syn_efficacy[hidden_num,t1,trial_ind])
                            tuning_results['synaptic_pev_test_suppression'][r,k,k1, hidden_num,t1] = 1 - mse/(response_var+1e-9)
                            tuning_results['synaptic_pref_dir_test_suppression'][r,k,k1,hidden_num,t1] = np.arctan2(weights[2,0],weights[1,0])



    return tuning_results


def run_model(x, hidden_init_org, syn_x_init_org, syn_u_init_org, weights, suppress_activity = None):

    """
    Run the reccurent network
    History of hidden state activity stored in self.hidden_state_hist
    """

    # copying data to ensure nothing gets changed upstream
    hidden_init = copy.deepcopy(hidden_init_org)
    syn_x_init = copy.deepcopy(syn_x_init_org)
    syn_u_init = copy.deepcopy(syn_u_init_org)

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

    """
    Update the hidden state
    All needed rectification has already occured
    """

    h = np.maximum(0, h*(1-par['alpha_neuron'])
                   + par['alpha_neuron']*(np.dot(np.maximum(0,weights['w_in']), np.maximum(0, rnn_input))
                   + np.dot(W_rnn_effective, h_post) + weights['b_rnn'])
                   + np.random.normal(0, par['noise_rnn'],size=(par['n_hidden'], h.shape[1])))

    h *= suppress_activity

    return h, syn_x_new, syn_u_new


def get_perf(y, y_hat, mask):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when y[0,:,:] is not 0
    y is the desired output
    y_hat is the actual output
    """
    mask = np.float32(mask > 0)
    y_hat_max = np.stack(y_hat, axis=1)
    y_hat_max = np.argmax(y_hat_max, axis = 0)
    mask_test = mask*(y[0,:,:]==0)
    mask_non_match = mask*(y[1,:,:]==1)
    mask_match = mask*(y[2,:,:]==1)
    y_max = np.argmax(y, axis = 0)
    accuracy = np.sum(np.float32(y_max == y_hat_max)*mask_test)/np.sum(mask_test)

    accuracy_non_match = np.sum(np.float32(y_max == y_hat_max)*np.squeeze(mask_non_match))/np.sum(mask_non_match)
    accuracy_match = np.sum(np.float32(y_max == y_hat_max)*np.squeeze(mask_match))/np.sum(mask_match)

    return accuracy, accuracy_non_match, accuracy_match
