import tensorflow as tf
import numpy as np
from parameters import *
from model_saver import *
import time
import os


###########################
### Model Data Routines ###
###########################

def initialize_test_data():

    N = par['batch_train_size']*par['num_test_batches']

    test_data = {
        'loss'          : np.zeros((par['num_test_batches']), dtype=np.float32),
        'perf_loss'     : np.zeros((par['num_test_batches']), dtype=np.float32),
        'spike_loss'    : np.zeros((par['num_test_batches']), dtype=np.float32),
        'dend_loss'     : np.zeros((par['num_test_batches']), dtype=np.float32),
        'omega_loss'    : np.zeros((par['num_test_batches']), dtype=np.float32),
        'mean_hidden'   : np.zeros((par['num_test_batches']), dtype=np.float32),
        'accuracy'      : np.zeros((par['num_test_batches']), dtype=np.float32),

        'y'             : np.zeros((par['num_test_batches'], par['num_time_steps'], par['n_output'], par['batch_train_size'])),
        'y_hat'         : np.zeros((par['num_test_batches'], par['n_output'], par['num_time_steps'], par['batch_train_size'])),
        'train_mask'    : np.zeros((par['num_test_batches'], par['num_time_steps'], par['batch_train_size'])),

        'sample_index'  : np.zeros((N, par['num_RFs']), dtype=np.uint8),
        'location_index': np.zeros((N, 1), dtype=np.uint8),
        'rule_index'    : np.zeros((N, 1), dtype=np.uint8),
        'state_hist'    : np.zeros((par['num_time_steps'], par['n_hidden'], N), dtype=np.float32)
    }

    if par['use_dendrites']:
        test_data['dend_hist'] = np.zeros((par['num_time_steps'], par['n_hidden'], par['den_per_unit'], N), dtype=np.float32)
        test_data['dend_exc_hist'] = np.zeros((par['num_time_steps'], par['n_hidden'], par['den_per_unit'], N), dtype=np.float32)
        test_data['dend_inh_hist'] = np.zeros((par['num_time_steps'], par['n_hidden'], par['den_per_unit'], N), dtype=np.float32)

    return test_data


def initialize_model_results():
    model_results = {'accuracy': [], 'rule_accuracy' : [], 'modularity': [], 'loss': [], 'perf_loss': [], \
                 'spike_loss': [], 'dend_loss': [], 'omega_loss': [], 'mean_hidden': [], 'trial': [], 'time': []}

    return model_results


def append_model_performance(model_results, test_data, trial_num, iteration_time):

    model_results['loss'].append(np.mean(test_data['loss']))
    model_results['spike_loss'].append(np.mean(test_data['spike_loss']))
    model_results['perf_loss'].append(np.mean(test_data['perf_loss']))
    model_results['dend_loss'].append(np.mean(test_data['dend_loss']))
    model_results['omega_loss'].append(np.mean(test_data['omega_loss']))
    model_results['mean_hidden'].append(np.mean(test_data['mean_hidden']))
    model_results['trial'].append(trial_num)
    model_results['time'].append(iteration_time)

    return model_results


def append_analysis_vals(model_results, analysis_val):

    for k in analysis_val.keys():
        if k == 'accuracy':
            model_results['accuracy'].append(analysis_val['accuracy'])
        elif k == 'rule_accuracy':
            model_results['rule_accuracy'].append(analysis_val['rule_accuracy'])
        elif k == 'modularity':
            model_results['modularity'].append(analysis_val['modularity'])
        elif not analysis_val[k] == []:
            for k1,v in analysis_val[k].items():
                current_key = k + '_' + k1
                if not current_key in model_results.keys():
                    model_results[current_key] = [v]
                else:
                    model_results[current_key].append([v])

    return model_results


def append_test_data(test_data, trial_info, state_hist_batch, dend_hist_batch, dend_exc_hist_batch, dend_inh_hist_batch, batch_num):

    trial_ind = range(batch_num*par['batch_train_size'], (batch_num+1)*par['batch_train_size'])

    # add stimulus information
    test_data['sample_index'][trial_ind,:] = trial_info['sample_index']
    test_data['rule_index'][trial_ind] = trial_info['rule_index']
    test_data['location_index'][trial_ind] = trial_info['location_index']

    # add neuronal activity
    test_data['state_hist'][:,:,trial_ind] = state_hist_batch
    if par['use_dendrites']:
        test_data['dend_hist'][:,:,:,trial_ind] = dend_hist_batch
        test_data['dend_exc_hist'][:,:,:,trial_ind] = dend_exc_hist_batch
        test_data['dend_inh_hist'][:,:,:,trial_ind] = dend_inh_hist_batch

    return test_data


def extract_weights():

    with tf.variable_scope('parameters', reuse=True):
        if par['use_dendrites']:
            with tf.variable_scope('dendrite'):
                W_rnn_dend  = tf.get_variable('W_rnn_dend')
                W_stim_dend = tf.get_variable('W_stim_dend')

                W_td_dend   = tf.get_variable('W_td_dend')
        if par['use_stim_soma']:
            with tf.variable_scope('soma'):
                W_rnn_soma  = tf.get_variable('W_rnn_soma')
                W_stim_soma = tf.get_variable('W_stim_soma')
                W_td_soma   = tf.get_variable('W_td_soma')
        else:
            W_stim_soma     = np.zeros([par['n_hidden'], par['num_stim_tuned']], dtype=np.float32)
            W_td_soma       = np.zeros([par['n_hidden'], par['n_input'] - par['num_stim_tuned']], dtype=np.float32)

        with tf.variable_scope('standard'):
            W_out           = tf.get_variable('W_out')
            b_out           = tf.get_variable('b_out')
            b_rnn           = tf.get_variable('b_rnn')

    weights = {
        'w_rnn_soma': W_rnn_soma.eval(),
        'w_out': W_out.eval(),
        'b_rnn': b_rnn.eval(),
        'b_out': b_out.eval()
        }

    if par['use_dendrites']:
        weights['w_stim_dend'] = W_stim_dend.eval()
        weights['w_td_dend'] = W_td_dend.eval()
        weights['w_rnn_dend'] = W_rnn_dend.eval()

    if par['use_stim_soma']:
        weights['w_stim_soma'] = W_stim_soma.eval()
        weights['w_td_soma'] = W_td_soma.eval()

    return weights


#########################
### General Utilities ###
#########################

def create_save_dir():

    # Generate an identifying timestamp and save directory for the model
    timestamp = "_D" + time.strftime("%y-%m-%d") + "_T" + time.strftime("%H-%M-%S")
    if par['use_dendrites']:
        dirpath = './savedir/model_' + par['stimulus_type'] + '_h' + \
            str(par['n_hidden']) + '_df' + par['df_num'] + timestamp + par['save_notes']
    else:
        dirpath = './savedir/model_' + par['stimulus_type'] + '_h' + \
            str(par['n_hidden']) + 'nd' + timestamp + par['save_notes']

    # Make new folder for parameters, results, and analysis
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # Store a copy of the parameters setup in its default state
    json_save(par, dirpath + '/parameters.json')

    # Create summary file
    with open(dirpath + '/model_summary.txt', 'w') as f:
        f.write('Trial\tTime\tPerf loss\tSpike loss\tDend loss\tOmega loss\tMean activity\tModularity\tCommunities\tAccuracy\tRule Accuracies\n')

    return timestamp, dirpath


def create_placeholders(general, default=False):
    g = []
    if not default:
        for p in general:
            g.append(tf.placeholder(tf.float32, shape=p[1]))
    else:
        for p in general:
            g.append(tf.placeholder_with_default(np.zeros(p[1], dtype=np.float32), shape=p[1]))
    return g


def zip_to_dict(g, s):
    r = {}
    if len(g) == len(s):
        for i in range(len(g)):
            r[g[i]] = s[i]
    else:
        print("ERROR: Lists in zip_to_dict must be of same size")
        quit()

    return r


def split_list(l):
    return l[:len(l)//2], l[len(l)//2:]


def get_vars_in_scope(scope_name):
    return sort_tf_vars(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name))


def sort_tf_vars(var_list):
    return sorted(var_list, key=lambda var: var.shape.as_list())


def print_tf_var_list(var_list):
    for v in var_list:
        print(v)
    print('')
    return 0


def sort_grads_and_vars(var_list):
    return sorted(var_list, key=lambda var: var[0].shape)

def sort_tf_grads_and_vars(var_list):
    return sorted(var_list, key=lambda var: var[0].shape.as_list())


def intersection_by_shape(l1, l2, flag=None):
    if flag == None:
        v1 = list(map(lambda w: w.shape.as_list(), l1))
        v2 = list(map(lambda w: w.shape.as_list(), l2))
        i1 = [i for i in range(len(v1)) if v1[i] in v2]
        i2 = [i for i in range(len(v2)) if v2[i] in v1]
        return [l1[n] for n in i1], [l2[n] for n in i2]
    elif flag == 'meta':
        v1 = list(map(lambda w: w.shape.as_list(), l1))
        v2 = list(map(lambda w: w.shape.as_list()[:-1], l2))
        i1 = [i for i in range(len(v1)) if v1[i] in v2]
        i2 = [i for i in range(len(v2)) if v2[i] in v1]
        return [l1[n] for n in i1], [l2[n] for n in i2]


def filter_adams(var_list):
    new_list = []
    for var in var_list:
        if 'Adam' in var.name:
            pass
        else:
            new_list.append(var)
    return new_list


#######################
### Console outputs ###
#######################

def print_data(dirpath, model_results, analysis):

    rule_accuracies = ''
    for a in range(len(model_results['rule_accuracy'][-1])):
        rule_accuracies += ('\t{0:4f}'.format(model_results['rule_accuracy'][-1][a]))

    with open(dirpath + '/model_summary.txt', 'a') as f:
        # In order, Trial | Time | Perf Loss | Spike Loss | Mean Activity | Accuracy | Rule Accuracy
        f.write('{:7d}'.format(model_results['trial'][-1]) \
            + '\t{:0.2f}'.format(model_results['time'][-1]) \
            + '\t{:0.4f}'.format(model_results['perf_loss'][-1]) \
            + '\t{:0.4f}'.format(model_results['spike_loss'][-1]) \
            + '\t{:0.4f}'.format(model_results['dend_loss'][-1]) \
            + '\t{:0.4f}'.format(model_results['omega_loss'][-1])
            + '\t{:0.4f}'.format(model_results['mean_hidden'][-1]) \
            #+ '\t{:0.4f}'.format(model_results['modularity'][-1]['mod']) \
            #+ '\t{:0.4f}'.format(model_results['modularity'][-1]['community']) \
            + '\t{:0.4f}'.format(model_results['accuracy'][-1]) \
            + rule_accuracies + '\n')

    # output model performance to screen
    print('\nIteration Summary:')
    print('------------------')
    print('Trial: {:13.0f} | Time: {:12.2f} s | Accuracy: {:13.4f}'.format( \
    model_results['trial'][-1], model_results['time'][-1], model_results['accuracy'][-1]))
    print('Perf. Loss: {:8.4f} | Dend. Loss: {:8.4f} | Mean Activity: {:8.4f}'.format( \
        model_results['perf_loss'][-1], model_results['dend_loss'][-1], model_results['mean_hidden'][-1]))
    print('Spike Loss: {:8.4f} | Omega Loss: {:8.4f} | '.format(model_results['spike_loss'][-1], model_results['omega_loss'][-1]))

    if par['stimulus_type'] == 'multitask':
        print('')
        print('Attention Accuracies:'.ljust(22) + str(np.round(model_results['rule_accuracy'][-1][0:2], 2)))
        print('DMC Accuracies:'.ljust(22) + str(np.round(model_results['rule_accuracy'][-1][2:4], 2)))
        print('DMRS Accuracies:'.ljust(22) + str(np.round(model_results['rule_accuracy'][-1][4:], 2)))
    else:
        print('\nRule Accuracies:\t', np.round(model_results['rule_accuracy'][-1], 2))

    if par['modul_vars']:
        print('\nModularity:')
        print('-----------')
        print('Modularity value'.ljust(22) + ': {:5.3f} '.format(model_results['modularity'][-1]['mod']))
        print('Number of communities'.ljust(22) + ': {:5.3f} '.format(model_results['modularity'][-1]['community']))
        print('Community size'.ljust(22) + ': {:5.3f} +/- {:5.3f} '.format(model_results['modularity'][-1]['mean'], model_results['modularity'][-1]['std']))
        print(''.ljust(22) + ': Max {:5.3f}, min {:5.3f} '.format(model_results['modularity'][-1]['max'], model_results['modularity'][-1]['min']))

    if par['anova_vars'] is not None:
        anova_print = [k[:-5].ljust(22) + ':  {:5.3f} '.format(np.mean(v<0.001)) for k,v in analysis['anova'].items() if k.count('pval')>0]
        print('\nAnova P < 0.001:')
        print('----------------')
        for i in range(0, len(anova_print), 2):
            print(anova_print[i] + "\t| " + anova_print[i+1])
        if len(anova_print)%2 != 0:
            print(anova_print[-1] + "\t|")
    if par['roc_vars'] is not None:
        roc_print = [k[:-5].ljust(22) + ':  {:5.3f} '.format(np.percentile(np.abs(v), 98)) for k,v in analysis['roc'].items()]
        print('\n98th prctile t-stat:')
        print('--------------------')
        for i in range(0, len(roc_print), 2):
            print(roc_print[i] + "\t| " + roc_print[i+1])
        if len(roc_print)%2 != 0:
            print(roc_print[-1] + "\t|")
    print("\n")


def print_startup_info():
    print('Using dendrites:\t', par['use_dendrites'])
    print('Using EI network:\t', par['EI'])
    print('Input-soma connection:\t', par['use_stim_soma'])
    print('Synaptic configuration:\t', par['synapse_config'], '\n')

    print("="*77)
    print('Stimulus type'.ljust(22) + ': ' + par['stimulus_type'].ljust(10) + '| ' + 'Loss function'.ljust(22) + ': ' + par['loss_function'])
    print('Num. stim. neurons'.ljust(22) + ': ' + str(par['num_stim_tuned']).ljust(10) + '| ' + 'Num. fix. neurons'.ljust(22) + ': ' + str(par['num_fix_tuned']))
    print('Num. rule neurons'.ljust(22) + ': ' + str(par['num_rule_tuned']).ljust(10) + '| ' + 'Num. spatial neurons'.ljust(22) + ': ' + str(par['num_spatial_cue_tuned']))
    print('Num. hidden neurons'.ljust(22) + ': ' + str(par['n_hidden']).ljust(10) + '| ' + 'Time step'.ljust(22) + ': ' + str(par['dt']) + ' ms')

    print('Num. dendrites'.ljust(22) + ': ' + str(par['den_per_unit']).ljust(10) + '| ' + 'Dendrite function'.ljust(22) + ': ' + str(par['df_num']))
    print('Spike cost'.ljust(22) + ': ' + str(par['spike_cost']).ljust(10) + '| ' + 'Dendrite cost'.ljust(22) + ': ' + str(par['dend_cost']))
    print('Stimulus noise'.ljust(22) + ': ' + str(np.round(par['input_sd'], 2)).ljust(10) + '| ' + 'Internal noise'.ljust(22) + ': ' + str(np.round(par['noise_sd'], 2)))
    print('Batch size'.ljust(22) + ': ' + str(par['batch_train_size']).ljust(10) + '| ' + 'Num. train batches'.ljust(22) + ': ' + str(par['num_train_batches']))
    print('Switch iteration'.ljust(22) + ': ' + str(par['switch_rule_iteration']).ljust(10) + '| ' + 'Num. test batches'.ljust(22) + ': ' + str(par['num_test_batches']))
    print("="*77 + '\n')
