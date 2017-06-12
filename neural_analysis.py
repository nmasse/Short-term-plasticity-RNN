import numpy as np
import pickle
import contrib_to_behavior
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import svm

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "arial"

class neural_analysis:
    
    def __init__(self, model_filename, ABBA=False, old_format = False):
        
        x = pickle.load(open(model_filename, 'rb'))
        self.ABBA = ABBA
        
        # reshape STP depression
        self.syn_x = np.stack(x['syn_x'],axis=2)
        self.syn_x = np.stack(self.syn_x,axis=1)
        if self.syn_x.shape[0] == 0:
            self.syn_x = None
        else:
            num_neurons, trial_length, num_blocks, trials_per_block = self.syn_x.shape
            self.syn_x = np.reshape(self.syn_x,(num_neurons,trial_length,num_blocks*trials_per_block))
        
        # reshape STP facilitation
        self.syn_u = np.stack(x['syn_u'],axis=2)
        self.syn_u = np.stack(self.syn_u,axis=1)
        if self.syn_u.shape[0] == 0:
            self.syn_u = None
        else:
            num_neurons, trial_length, num_blocks, trials_per_block = self.syn_u.shape
            self.syn_u = np.reshape(self.syn_u,(num_neurons,trial_length,num_blocks*trials_per_block))
        
        
        # reshape RNN outputs
        self.rnn_outputs = np.stack(x['hidden_state'],axis=2)
        self.rnn_outputs = np.stack(self.rnn_outputs,axis=1)
        num_neurons, trial_length, num_blocks, trials_per_block = self.rnn_outputs.shape
        self.rnn_outputs = np.reshape(self.rnn_outputs,(num_neurons,trial_length,num_blocks*trials_per_block))
        
        # reshape desired outputs
        self.desired_outputs = x['desired_output']
        if old_format:
            self.desired_outputs = np.transpose(self.desired_outputs,(2,0,1))
        
        # reshape train_mask
        self.train_mask = x['train_mask']
        self.train_mask = np.transpose(self.train_mask,(0,1))

        
        # reshape RNN inputs
        self.rnn_inputs = x['rnn_input']
        self.rnn_inputs = np.transpose(self.rnn_inputs,(2,0,1))

        
        # reshape model outputs
        self.model_outputs = np.stack(x['model_outputs'],axis=2)
        self.model_outputs = np.stack(self.model_outputs,axis=1)
        num_classes = self.model_outputs.shape[0]
        self.model_outputs = np.reshape(self.model_outputs,(num_classes,trial_length,num_blocks*trials_per_block))
        
        """
        rnn_inputs, desired_outputs, rnn_outputs, model_outputs
        should be of shape neurons X time X trials
        print(self.rnn_inputs.shape, self.desired_outputs.shape,self.rnn_outputs.shape,self.model_outputs.shape, self.train_mask.shape)
        """

        # reshape trial_conds
        self.sample_dir = x['sample_dir']
        self.test_dir = x['test_dir']
        self.match = x['match']
        self.rule = x['rule']
        self.catch = x['catch']
        self.probe = x['probe']
        # for the ABBA trials
        if self.ABBA:
            self.num_test_stim = x['num_test_stim']
            self.repeat_test_stim = x['repeat_test_stim']
            self.ABBA_delay = x['params']['ABBA_delay']
        
        # other info
        #self.EI_list = x['params']['EI_list']
        self.num_rules = len(x['params']['possible_rules'])
        self.possible_rules = x['params']['possible_rules']
        self.num_motion_dirs = x['params']['num_motion_dirs']
        self.U = x['U']
        self.W_rnn = x['w_rnn']
        self.b_rnn = x['b_rnn']
        self.W_in = x['w_in']
        self.EI_list = x['params']['EI_list']
        self.dead_time = x['params']['dead_time']
        self.fix_time = x['params']['fix_time']
        self.delta_t = x['params']['dt']
        if self.ABBA:
            self.max_num_tests = x['params']['max_num_tests']
            self.ABBA_accuracy_match, self.ABBA_accuracy_non_match = self.performance_ABBA()
            
        else:
            pass
            #accuracy = self.performance()
            #print(accuracy)
           
        
    def calc_native_tuning(self):
        
        rule = 0
        sample_rng = range(8+20,8+20+20)
        #sample_rng = range(88,108)
        num_dirs = self.num_motion_dirs
        num_input_neurons, trial_length, num_trials = self.rnn_inputs.shape
        mean_input_resp = np.zeros((num_input_neurons, num_dirs))
        num_rnn_neurons = self.rnn_outputs.shape[0]
        native_tuning = np.zeros((num_rnn_neurons, num_dirs))

        for d in range(num_dirs):
            ind = np.where((self.rule == self.possible_rules[rule])*(self.sample_dir==d))
            #ind = np.where((self.rule == self.possible_rules[rule])*(self.test_dir==d))
            s = np.mean(self.rnn_inputs[:,:,ind[0]],axis=2)
            mean_input_resp[:,d] = np.mean(s[:,sample_rng],axis=1)

        native_tuning = np.dot(self.W_in, mean_input_resp)
        
        return native_tuning
        
    def motion_tuning(self):
        
        num_neurons, trial_length, num_trials = self.rnn_outputs.shape
        sample_pd = np.zeros((num_neurons, trial_length))
        sample_pev = np.zeros((num_neurons, trial_length))
        sample_amp = np.zeros((num_neurons, trial_length))
        test_pd = np.zeros((num_neurons, 2, trial_length))
        test_pev = np.zeros((num_neurons, 2, trial_length))
        test_amp = np.zeros((num_neurons, 2, trial_length))
        
        sample_dir = np.ones((num_trials, 3))
        sample_dir[:,1] = np.cos(2*np.pi*self.sample_dir/self.num_motion_dirs)  
        sample_dir[:,2] = np.sin(2*np.pi*self.sample_dir/self.num_motion_dirs) 
        
        test_dir = np.ones((num_trials, 3))
        test_dir[:,1] = np.cos(2*np.pi*self.test_dir/self.num_motion_dirs)  
        test_dir[:,2] = np.sin(2*np.pi*self.test_dir/self.num_motion_dirs) 

        for n in range(num_neurons):
            for t in range(trial_length):
                h = np.linalg.lstsq(sample_dir, self.rnn_outputs[n,t,:])
                pred_err = self.rnn_outputs[n,t,:] - np.dot(h[0], sample_dir.T)
                mse = np.mean(pred_err**2)
                response_var = np.var(self.rnn_outputs[n,t,:])
                sample_pev[n,t] = 1 - (mse)/(response_var+1e-9)
                sample_pd[n,t] = np.arctan2(h[0][2],h[0][1])
                sample_amp[n,t] = np.sqrt(h[0][0]**2+h[0][1]**2)
                
                for m in range(2):
                    ind = np.where(self.match==m)[0]
                
                    h = np.linalg.lstsq(test_dir[ind], self.rnn_outputs[n,t,ind])
                    pred_err = self.rnn_outputs[n,t,ind] - np.dot(h[0], test_dir[ind].T)
                    mse = np.mean(pred_err**2)
                    response_var = np.var(self.rnn_outputs[n,t,ind])
                    test_pev[n,m,t] = 1 - (mse)/(response_var+1e-9)
                    test_pd[n,m,t] = np.arctan2(h[0][2],h[0][1])
                    test_amp[n,m,t] = np.sqrt(h[0][0]**2+h[0][1]**2)
                
        return sample_pd, sample_pev, sample_amp, test_pd, test_pev, test_amp
    
    def recreate_effective_weight_matrix(self, EI = False):
        
        rule = 0
        num_neurons, trial_length, num_trials = self.syn_u.shape
        W = np.zeros((num_neurons,num_neurons,self.num_motion_dirs, trial_length))
        
        mean_efficacy = np.zeros((num_neurons,self.num_motion_dirs, trial_length))
        
        for d in range(self.num_motion_dirs):
            ind = np.where((self.rule == self.possible_rules[rule])*(self.sample_dir==d)*(self.match==1))[0]
            mean_efficacy[:,d,:] = np.mean(self.syn_u[:,:,ind]*self.syn_x[:,:,ind],axis=2)
            
        if EI:
            ei_diag = np.diag(self.EI_list)
            W_rnn = np.dot(np.maximum(0,self.W_rnn), ei_diag)
        else:
            W_rnn = self.W_rnn
                
        for n1 in range(num_neurons):
            for n2 in range(num_neurons):
                for d in range(self.num_motion_dirs):
                    W[n1,n2,d,:] = mean_efficacy[n2,d,:]*W_rnn[n1,n2]        
                
        return W
    
    
    def recreate_output_current(self, EI = False):
        
        rule = 0
        num_neurons, trial_length, num_trials = self.syn_u.shape
        out_current = np.zeros((num_neurons,self.num_motion_dirs, self.num_motion_dirs, trial_length))
        
        out_current = np.zeros((num_neurons,self.num_motion_dirs, self.num_motion_dirs, trial_length))
        
        for s in range(self.num_motion_dirs):
            for t in range(self.num_motion_dirs):
                ind = np.where((self.rule == self.possible_rules[rule])*(self.sample_dir==s)*(self.test_dir==t))[0]
                out_current[:,s,t,:] = np.mean(self.syn_u[:,:,ind]*self.syn_x[:,:,ind]*self.rnn_outputs[:,:,ind],axis=2)
                
        """
            
        if EI:
            ei_diag = np.diag(self.EI_list)
            W_rnn = np.dot(np.maximum(0,self.W_rnn), ei_diag)
        else:
            W_rnn = self.W_rnn
                
        for n1 in range(num_neurons):
            for n2 in range(num_neurons):
                for s in range(self.num_motion_dirs):
                    for t in range(self.num_motion_dirs):
                        out_current[n1,n2,s,t,:] = post_syn[n2,s,t,:]*W_rnn[n1,n2]        
        """        
        return out_current
        
        
        
            
        
    def performance(self):
        
        n = 18 # number of time steps to measure during test, this will be the basis of performance
        time_correct = np.zeros((self.num_rules, self.num_motion_dirs, 2))
        count = np.zeros((self.num_rules, self.num_motion_dirs, 2))
        for i in range(len(self.sample_dir)):
            if self.catch[i]==0:
                s = np.int_(self.sample_dir[i])
                m = np.int_(self.match[i])
                r = np.int_(np.where(self.rule[i]==self.possible_rules)[0])
                count[r,s,m] +=1
                if m==1:
                    score=np.mean((self.model_outputs[2,-n:,i]>self.model_outputs[1,-n:,i])*(self.model_outputs[2,-n:,i]>self.model_outputs[0,-n:,i]))
                else:
                    score=np.mean((self.model_outputs[1,-n:,i]>self.model_outputs[2,-n:,i])*(self.model_outputs[1,-n:,i]>self.model_outputs[0,-n:,i]))  
                time_correct[r,s,m] += score
                
        return time_correct/count
    
    
    def performance_ABBA(self):
        
        ABBA_delay = self.ABBA_delay//self.delta_t
        eof = (self.dead_time+self.fix_time)//self.delta_t
        eos = eof + ABBA_delay
        
        # performance is measured with and without a repeated distractor
        time_correct_match = np.zeros((self.max_num_tests))  
        time_correct_non_match = np.zeros((self.max_num_tests))
        time_match = np.zeros((self.max_num_tests))  
        time_non_match = np.zeros((self.max_num_tests))
        
        for i in range(len(self.sample_dir)):
            for j in range(self.num_test_stim[i]):
                # will discard the first time point of each test stim
                test_rng = range(1+eos+(2*j+1)*ABBA_delay, eos+(2*j+2)*ABBA_delay)
                matching_stim = self.match[i]==1 and j==self.num_test_stim[i]-1
                
                if matching_stim:
                    time_match[j] +=  ABBA_delay-1  # -1 because we're discarding the first time point of each test stim
                    time_correct_match[j] +=  np.sum((self.model_outputs[2,test_rng,i]>self.model_outputs[1,test_rng,i])*(self.model_outputs[2,test_rng,i]>self.model_outputs[0,test_rng,i]))
                else:
                    time_non_match[j] +=  ABBA_delay-1
                    time_correct_non_match[j] +=  np.sum((self.model_outputs[1,test_rng,i]>self.model_outputs[2,test_rng,i])*(self.model_outputs[1,test_rng,i]>self.model_outputs[0,test_rng,i]))
                
            
        auccracy_match = time_correct_match/time_match
        auccracy_non_match = time_correct_non_match/time_non_match
        
        print('Accuracy')
        print(time_correct_non_match, time_non_match)
        print(time_correct_match, time_match)

        return auccracy_match, auccracy_non_match
                
            
        
        
    def show_results(self):
        print(self.results)
        
    def plot_example_neurons(self, example_numbers):
        
        mean_resp = calc_mean_responses(self)
        1/0
        f = plt.figure(figsize=(12,8))
        ax = f.add_subplot(1, 3, 1)
        ax.imshow(trial_info['sample_direction'],interpolation='none',aspect='auto')
        ax = f.add_subplot(1, 3, 2)
        ax.imshow(trial_info['test_direction'],interpolation='none',aspect='auto')
        ax = f.add_subplot(1, 3, 3)
        ax.imshow(trial_info['match'],interpolation='none',aspect='auto')
        plt.show()
        1/0  
        
    def calculate_svms(self, num_reps = 3, DMC = [False], decode_test = False):
        
        lin_clf = svm.SVC(C=1,kernel='linear',decision_function_shape='ovr', shrinking=False, tol=1e-4)
        num_neurons, trial_length, num_trials = self.rnn_outputs.shape
        spike_decoding = np.zeros((trial_length,self.num_rules,num_reps))
        synapse_decoding = np.zeros((trial_length,self.num_rules,num_reps))
        spike_decoding_test = np.zeros((trial_length,self.num_rules,num_reps))
        synapse_decoding_test = np.zeros((trial_length,self.num_rules,num_reps))
        N = self.num_motion_dirs
        
        sample_cat = np.floor(self.sample_dir/(self.num_motion_dirs/2)*np.ones_like(self.sample_dir))
        if self.ABBA:
            test_dir = self.test_dir[:,0]
        else:
            test_dir = self.test_dir
        test_cat = np.floor(test_dir/(self.num_motion_dirs/2)*np.ones_like(test_dir))
        
        for r in range(self.num_rules):
            if self.ABBA:
                ind = np.where((self.num_test_stim>=4))[0]
            else:
                ind = np.where((self.rule==self.possible_rules[r]))[0]
            for t in range(trial_length):
                if DMC[r]:
                    spike_decoding[t,r,:] = self.calc_svm_equal_trials(lin_clf,self.rnn_outputs[:,t,ind].T, sample_cat[ind],num_reps,2)
                    if decode_test:
                        spike_decoding_test[t,r,:] = self.calc_svm_equal_trials(lin_clf,self.rnn_outputs[:,t,ind].T, test_cat[ind],num_reps,2)
                else:
                    spike_decoding[t,r,:] = self.calc_svm_equal_trials(lin_clf,self.rnn_outputs[:,t,ind].T, self.sample_dir[ind],num_reps,N)
                    if decode_test:
                        spike_decoding_test[t,r,:] = self.calc_svm_equal_trials(lin_clf,self.rnn_outputs[:,t,ind].T, test_dir[ind],num_reps,N)

                if self.syn_x is not None:
                    effective_current = self.syn_x[:,t,ind].T*self.syn_u[:,t,ind].T
                    if DMC[r]:
                        synapse_decoding[t,r,:] = self.calc_svm_equal_trials(lin_clf,effective_current, sample_cat[ind],num_reps,2)
                        if decode_test:
                            synapse_decoding_test[t,r,:] = self.calc_svm_equal_trials(lin_clf,effective_current, test_cat[ind],num_reps,2)
                    else:
                        synapse_decoding[t,r,:] = self.calc_svm_equal_trials(lin_clf,effective_current, self.sample_dir[ind],num_reps,N)
                        if decode_test:
                            synapse_decoding_test[t,r,:] = self.calc_svm_equal_trials(lin_clf,effective_current, test_dir[ind],num_reps,N)
                    
        
        return spike_decoding, synapse_decoding, spike_decoding_test, synapse_decoding_test
    
    def calculate_autocorr(self, time_start, time_end):
        
        num_neurons, trial_length, num_trials = self.rnn_outputs.shape
        num_lags = time_end-time_start
        spike_autocorr = np.zeros((num_neurons, num_lags))
        syn_x_autocorr = np.zeros((num_neurons, num_lags))
        syn_adapt_autocorr = np.zeros((num_neurons, num_lags))

        for n in range(num_neurons):
            count = np.zeros((num_lags))
            for i in range(time_start, time_end):
                for j in range(time_start, time_end):
                    lag = np.abs(i-j)
                    
                    for s in range(4):
                        ind = np.where(self.sample_dir==s)
                        ind = np.where(self.match==1)
                        ind = ind[0]
                        count[lag] += 1
                        
                        r1 = np.corrcoef(self.rnn_outputs[n,i,ind], self.rnn_outputs[n,j,ind])
                        spike_autocorr[n, lag] += r1[0,1]
                            
                        if self.syn_x is not None:
                            r1 = np.corrcoef(self.syn_x[n,i,ind], self.syn_x[n,j,ind])
                            syn_x_autocorr[n, lag] += r1[0,1]
                                
                        if self.sa is not None:
                            r1 = np.corrcoef(self.sa[n,i,ind], self.sa[n,j,ind])
                            syn_adapt_autocorr[n, lag] += r1[0,1]
                            
            spike_autocorr[n,:] /= count
            syn_x_autocorr[n,:] /= count
            syn_adapt_autocorr[n,:] /= count
    
        return spike_autocorr,syn_x_autocorr,syn_adapt_autocorr
        
        
    def calc_mean_responses(self):
        
        num_rules = self.num_rules
        num_dirs = self.num_motion_dirs
        num_neurons, trial_length, num_trials = self.rnn_outputs.shape
        num_classes = self.model_outputs.shape[0]
        mean_resp = np.zeros((num_neurons, num_rules, num_dirs, trial_length))
        mean_out_match = np.zeros((num_classes, num_rules, trial_length))
        mean_out_non_match = np.zeros((num_classes, num_rules, trial_length))
        
        for n in range(num_neurons):
            for r in range(num_rules):
                for d in range(num_dirs):
                    if self.ABBA:
                        ind = np.where((self.num_test_stim>=4)*(self.sample_dir==d))[0]
                    else:
                        ind = np.where((self.rule == self.possible_rules[r])*(self.sample_dir==d))[0]
                    mean_resp[n,r,d,:] = np.mean(self.rnn_outputs[n,:,ind],axis=0)
                    
         
        for n in range(num_classes):
            for r in range(num_rules):
                ind_match = np.where((self.rule == self.possible_rules[r])*(self.match==1)*(self.catch==0))
                ind_non_match = np.where((self.rule == self.possible_rules[r])*(self.match==0)*(self.catch==0))
                mean_out_match[n,r,:] = np.mean(self.model_outputs[n,:,ind_match[0]],axis=0)
                mean_out_non_match[n,r,:] = np.mean(self.model_outputs[n,:,ind_non_match[0]],axis=0)
                
        return mean_resp, mean_out_match, mean_out_non_match
    
    
    def decoding_accuracy_postle(self, num_reps = 10):
        
        lin_clf = svm.SVC(C=1,kernel='linear',decision_function_shape='ovr', shrinking=False, tol=1e-5)
        
        num_neurons, trial_length, num_trials = self.rnn_outputs.shape
        sample_pev = np.zeros((num_neurons, 2,2,2,2,trial_length))
        sample_stp_pev = np.zeros((num_neurons, 2,2,2,2,trial_length))
        
        sample_decoding = np.zeros((2,2,2,2,trial_length,num_reps))
        sample_stp_decoding = np.zeros((2,2,2,2,trial_length,num_reps))
        
        model_output = np.zeros((2,2,3,trial_length))
        
        
        # r1 and r2 refer to the first and second rule (attention) cue
        # m refers to the modality
        # p refers to the presence or absence of a probe
        
        for m1 in range(2):
            for m2 in range(2):
                ind = np.where((self.match[:,0] == m1)*(self.match[:,1] == m2)*(self.probe[:,1]==0))[0]
                model_output[m1,m2,:,:] = np.mean(self.model_outputs[:,:,ind],axis=2)
        
        for r1 in range(2):
            for r2 in range(2):
                for p in range(2):
                    ind = np.where((self.rule[:,0] == r1)*(self.rule[:,1] == r2)*(self.probe[:,1]==p))[0]
                    #ind = np.where((self.rule[:,0] == r1)*(self.rule[:,1] == r2)*(self.probe[:,1]>=0))[0]
                    for m in range(2):
                        for t in range(trial_length):
                            for n in range(num_neurons):
                                sample_pev[n,r1,r2,p,m,t] = self.calc_pev(self.rnn_outputs[n,t,ind], self.sample_dir[ind,m])
                            sample_decoding[r1,r2,p,m,t,:] = self.calc_svm_equal_trials(lin_clf,self.rnn_outputs[:,t,ind].T, self.sample_dir[ind,m],num_reps, self.num_motion_dirs)
                            if self.syn_x is not None:
                                for n in range(num_neurons):
                                    effective_current = self.syn_x[n,t,ind]*self.syn_u[n,t,ind]
                                    sample_stp_pev[n,r1,r2,p,m,t] = self.calc_pev(effective_current, self.sample_dir[ind,m])
                                effective_current = self.syn_x[:,t,ind]*self.syn_u[:,t,ind]
                                sample_stp_decoding[r1,r2,p,m,t,:] = self.calc_svm_equal_trials(lin_clf,effective_current.T, self.sample_dir[ind,m],num_reps, self.num_motion_dirs)
                   
        
        return sample_pev, sample_stp_pev, sample_decoding, sample_stp_decoding, model_output
    
    @staticmethod
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
            #dec = lin_clf.decision_function(y[test_ind,:])
            dec = lin_clf.predict(y_equal[test_ind,:])
 
            for i in range(len(test_ind)): 
                if conds_equal[test_ind[i]]==dec[i]:
                    score[r] += 1/len(test_ind)
            
           
        return score
        
    @staticmethod
    def calc_svm(lin_clf, y, conds, num_reps):
        
        num_conds = len(np.unique(conds))
        y = np.squeeze(y).T
        # normalize values between 0 and 1
        for i in range(y.shape[1]):
            m1 = y[:,i].min()
            m2 = y[:,i].max()
            y[:,i] -= m1
            if m2>m1:
                y[:,i] /=(m2-m1)
        train_pct = 0.75
        score = np.zeros((num_reps))
        for r in range(num_reps):
            q = np.random.permutation(len(conds))
            i = np.int_(np.round(len(conds)*train_pct))
            train_ind = q[:i]
            test_ind = q[i:]
            
            lin_clf.fit(y[train_ind,:], conds[train_ind])
            dec = lin_clf.decision_function(y[test_ind,:])

            if num_conds>2:
                dec = np.argmax(dec, 1)
            else:
                dec = np.int_(np.sign(dec)*0.5+0.5)
                
            for i in range(len(test_ind)): 
                if conds[test_ind[i]]==dec[i]:
                    score[r] += 1/len(test_ind)
            
           
        return score    
        
    def calculate_pevs(self):
        
        num_neurons, trial_length, num_trials = self.rnn_outputs.shape
        sample_pev = np.zeros((num_neurons, self.num_rules,trial_length))
        test_pev = np.zeros((num_neurons, self.num_rules,trial_length))
        rule_pev = np.zeros((num_neurons,trial_length))
        match_pev = np.zeros((num_neurons, self.num_rules,trial_length))
        sample_stp_pev = np.zeros((num_neurons, self.num_rules,trial_length))
        sample_cat_pev = np.zeros((num_neurons, self.num_rules,trial_length))
        sample_cat_stp_pev = np.zeros((num_neurons, self.num_rules,trial_length))
        test_stp_pev = np.zeros((num_neurons, self.num_rules,trial_length))
        
        for r in range(self.num_rules):
            if self.ABBA:
                ind = np.where((self.num_test_stim>=4))[0]
            else:
                ind = np.where((self.rule == self.possible_rules[r]))[0]
                ind_test = np.where((self.rule == self.possible_rules[r])*(self.match == 0))[0]
                
            for n in range(num_neurons):
                for t in range(trial_length):
                    sample_pev[n,r,t] = self.calc_pev(self.rnn_outputs[n,t,ind], self.sample_dir[ind])
                    sample_cat_pev[n,r,t] = self.calc_pev(self.rnn_outputs[n,t,ind], np.floor(self.sample_dir[ind]/(self.num_motion_dirs/2)))
                
                    if not self.ABBA:
                        test_pev[n,r,t] = self.calc_pev(self.rnn_outputs[n,t,ind_test], self.test_dir[ind_test])
                        rule_pev[n,t] = self.calc_pev(self.rnn_outputs[n,t,:], self.rule)
                        match_pev[n,r,t] = self.calc_pev(self.rnn_outputs[n,t,ind], self.match[ind])
                
                    if self.syn_x is not None:
                        effective_current = self.syn_x[n,t,ind]*self.syn_u[n,t,ind]
                        sample_stp_pev[n,r,t] = self.calc_pev(effective_current, self.sample_dir[ind])
                        if not self.ABBA:
                            test_stp_pev[n,r,t] = self.calc_pev(effective_current, self.test_dir[ind_test])
                            sample_cat_stp_pev[n,r,t] = self.calc_pev(effective_current, np.floor(self.sample_dir[ind]/(self.num_motion_dirs/2)))
        
        return sample_pev, test_pev, rule_pev, match_pev, sample_stp_pev, sample_cat_pev, sample_cat_stp_pev, test_stp_pev
        
    @staticmethod
    def calc_pev(x, conds):
        
        unique_conds = np.unique(conds)
        m = len(unique_conds)
        lx = len(x)
        xr = x - np.mean(x)
        xm = np.zeros((1,m))
        countx = np.zeros((1,m))
        
        for (j,i) in enumerate(unique_conds):
            ind = np.where(conds==i)
            countx[0,j] = len(ind[0])
            xm[0,j] = np.mean(xr[ind[0]])
        gm = np.mean(xr)
        df1 = np.sum(countx>0)-1
        df2 = lx - df1 - 1
        xc = xm - gm
        ix = np.where(countx==0)
        xc[ix] = 0
        RSS = np.dot(countx, np.transpose(xc**2))
        #TSS = (xr - gm)**2
        TSS = np.dot(np.transpose(xr - gm),xr - gm)
        #print(TSS.shape)
        SSE = TSS - RSS
        if df2 > 0:
            mse = SSE/df2
        else:
            mse = np.NaN
        F = (RSS/df1)/mse
        """
        Table = np.zeros((3,5))
        Table[:,0] = [RSS,SSE,TSS]
        Table[:,1] = [df1,df2,df1+df2]
        Table[:,2] = [RSS/df1,mse,999];
        Table[:,3] = [F,999,999]
        """
    
        SS_groups = RSS;
        SS_total = TSS;
        df_groups = df1;
        MS_error = mse;
        pev = (SS_groups-df_groups*MS_error)/(SS_total+MS_error)
        
        if np.isnan(pev):
            pev = 0
            
        return pev
            
            
    def plot_all_figures(self, rule,dt=25, STP=False, DMC = [False], f=None, start_sp=0, num_rows=3, tight=False, two_rules = False, decode_test = False):
        
        font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
        
        mean_resp, mean_out_match, mean_out_non_match = self.calc_mean_responses()
        spike_decode, synapse_decode, spike_decode_test, synapse_decode_test = self.calculate_svms(DMC=DMC,decode_test=decode_test)
        sample_pev, test_pev, rule_pev, _, sample_stp_pev, sample_cat_pev, sample_cat_stp_pev, test_stp_pev  = self.calculate_pevs()
        
        if DMC[0]:
            sample_pev = sample_cat_pev
            sample_stp_pev = sample_cat_stp_pev
            chance_level = 1/2
        else:
            chance_level = 1/8
            
        if two_rules:
            num_cols = 4
        else:
            num_cols = 3
            
        # find good example neuron
        mean_pev = np.mean(sample_pev[:, rule, 30:],axis=1)
        ind = np.argsort(mean_pev)
        example_neuron = ind[-1]
        
        trial_length_steps = sample_pev.shape[2]
        trial_length = np.int_(trial_length_steps*dt)
        
        t = np.arange(0,trial_length,dt)
        t -= 900 # assuming 400 ms dead time, 500 ms fixation
        
        if self.ABBA:
            t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)
        else:
            t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)

        if f is None:   
            f = plt.figure(figsize=(8,2*num_rows))
        
        ax = f.add_subplot(num_rows, num_cols, start_sp+1)
        if self.ABBA:
            # plot accuracy bar plot instead
            
            x=np.array([0,1,2,3])
            ax.bar(x+0.1, self.ABBA_accuracy_match,width=0.2,color='r',align='center')
            ax.bar(x-0.1, self.ABBA_accuracy_non_match,width=0.2,color='b',align='center')
            ax.set_title('Accuracy')
            ax.set_ylabel('Fraction correct')
            ax.set_xlabel('Num. of distractors')
            
        else:    
            ax.hold(True)
            if two_rules:
                ax.plot(t, mean_out_match[0,0,:] ,'k',linewidth=2,label='Fixation')
                ax.plot(t, mean_out_match[1,0,:] ,'m',linewidth=2,label='Non-match')
                ax.plot(t, mean_out_match[2,0,:] ,'g',linewidth=2,label='Match')
                ax.plot(t, mean_out_match[0,1,:] ,'k--',linewidth=2,label='Fixation')
                ax.plot(t, mean_out_match[1,1,:] ,'m--',linewidth=2,label='Non-match')
                ax.plot(t, mean_out_match[2,1,:] ,'g--',linewidth=2,label='Match')
            else:
                ax.plot(t, mean_out_match[0,rule,:] ,'k',linewidth=2,label='Fixation')
                ax.plot(t, mean_out_match[1,rule,:] ,'m',linewidth=2,label='Non-match')
                ax.plot(t, mean_out_match[2,rule,:] ,'g',linewidth=2,label='Match')
            #plt.legend(loc=3)
            self.add_subplot_fixings(ax)
            ax.set_title('Network output - match trials')
        
       
        if self.ABBA:
            pass
        else:
            ax = f.add_subplot(num_rows, num_cols, start_sp+2)
            ax.hold(True)
            if two_rules:
                ax.plot(t, mean_out_non_match[0,0,:] ,'k',linewidth=2,label='Fixation')
                ax.plot(t, mean_out_non_match[1,0,:] ,'m',linewidth=2,label='Non-match')
                ax.plot(t, mean_out_non_match[2,0,:] ,'g',linewidth=2,label='Match')
                ax.plot(t, mean_out_non_match[0,1,:] ,'k--',linewidth=2,label='Fixation')
                ax.plot(t, mean_out_non_match[1,1,:] ,'m--',linewidth=2,label='Non-match')
                ax.plot(t, mean_out_non_match[2,1,:] ,'g--',linewidth=2,label='Match')
            else:
                ax.plot(t, mean_out_non_match[0,rule,:] ,'k',linewidth=2)
                ax.plot(t, mean_out_non_match[1,rule,:] ,'m',linewidth=2)
                ax.plot(t, mean_out_non_match[2,rule,:] ,'g',linewidth=2)
            self.add_subplot_fixings(ax)
            ax.set_title('Network output - non-match trials')
        
        ax = f.add_subplot(num_rows, num_cols, start_sp+3)
        ax.hold(True)
        
        # if plotting the result of the delayed rule task, show rule PEV instead of example neuron
        if two_rules:
            max_val = np.max(rule_pev)
            ax.plot(t,np.mean(rule_pev, axis=0), linewidth=2)
            self.add_subplot_fixings(ax,chance_level=0,ylim=0.2)
            ax.set_title('Rule selectivity')
            ax.set_ylabel('Normalized PEV')
            
        else:
            """
            max_val = np.max(mean_resp[example_neuron,rule,:,:])
            print(max_val)
            for i in range(8):
                ax.plot(t,mean_resp[example_neuron,rule,i,:],color=[1-i/7,0,i/7], linewidth=1)
            self.add_subplot_fixings(ax,chance_level=0,ylim=max_val*1.05)
            ax.set_title('Example neuron')
            ax.set_ylabel('Activity (a.u.)')
            
            # plot the mean population response from those neurons whose synapses are informative of sample
            """
            #syn_pev = np.mean(sample_stp_pev[:,0,t2[0]:t3[0]], axis=1)
            #ind_syn = np.where(syn_pev > 0.1)[0]
            #print('Informative synapses ', ind_syn)
            s = np.mean(mean_resp[:,rule,:,:],axis=0)
            max_val = np.max(s)
            for i in range(8):
                ax.plot(t,s[i,:],color=[1-i/7,0,i/7], linewidth=1)
            self.add_subplot_fixings(ax,chance_level=0,ylim=0.5)
            ax.set_title('Mean response from synpases informative neurons')
            ax.set_ylabel('Activity (a.u.)')
            ax.set_ylim([0, 0.5])
        
        if two_rules:
            ax = f.add_subplot(num_rows, num_cols, start_sp+5)
            im = ax.imshow(sample_pev[:,0,:],aspect='auto',interpolation=None)
            f.colorbar(im,orientation='vertical')
            ax.spines['right'].set_visible(False)
            ax.set_ylabel('Neuron number')
            ax.set_xlabel('Time relative to sample onset (ms)')
            ax.spines['top'].set_visible(False)
            ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
            ax.set_xticklabels([-500,0,500,1500])
            ax.set_title('Neuronal sample \nselectvity - DMS task')
                
            ax = f.add_subplot(num_rows, num_cols, start_sp+6)
            im = ax.imshow(sample_pev[:,1,:],aspect='auto',interpolation=None)
            f.colorbar(im,orientation='vertical')
            ax.spines['right'].set_visible(False)
            ax.set_ylabel('Neuron number')
            ax.set_xlabel('Time relative to sample onset (ms)')
            ax.spines['top'].set_visible(False)
            ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
            ax.set_xticklabels([-500,0,500,1500])
            ax.set_title('Neuronal sample \nselectvity - DMrS task')
        else:
            ax = f.add_subplot(num_rows, 3, start_sp+4)
            im = ax.imshow(sample_pev[:,rule,:],aspect='auto',interpolation=None)
            f.colorbar(im,orientation='vertical')
            ax.spines['right'].set_visible(False)
            ax.set_ylabel('Neuron number')
            ax.set_xlabel('Time relative to sample onset (ms)')
            ax.spines['top'].set_visible(False)
            if DMC:
                ax.set_title('Neuronal sample \ncategory selectvity')
            else:
                ax.set_title('Neuronal sample selectvity')
            if self.ABBA:
                ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
                ax.set_xticklabels([-500,0,500,1500])
            else:
                ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
                ax.set_xticklabels([-500,0,500,1500])
            
        if two_rules:
            ax = f.add_subplot(num_rows, num_cols, start_sp+7)
            plt.hold(True)
            u = np.mean(sample_pev[:,0,:],axis=0)
            se = np.std(sample_pev[:,0,:],axis=0)/np.sqrt(sample_pev.shape[0])
            ax.plot(t,u,'g')
            sample_max1 = np.max(u)
            ax.fill_between(t,u-se,u+se,facecolor=(0,1,0,0.5))
            u = np.mean(sample_pev[:,1,:],axis=0)
            se = np.std(sample_pev[:,1,:],axis=0)/np.sqrt(sample_pev.shape[0])
            ax.plot(t,u,'m')
            sample_max = np.max(u)
            sample_max = np.max([sample_max, sample_max1])
            ax.fill_between(t,u-se,u+se,facecolor=(1,0,1,0.5))
        else:
            ax = f.add_subplot(num_rows, num_cols, start_sp+5)
            u = np.mean(sample_pev[:,rule,:],axis=0)
            se = np.std(sample_pev[:,rule,:],axis=0)/np.sqrt(sample_pev.shape[0])
            ax.plot(t,u,'k')
            sample_max = np.max(u)
            ax.fill_between(t,u-se,u+se,facecolor=(0,0,0,0.5))
        self.add_subplot_fixings(ax,chance_level=0,ylim=sample_max*2)
        if DMC:
            ax.set_title('Neuronal sample \ncategory selectivity')
        else:
            ax.set_title('Neuronal sample selectivity')
        ax.set_ylabel('Normalized PEV')

        if two_rules:
            ax = f.add_subplot(num_rows, num_cols, start_sp+8)
            u = np.mean(spike_decode[:,0,:],axis=1)
            se = np.std(spike_decode[:,0,:],axis=1)
            ax.plot(t,u,'g')
            ax.fill_between(t,u-se,u+se,facecolor=(0,1,0,0.5))
            u = np.mean(spike_decode[:,1,:],axis=1)
            se = np.std(spike_decode[:,1,:],axis=1)
            ax.plot(t,u,'m')
            ax.fill_between(t,u-se,u+se,facecolor=(1,0,1,0.5))
            self.add_subplot_fixings(ax, chance_level=chance_level)
            
        else:
            ax = f.add_subplot(num_rows, num_cols, start_sp+6)
            u = np.mean(spike_decode[:,rule,:],axis=1)
            se = np.std(spike_decode[:,rule,:],axis=1)
            ax.plot(t,u,'k')
            ax.fill_between(t,u-se,u+se,facecolor=(0,0,0,0.5))
            u = np.mean(spike_decode_test[:,rule,:],axis=1)
            se = np.std(spike_decode_test[:,rule,:],axis=1)
            ax.plot(t,u,'c')
            ax.fill_between(t,u-se,u+se,facecolor=(0,1,1,0.5))
            self.add_subplot_fixings(ax, chance_level=chance_level)
        if DMC:
            ax.set_title('Neuronal sample \ncategory decoding')
        else:
            ax.set_title('Neuronal sample decoding')
        ax.set_ylabel('Decoding accuracy')
        
        # add short term plasticity plots
        if STP:
            
            if two_rules:
                ax = f.add_subplot(num_rows, num_cols, start_sp+9)
                im = ax.imshow(sample_stp_pev[:,0,:],aspect='auto',interpolation=None)
                f.colorbar(im,orientation='vertical')
                ax.spines['right'].set_visible(False)
                ax.set_ylabel('Neuron number')
                ax.set_xlabel('Time relative to sample onset (ms)')
                ax.spines['top'].set_visible(False)
                ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
                ax.set_xticklabels([-500,0,500,1500])
                ax.set_title('Synaptic sample \nselectvity - DMS task')
                
                ax = f.add_subplot(num_rows, num_cols, start_sp+10)
                im = ax.imshow(sample_stp_pev[:,1,:],aspect='auto',interpolation=None)
                f.colorbar(im,orientation='vertical')
                ax.spines['right'].set_visible(False)
                ax.set_ylabel('Neuron number')
                ax.set_xlabel('Time relative to sample onset (ms)')
                ax.spines['top'].set_visible(False)
                ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
                ax.set_xticklabels([-500,0,500,1500])
                ax.set_title('Synaptic sample \nselectvity - DMrS task')
                
                
            else:
                ax = f.add_subplot(num_rows, 3, start_sp+7)
                im = ax.imshow(sample_stp_pev[:,rule,:],aspect='auto',interpolation=None)
                f.colorbar(im,orientation='vertical')
                ax.spines['right'].set_visible(False)
                ax.set_ylabel('Neuron number')
                ax.set_xlabel('Time relative to sample onset (ms)')
                ax.spines['top'].set_visible(False)
                if DMC:
                    ax.set_title('Synaptic sample \ncategory selectvity')
                else:
                    ax.set_title('Synaptic sample selectvity')
            if self.ABBA:
                ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
                ax.set_xticklabels([-500,0,500,1500])
            else:
                ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
                ax.set_xticklabels([-500,0,500,1500])
            
            
            
            if two_rules:
                ax = f.add_subplot(num_rows, num_cols, start_sp+11)
                plt.hold(True)
                u = np.mean(sample_stp_pev[:,0,:],axis=0)
                se = np.std(sample_stp_pev[:,0,:],axis=0)/np.sqrt(sample_pev.shape[0])
                ax.plot(t,u,'g')
                ax.fill_between(t,u-se,u+se,facecolor=(0,1,0,0.5))
                u = np.mean(sample_stp_pev[:,1,:],axis=0)
                se = np.std(sample_stp_pev[:,1,:],axis=0)/np.sqrt(sample_pev.shape[0])
                ax.plot(t,u,'m')
                ax.fill_between(t,u-se,u+se,facecolor=(1,0,1,0.5))
                ax.set_title('Synaptic sample selectivity')
                
            else:
        
                ax = f.add_subplot(num_rows, num_cols, start_sp+8)
                u = np.mean(sample_stp_pev[:,rule,:],axis=0)
                se = np.std(sample_stp_pev[:,rule,:],axis=0)/np.sqrt(sample_pev.shape[0])
                ax.plot(t,u,'k')
                ax.fill_between(t,u-se,u+se,facecolor=(0,0,0,0.5))
                
                if DMC:
                    ax.set_title('Synaptic sample \ncategory selectivity')
                else:
                    ax.set_title('Synaptic sample selectivity')
            self.add_subplot_fixings(ax,chance_level=0,ylim=sample_max*2)
            ax.set_ylabel('Normalized PEV')
        
            if two_rules:
                ax = f.add_subplot(num_rows, num_cols, start_sp+12)
                u = np.mean(synapse_decode[:,0,:],axis=1)
                se = np.std(synapse_decode[:,0,:],axis=1)
                ax.plot(t,u,'g')
                ax.fill_between(t,u-se,u+se,facecolor=(0,1,0,0.5))
                u = np.mean(synapse_decode[:,1,:],axis=1)
                se = np.std(synapse_decode[:,1,:],axis=1)
                ax.plot(t,u,'m')
                ax.fill_between(t,u-se,u+se,facecolor=(0.5,0,0.5))
                ax.set_title('Synaptic sample decoding')
                
            else:
                ax = f.add_subplot(num_rows, num_cols, start_sp+9)
                u = np.mean(synapse_decode[:,rule,:],axis=1)
                se = np.std(synapse_decode[:,rule,:],axis=1)
                ax.plot(t,u,'k')
                ax.fill_between(t,u-se,u+se,facecolor=(0,0,0,0.5))
                u = np.mean(synapse_decode_test[:,rule,:],axis=1)
                se = np.std(synapse_decode_test[:,rule,:],axis=1)
                ax.plot(t,u,'c')
                ax.fill_between(t,u-se,u+se,facecolor=(0,1,1,0.5))
                
                if DMC:
                    ax.set_title('Synaptic sample \ncategory decoding')
                else:
                    ax.set_title('Synaptic sample decoding')
            self.add_subplot_fixings(ax, chance_level=chance_level)
            ax.set_ylabel('Decoding accuracy')
            
            if tight:
                plt.tight_layout()
                plt.savefig('DMS summary.pdf', format='pdf')
            plt.show()
            
     
    def plot_postle_figure(self,dt=20, STP=False, tight=False):
        
        # declare that we're analyzing a postle task
        self.postle = True
        
        sample_pev, sample_stp_pev, sample_decoding, sample_stp_decoding, model_output = self.decoding_accuracy_postle(num_reps=10)
        
        t = np.arange(0,220*dt,dt)
        t -= 900 # assuming 400 ms dead time, 500 ms fixation
        t0,t1,t2,t3,t4,t5,t6 = np.where(t==-500), np.where(t==0), np.where(t==500), np.where(t==1000), np.where(t==1500), np.where(t==2000), np.where(t==2500)

        f = plt.figure(figsize=(6,6))
        
        for i in range(2):
            for j in range(2):
                ax = f.add_subplot(3, 2, 2*i+j+1)
                u = np.mean(sample_decoding[i,j,0,0,:,:],axis=1)
                se = np.std(sample_decoding[i,j,0,0,:,:],axis=1)
                ax.fill_between(t,u-se,u+se,facecolor=(0,1,0,0.5))
                ax.plot(t,u,'g')
                u = np.mean(sample_decoding[i,j,0,1,:,:],axis=1)
                se = np.std(sample_decoding[i,j,0,1,:,:],axis=1)
                ax.fill_between(t,u-se,u+se,facecolor=(1,0.6,0,0.5))
                ax.plot(t,u,color=[1,0.6,0])
                
                if STP:
                    u = np.mean(sample_stp_decoding[i,j,0,0,:,:],axis=1)
                    se = np.std(sample_stp_decoding[i,j,0,0,:,:],axis=1)
                    ax.fill_between(t,u-se,u+se,facecolor=(1,0,1,0.5))
                    ax.plot(t,u,'m')
                    
                    u = np.mean(sample_stp_decoding[i,j,0,1,:,:],axis=1)
                    se = np.std(sample_stp_decoding[i,j,0,1,:,:],axis=1)
                    ax.fill_between(t,u-se,u+se,facecolor=(0,1,1,0.5))
                    ax.plot(t,u,'c')
                self.add_subplot_fixings(ax, chance_level=1/8, ylim=1.1)
                ax.set_ylabel('Decoding accuracy')
                    
                    

        ax = f.add_subplot(3, 2, 5)
        u = np.mean(sample_decoding[0,0,1,1,:,:],axis=1)
        se = np.std(sample_decoding[0,0,1,1,:,:],axis=1)
        #u = np.mean(np.mean(sample_decoding[0,:,1,1,:,:],axis=0),axis=1)
        #se = np.std(np.mean(sample_decoding[0,:,1,1,:,:],axis=0),axis=1)
        ax.fill_between(t,u-se,u+se,facecolor=(0,0,0,0.5))
        ax.plot(t,u,'k')
        
        u = np.mean(sample_decoding[0,0,0,1,:,:],axis=1)
        se = np.std(sample_decoding[0,0,0,1,:,:],axis=1)
        #u = np.mean(np.mean(sample_decoding[0,:,0,1,:,:],axis=0),axis=1)
        #se = np.std(np.mean(sample_decoding[0,:,0,1,:,:],axis=0),axis=1)
        ax.fill_between(t,u-se,u+se,facecolor=(1,0.6,0,0.5))
        ax.plot(t,u,color=[1,0.6,0])
        self.add_subplot_fixings(ax, chance_level=1/8, ylim=1.1)
        ax.plot([2400,2400],[-2, 99],'y--')
        ax.set_ylabel('Decoding accuracy')
        

        ax = f.add_subplot(3, 2, 6)
        u = np.mean(sample_decoding[1,1,1,0,:,:],axis=1)
        se = np.std(sample_decoding[1,1,1,0,:,:],axis=1)
        #u = np.mean(np.mean(sample_decoding[1,:,1,0,:,:],axis=0),axis=1)
        #se = np.std(np.mean(sample_decoding[1,:,1,0,:,:],axis=0),axis=1)
        ax.fill_between(t,u-se,u+se,facecolor=(0,0,0,0.5))
        ax.plot(t,u,'k')
        
        u = np.mean(sample_decoding[1,1,0,0,:,:],axis=1)
        se = np.std(sample_decoding[1,1,0,0,:,:],axis=1)
        #u = np.mean(np.mean(sample_decoding[1,:,0,0,:,:],axis=0),axis=1)
        #se = np.std(np.mean(sample_decoding[1,:,0,0,:,:],axis=0),axis=1)
        ax.fill_between(t,u-se,u+se,facecolor=(0,1,0,0.5))
        ax.plot(t,u,'g')
        self.add_subplot_fixings(ax, chance_level=1/8, ylim=1.1)
        ax.plot([2400,2400],[-2, 99],'y--')
        ax.set_ylabel('Decoding accuracy')
            
        plt.tight_layout()
        plt.savefig('postle summary.pdf', format='pdf')    
        plt.show()   
        
        return sample_decoding, sample_stp_decoding
                
    
    def plot_ABBA_figures(self,dt=25, STP=False, tight=False):
        
        
        mean_resp, mean_out_match, mean_out_non_match = self.calc_mean_responses()
        spike_decode, synapse_decode, spike_decode_test, synapse_decode_test = self.calculate_svms(DMC=[False],decode_test=True)
        #sample_pev, test_pev, rule_pev, _, sample_stp_pev, sample_cat_pev, sample_cat_stp_pev, test_stp_pev  = self.calculate_pevs()
        
        chance_level = 1/2

        
        trial_length_steps = self.rnn_outputs.shape[1]
        trial_length = np.int_(trial_length_steps*dt)
        
        t = np.arange(0,trial_length,dt)
        t -= 900 # assuming 400 ms dead time, 500 ms fixation

        t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)

        f = plt.figure(figsize=(6,4))
        
        ax = f.add_subplot(2, 2, 1)
        # plot accuracy bars
        x=np.array([0,1,2,3])
        ax.bar(x+0.1, self.ABBA_accuracy_match,width=0.2,color='r',align='center')
        ax.bar(x-0.1, self.ABBA_accuracy_non_match,width=0.2,color='b',align='center')
        ax.set_title('Accuracy')
        ax.set_ylabel('Fraction correct')
        ax.set_xlabel('Num. of distractors')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
  
        ax = f.add_subplot(2, 2, 2)
        ax.hold(True)
        
        s = np.mean(mean_resp[:,0,:,:],axis=0)
        max_val = np.max(s)
        for i in range(8):
            ax.plot(t,s[i,:],color=[1-i/7,0,i/7], linewidth=1)
        self.add_subplot_fixings(ax,chance_level=0,ylim=0.5)
        ax.set_title('Mean response from synpases informative neurons')
        ax.set_ylabel('Activity (a.u.)')
        ax.set_ylim([0, 0.5])
        
        ax = f.add_subplot(2, 2, 3)
        u = np.mean(spike_decode[:,0,:],axis=1)
        se = np.std(spike_decode[:,0,:],axis=1)
        ax.plot(t,u,'b')
        ax.fill_between(t,u-se,u+se,facecolor=(0,0,1,0.5))
        u = np.mean(spike_decode_test[:,0,:],axis=1)
        se = np.std(spike_decode_test[:,0,:],axis=1)
        ax.plot(t,u,'r')
        ax.fill_between(t,u-se,u+se,facecolor=(1,0,0,0.5))
        self.add_subplot_fixings(ax, chance_level=1/8, ylim=1.1)
        ax.set_ylabel('Decoding accuracy')
        
        ax = f.add_subplot(2, 2, 4)
        u = np.mean(spike_decode[:,0,:],axis=1)
        se = np.std(spike_decode[:,0,:],axis=1)
        ax.plot(t,u,'b')
        ax.fill_between(t,u-se,u+se,facecolor=(0,0,1,0.5))
        u = np.mean(synapse_decode_test[:,0,:],axis=1)
        se = np.std(synapse_decode_test[:,0,:],axis=1)
        ax.plot(t,u,'r')
        ax.fill_between(t,u-se,u+se,facecolor=(1,0,0,0.5))
        self.add_subplot_fixings(ax, chance_level=1/8, ylim=1.1)
        ax.set_ylabel('Decoding accuracy')
          
        if tight:
            plt.tight_layout()
            plt.savefig('ABBA summary.pdf', format='pdf')
        plt.show()
        
    def add_subplot_fixings(self, ax, chance_level = 0, ylim = 1.1, delayed_rule=False):
    
        ax.plot([0,0],[-2, 99],'k--')
        
        if self.ABBA:
            ax.plot([500,500],[-2, 99],'k--')
            ax.plot([1000,1000],[-2, 99],'k--')
            ax.plot([1500,1500],[-2, 99],'k--')
            ax.plot([2000,2000],[-2, 99],'k--')
            ax.set_xlim([-500,2500])
            ax.set_xticks([-500,0,500,1000,1500,2000,2500])
        elif self.postle:
            ax.plot([500,500],[-2, 99],'k--')
            ax.plot([1000,1000],[-2, 99],'k--')
            ax.plot([1500,1500],[-2, 99],'k--')
            ax.plot([2000,2000],[-2, 99],'k--')
            ax.plot([2500,2500],[-2, 99],'k--')
            ax.plot([3000,3000],[-2, 99],'k--')
            ax.set_xlim([-500,3500])
            ax.set_xticks([-500,0,500,1000,1500,2000,2500,3000])   
        else:
            ax.plot([500,500],[-2, 99],'k--')
            ax.plot([1500,1500],[-2, 99],'k--')
            ax.set_xlim([-500,2000])
            ax.set_xticks([-500,0,500,1500])
            if delayed_rule:
                ax.set_xticks([-500,0,500,1000,1500])
                ax.plot([1000,1000],[-2, 99],'k--')
            
        ax.plot([-700,3600],[chance_level, chance_level],'k--')
        ax.set_ylim([-0.1, ylim])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        #ax.set_title('Tuning similarity between proximal and distal neurons')
        ax.set_ylabel('Response')
        ax.set_xlabel('Time relative to sample onset (ms)')
      
        
def compare_two_tasks(fn1, fn2, DMC=False, ABBA_flag=False,rule = 0, dt=25):
    
    # enter the two filenames, fn1 and fn2
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
    
    
    na1 = neural_analysis(fn1, ABBA = ABBA_flag)
    na2 = neural_analysis(fn2, ABBA = ABBA_flag)
    
    mean_resp1, _, _ = na1.calc_mean_responses()
    svm_results1 = na1.calculate_svms(ABBA = ABBA_flag)
    sample_pev1, _, rule_pev1, _, sample_stp_pev1, _ , sample_cat_pev1, sample_cat_stp_pev1 = na1.calculate_pevs(ABBA = ABBA_flag)
    
    mean_resp2, _, _ = na2.calc_mean_responses()
    svm_results2 = na2.calculate_svms()
    sample_pev2, _, rule_pev2, _, sample_stp_pev2, _ ,sample_cat_pev2, sample_cat_stp_pev2 = na2.calculate_pevs()
    
    if DMC:
        sample_pev1 = sample_cat_pev1
        sample_pev2 = sample_cat_pev2
        sample_stp_pev1 = sample_cat_stp_pev1
        sample_stp_pev2 = sample_cat_stp_pev2
        svm_results1['sample_full'] = svm_results1['sample_full_cat']
        svm_results2['sample_full'] = svm_results2['sample_full_cat']
        svm_results1['sample_full_stp'] = svm_results1['sample_full_cat_stp']
        svm_results2['sample_full_stp'] = svm_results2['sample_full_cat_stp']
        
    if na1.num_rules>1 and False:
        # not sure if I want this. If there are more than one task rules, this part will average 
        # across all rules
        sample_pev1[:,0,:] = np.mean(sample_cat_pev1,axis=1)
        sample_pev2[:,0,:] = np.mean(sample_cat_pev2,axis=1)
        sample_stp_pev1[:,0,:] = np.mean(sample_cat_stp_pev1,axis=1)
        sample_stp_pev2[:,0,:] = np.mean(sample_cat_stp_pev2,axis=1)
        svm_results1['sample_full'][:,0,:] = np.mean(svm_results1['sample_full'],axis=1)
        svm_results2['sample_full'][:,0,:] = np.mean(svm_results2['sample_full'],axis=1)
        svm_results1['sample_full_stp'][:,0,:] = np.mean(svm_results1['sample_full_stp'],axis=1)
        svm_results2['sample_full_stp'][:,0,:] = np.mean(svm_results2['sample_full_stp'],axis=1)
        rule = 0
        
    
    f = plt.figure(figsize=(8,4))
    t = np.arange(0,2700,dt)
    
    t -= 900
    t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)
    if ABBA_flag:
        t = np.arange(0,200+500+1500+300+300,dt)
        t = np.arange(0,200+500+250+2000,dt)
        t -= 700
        t0,t1,t2,t3,t4,t5,t6 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1000),np.where(t==1500),np.where(t==2000), np.where(t==2500)
        
       
    ax = f.add_subplot(2, 3, 1)
    ax.hold(True)
    u1 = np.mean(np.mean(mean_resp1[:,rule,:,:],axis=1),axis=0)
    u2 = np.mean(np.mean(mean_resp2[:,rule,:,:],axis=1),axis=0)
    se1 = np.std(np.mean(mean_resp1[:,rule,:,:],axis=1),axis=0)/np.sqrt(mean_resp1.shape[0])
    se2 = np.std(np.mean(mean_resp2[:,rule,:,:],axis=1),axis=0)/np.sqrt(mean_resp1.shape[0])
    ax.fill_between(t,u1-se1,u1+se1,facecolor=(0,1,0))
    ax.fill_between(t,u2-se2,u2+se2,facecolor=(1,0,1))
    ax.plot(t,u1,'g',label='without STP',color=(0,0.5,0),linewidth=2)
    ax.plot(t,u2,'m',label='with STP',color=(0.5,0,0.5),linewidth=2)
    na1.add_subplot_fixings(ax, chance_level=0, ylim=6, ABBA_task=ABBA_flag)
    green_patch = mpatches.Patch(color='green', label='without STP')
    magenta_patch = mpatches.Patch(color='magenta', label='with STP')
    plt.legend(loc=0, handles=[green_patch,magenta_patch],prop={'size':6})
    ax.set_title('Mean population response')
    ax.set_ylabel('Mean response')
    
    ax = f.add_subplot(2, 3, 2)
    ax.hold(True)
    u1 = np.mean(sample_pev1[:,rule,:],axis=0)
    u2 = np.mean(sample_pev2[:,rule,:],axis=0)
    u3 = np.mean(rule_pev2,axis=0)
    se1 = np.std(sample_pev1[:,rule,:],axis=0)/np.sqrt(sample_pev1.shape[0])
    se2 = np.std(sample_pev2[:,rule,:],axis=0)/np.sqrt(sample_pev1.shape[0])
    ax.fill_between(t,u1-se1,u1+se1,facecolor=(0,1,0))
    ax.fill_between(t,u2-se2,u2+se2,facecolor=(1,0,1))
    ax.plot(t,u1,'g',label='without STP',color=(0,0.5,0),linewidth=2)
    ax.plot(t,u2,'m',label='with STP',color=(0.5,0,0.5),linewidth=2)
    ax.plot(t,u3,label='rule with STP',color=(0,0,0),linewidth=2)
    na1.add_subplot_fixings(ax, chance_level=0, ylim=0.35,ABBA_task=ABBA_flag)
    #green_patch = mpatches.Patch(color='green', label='without STP')
    #magenta_patch = mpatches.Patch(color='magenta', label='with STP')
    #plt.legend(loc=0, handles=[green_patch,magenta_patch])
    ax.set_title('Neuron sample selectivity')
    ax.set_ylabel('Normalized PEV')
    
    ax = f.add_subplot(2, 3, 3)
    ax.hold(True)
    u1 = np.mean(svm_results1['sample_full'][:,rule,:],axis=1)
    u2 = np.mean(svm_results2['sample_full'][:,rule,:],axis=1)
    se1 = np.std(svm_results1['sample_full'][:,rule,:],axis=1)
    se2 = np.std(svm_results2['sample_full'][:,rule,:],axis=1)
    se1[np.isnan(se1)] = 0
    se2[np.isnan(se2)] = 0
    ax.fill_between(t,u1-se1,u1+se1,facecolor=(0,1,0))
    ax.fill_between(t,u2-se2,u2+se2,facecolor=(1,0,1))
    ax.plot(t,u1,'g',label='without STP',color=(0,0.5,0),linewidth=2)
    ax.plot(t,u2,'m',label='with STP',color=(0.5,0,0.5),linewidth=2)
    na1.add_subplot_fixings(ax, chance_level=1/8, ylim=1.1,ABBA_task=ABBA_flag)
    #green_patch = mpatches.Patch(color='green', label='without STP')
    #magenta_patch = mpatches.Patch(color='magenta', label='with STP')
    #plt.legend(loc=0, handles=[green_patch,magenta_patch])
    ax.set_title('Neuron sample decoding accuracy')
    ax.set_ylabel('Decoding accuracy')
    
    
    ax = f.add_subplot(2, 3, 4)
    im = ax.imshow(sample_stp_pev2[:,rule,:],aspect='auto',interpolation=None)
    if not ABBA_flag:
        ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
        ax.set_xticklabels([-500,0,500,1500])
    else:
        ax.set_xticks([t0[0], t1[0], t2[0], t3[0], t4[0], t5[0], t6[0]])
        ax.set_xticklabels([-500,0,500,1000,1500,2000,2000,2500])
    f.colorbar(im,orientation='vertical')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Neuron number')
    ax.set_xlabel('Time relative to sample onset (ms)')
    ax.set_title('Synaptic sample selectivity')
    
    ax = f.add_subplot(2, 3, 5)
    ax.hold(True)
    u2 = np.mean(sample_stp_pev2[:,rule,:],axis=0)
    se2 = np.std(sample_stp_pev2[:,rule,:],axis=0)/np.sqrt(sample_pev1.shape[0])
    ax.fill_between(t,u2-se2,u2+se2,facecolor=(1,0,1))
    ax.plot(t,u2,'m',label='with STP',color=(0.5,0,0.5),linewidth=2)
    na1.add_subplot_fixings(ax, chance_level=0, ylim=0.3,ABBA_task=ABBA_flag)
    #green_patch = mpatches.Patch(color='green', label='without STP')
    #magenta_patch = mpatches.Patch(color='magenta', label='with STP')
    #plt.legend(loc=0, handles=[green_patch,magenta_patch])
    ax.set_title('Synaptic sample selectivity')
    ax.set_ylabel('Normalized PEV')
    
    ax = f.add_subplot(2, 3, 6)
    ax.hold(True)
    u2 = np.mean(svm_results2['sample_full_stp'][:,rule,:],axis=1)
    se2 = np.std(svm_results2['sample_full_stp'][:,rule,:],axis=1)
    se2[np.isnan(se2)] = 0
    ax.fill_between(t,u2-se2,u2+se2,facecolor=(1,0,1))
    ax.plot(t,u2,'m',label='with STP',color=(0.5,0,0.5),linewidth=2)
    na1.add_subplot_fixings(ax, chance_level=1/8, ylim=1.1,ABBA_task=ABBA_flag)
    #green_patch = mpatches.Patch(color='green', label='without STP')
    #magenta_patch = mpatches.Patch(color='magenta', label='with STP')
    #plt.legend(loc=0, handles=[green_patch,magenta_patch])
    ax.set_title('Synaptic sample decoding accuray')
    ax.set_ylabel('Decoding accuracy')
    
    plt.tight_layout()
    plt.savefig('DMS comparison.pdf', format='pdf')
    plt.show()
      
        
def compare_two_tasks_two_rules(fn1, fn2, DMC=False, ABBA=False, dt=25):
    
    # enter the two filenames, fn1 and fn2
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
    
    
    na1 = neural_analysis(fn1)
    na2 = neural_analysis(fn2)
    
    mean_resp1, _, _ = na1.calc_mean_responses()
    svm_results1 = na1.calculate_svms()
    sample_pev1, _, rule_pev1, _, sample_stp_pev1, _ , sample_cat_pev1, sample_cat_stp_pev1 = na1.calculate_pevs()
    
    mean_resp2, _, _ = na2.calc_mean_responses()
    svm_results2 = na2.calculate_svms()
    sample_pev2, _, rule_pev2, _, sample_stp_pev2, _ ,sample_cat_pev2, sample_cat_stp_pev2 = na2.calculate_pevs() 
    
    if DMC:
        sample_pev1 = sample_cat_pev1
        sample_pev2 = sample_cat_pev2
        sample_stp_pev1 = sample_cat_stp_pev1
        sample_stp_pev2 = sample_cat_stp_pev2
        svm_results1['sample_full'] = svm_results1['sample_full_cat']
        svm_results2['sample_full'] = svm_results2['sample_full_cat']
        svm_results1['sample_full_stp'] = svm_results1['sample_full_cat_stp']
        svm_results2['sample_full_stp'] = svm_results2['sample_full_cat_stp']
        
    
    f = plt.figure(figsize=(8,9))
    t = np.arange(0,2700,dt)
    t -= 900
    t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)
    if ABBA:
        t = np.arange(0,200+500+1500+300+300,dt)
        t -= 700
        t0,t1,t2,t3,t4,t5,t6,t7 = np.where(t==-500), np.where(t==0),np.where(t==300),np.where(t==600),np.where(t==900),np.where(t==1200), np.where(t==1500), np.where(t==1800)
        
       
    ax = f.add_subplot(5, 2, 1)
    ax.hold(True)
    u1 = np.mean(np.mean(np.mean(mean_resp1[:,:,:,:],axis=1),axis=1),axis=0)
    u2 = np.mean(np.mean(np.mean(mean_resp2[:,:,:,:],axis=1),axis=1),axis=0)
    se1 = np.std(np.mean(np.mean(mean_resp1[:,:,:,:],axis=1),axis=1),axis=0)/np.sqrt(mean_resp1.shape[0])
    se2 = np.std(np.mean(np.mean(mean_resp2[:,:,:,:],axis=1),axis=1),axis=0)/np.sqrt(mean_resp1.shape[0])
    ax.fill_between(t,u1-se1,u1+se1,facecolor=(0,1,0))
    ax.fill_between(t,u2-se2,u2+se2,facecolor=(1,0,1))
    ax.plot(t,u1,'g',label='without STP',color=(0,0.5,0),linewidth=2)
    ax.plot(t,u2,'m',label='with STP',color=(0.5,0,0.5),linewidth=2)
    na1.add_subplot_fixings(ax, chance_level=0, ylim=4, ABBA_task=ABBA,delayed_rule=True)
    green_patch = mpatches.Patch(color='green', label='without STP')
    magenta_patch = mpatches.Patch(color='magenta', label='with STP')
    plt.legend(loc=0, handles=[green_patch,magenta_patch],prop={'size':6})
    ax.set_title('Mean population response')
    ax.set_ylabel('Mean response')
    
    ax = f.add_subplot(5, 2, 2)
    ax.hold(True) 
    u1 = np.mean(rule_pev1,axis=0)
    u2 = np.mean(rule_pev2,axis=0)
    se1 = np.std(rule_pev1,axis=0)/np.sqrt(sample_pev1.shape[0])
    se2 = np.std(rule_pev2,axis=0)/np.sqrt(sample_pev1.shape[0])
    ax.fill_between(t,u1-se1,u1+se1,facecolor=(0,1,0))
    ax.fill_between(t,u2-se2,u2+se2,facecolor=(1,0,1))
    ax.plot(t,u1,'g',label='without STP',color=(0,0.5,0),linewidth=2)
    ax.plot(t,u2,'m',label='with STP',color=(0.5,0,0.5),linewidth=2)
    na1.add_subplot_fixings(ax, chance_level=0, ylim=0.3,ABBA_task=ABBA,delayed_rule=True)
    ax.set_title('Neuron rule selectivity')
    ax.set_ylabel('Normalized PEV')
    
    for rule in range(2):
        ax = f.add_subplot(5, 2, 3+2*rule)
        ax.hold(True)
        u1 = np.mean(sample_pev1[:,rule,:],axis=0)
        u2 = np.mean(sample_pev2[:,rule,:],axis=0)
        se1 = np.std(sample_pev1[:,rule,:],axis=0)/np.sqrt(sample_pev1.shape[0])
        se2 = np.std(sample_pev2[:,rule,:],axis=0)/np.sqrt(sample_pev1.shape[0])
        ax.fill_between(t,u1-se1,u1+se1,facecolor=(0,1,0))
        ax.fill_between(t,u2-se2,u2+se2,facecolor=(1,0,1))
        ax.plot(t,u1,'g',label='without STP',color=(0,0.5,0),linewidth=2)
        ax.plot(t,u2,'m',label='with STP',color=(0.5,0,0.5),linewidth=2)
        na1.add_subplot_fixings(ax, chance_level=0, ylim=0.7,ABBA_task=ABBA,delayed_rule=True)
        ax.set_title('Neuron sample selectivity')
        ax.set_ylabel('Normalized PEV')
        
        ax = f.add_subplot(5, 2, 4+2*rule)
        ax.hold(True)
        u1 = np.mean(svm_results1['sample_full'][:,rule,:],axis=1)
        u2 = np.mean(svm_results2['sample_full'][:,rule,:],axis=1)
        se1 = np.std(svm_results1['sample_full'][:,rule,:],axis=1)
        se2 = np.std(svm_results2['sample_full'][:,rule,:],axis=1)
        se1[np.isnan(se1)] = 0
        se2[np.isnan(se2)] = 0
        ax.fill_between(t,u1-se1,u1+se1,facecolor=(0,1,0))
        ax.fill_between(t,u2-se2,u2+se2,facecolor=(1,0,1))
        ax.plot(t,u1,'g',label='without STP',color=(0,0.5,0),linewidth=2)
        ax.plot(t,u2,'m',label='with STP',color=(0.5,0,0.5),linewidth=2)
        na1.add_subplot_fixings(ax, chance_level=1/8, ylim=1.1,ABBA_task=ABBA,delayed_rule=True)
        ax.set_title('Neuron sample decoding accuracy')
        ax.set_ylabel('Decoding accuracy')
        
        ax = f.add_subplot(5, 2, 7+2*rule)
        ax.hold(True)
        u1 = np.mean(sample_stp_pev1[:,rule,:],axis=0)
        u2 = np.mean(sample_stp_pev2[:,rule,:],axis=0)
        se1 = np.std(sample_stp_pev1[:,rule,:],axis=0)/np.sqrt(sample_pev1.shape[0])
        se2 = np.std(sample_stp_pev2[:,rule,:],axis=0)/np.sqrt(sample_pev1.shape[0])
        ax.fill_between(t,u1-se1,u1+se1,facecolor=(0,1,0))
        ax.fill_between(t,u2-se2,u2+se2,facecolor=(1,0,1))
        ax.plot(t,u1,'g',label='without STP',color=(0,0.5,0),linewidth=2)
        ax.plot(t,u2,'m',label='with STP',color=(0.5,0,0.5),linewidth=2)
        na1.add_subplot_fixings(ax, chance_level=0, ylim=0.7,ABBA_task=ABBA,delayed_rule=True)
        ax.set_title('Synaptic sample selectivity')
        ax.set_ylabel('Normalized PEV')
        
        ax = f.add_subplot(5, 2, 8+2*rule)
        ax.hold(True)
        u1 = np.mean(svm_results1['sample_full_stp'][:,rule,:],axis=1)
        u2 = np.mean(svm_results2['sample_full_stp'][:,rule,:],axis=1)
        se1 = np.std(svm_results1['sample_full_stp'][:,rule,:],axis=1)
        se2 = np.std(svm_results2['sample_full_stp'][:,rule,:],axis=1)
        se1[np.isnan(se1)] = 0
        se2[np.isnan(se2)] = 0
        ax.fill_between(t,u1-se1,u1+se1,facecolor=(0,1,0))
        ax.fill_between(t,u2-se2,u2+se2,facecolor=(1,0,1))
        ax.plot(t,u1,'g',label='without STP',color=(0,0.5,0),linewidth=2)
        ax.plot(t,u2,'m',label='with STP',color=(0.5,0,0.5),linewidth=2)
        na1.add_subplot_fixings(ax, chance_level=1/8, ylim=1.1,ABBA_task=ABBA,delayed_rule=True)
        ax.set_title('Synaptic sample decoding accuracy')
        ax.set_ylabel('Decoding accuracy')
        
    plt.tight_layout()
    plt.savefig('Two rules comparison.pdf', format='pdf')
    plt.show()
    
    
def plt_dual_figures(fn1, fn2, ABBA=False, DMC=False, two_rules=False):
    
    # assume fn1 has no STP, and fn2 does
    if two_rules:
        fig_handle = plt.figure(figsize=(10,10))
        sp = 8
    else:
        fig_handle = plt.figure(figsize=(8,10))
        sp = 6
    na = neural_analysis(fn1, ABBA=ABBA)
    na.plot_all_figures(rule=0, STP=False, ABBA=ABBA, DMC=DMC, two_rules=two_rules,f=fig_handle, start_sp=0, num_rows=5, tight=False)
    na = neural_analysis(fn2, ABBA=ABBA)
    na.plot_all_figures(rule=0, STP=True, ABBA=ABBA, DMC=DMC, two_rules=two_rules,f=fig_handle, start_sp=sp, num_rows=5, tight=True)
    
    
def plot_summary_decoding_figure():

    fn1 = 'C:/Users/nicol_000/Projects/RNN STP Model/saved_model_files/DMS.pkl'
    fn2 = 'C:/Users/nicol_000/Projects/RNN STP Model/saved_model_files/DMS_std_stf.pkl'
    fn3 = 'C:/Users/nicol_000/Projects/RNN STP Model/saved_model_files/DMC.pkl'
    fn4 = 'C:/Users/nicol_000/Projects/RNN STP Model/saved_model_files/DMC_std_stf.pkl'
    fn5 = 'C:/Users/nicol_000/Projects/RNN STP Model/saved_model_files/DMS_rotation.pkl'
    fn6 = 'C:/Users/nicol_000/Projects/RNN STP Model/saved_model_files/DMS_rotate_std_stf_v3.pkl'
    fn7 = 'C:/Users/nicol_000/Projects/RNN STP Model/saved_model_files/DMS_and_rotate_v3.pkl'
    fn8 = 'C:/Users/nicol_000/Projects/RNN STP Model/saved_model_files/DMS_and_rotate_std_stf_v3.pkl'
    fn9 = 'C:/Users/nicol_000/Projects/RNN STP Model/saved_model_files/ABBA.pkl'
    fn10 = 'C:/Users/nicol_000/Projects/RNN STP Model/saved_model_files/ABBA_std_stf_v2.pkl'

    fig_handle = plt.figure(figsize=(6,10))
    num_rows = 5
    plot_decoding_pairs(fn1, fn2, fig_handle, num_rows=num_rows, start_sp=0, DMC=False, ABBA=False, two_rules=False)
    plot_decoding_pairs(fn3, fn4, fig_handle, num_rows=num_rows, start_sp=2, DMC=True, ABBA=False, two_rules=False)
    plot_decoding_pairs(fn5, fn6, fig_handle, num_rows=num_rows, start_sp=4, DMC=False, ABBA=False, two_rules=False)
    plot_decoding_pairs(fn7, fn8, fig_handle, num_rows=num_rows, start_sp=6, DMC=False, ABBA=False, two_rules=True)
    plot_decoding_pairs(fn9, fn10, fig_handle, num_rows=num_rows, start_sp=8, DMC=False, ABBA=True, two_rules=False)
  
    plt.tight_layout()
    plt.savefig('Summary.pdf', format='pdf')
    plt.show()
    
def plot_decoding_pairs(fn1, fn2, f, num_rows, start_sp, DMC=False, ABBA=False, two_rules=False):
    
    dt = 25
    na = neural_analysis(fn1, ABBA=False)
    svm_results1 = na.calculate_svms()
    na = neural_analysis(fn2, ABBA=False)
    svm_results2 = na.calculate_svms()
    trial_length_steps = svm_results1['sample_full'].shape[0]
    trial_length = np.int_(trial_length_steps*dt)
        
    t = np.arange(0,trial_length,dt)
    t -= 900 # assuming 400 ms dead time, 500 ms fixation
    
    if DMC:
        svm_results1['sample_full'] = svm_results1['sample_full_cat']
        svm_results2['sample_full'] = svm_results2['sample_full_cat']
        svm_results1['sample_full_stp'] = svm_results1['sample_full_cat_stp']
        svm_results2['sample_full_stp'] = svm_results2['sample_full_cat_stp']
    if two_rules:
        svm_results1['sample_full'] = np.mean(svm_results1['sample_full'],axis=1)
        svm_results2['sample_full'] = np.mean(svm_results2['sample_full'],axis=1)
        svm_results1['sample_full_stp'] = np.mean(svm_results1['sample_full_stp'],axis=1)
        svm_results2['sample_full_stp'] = np.mean(svm_results2['sample_full_stp'],axis=1)
    else:
        svm_results1['sample_full'] = np.squeeze(svm_results1['sample_full'][:,0,:])
        svm_results2['sample_full'] = np.squeeze(svm_results2['sample_full'][:,0,:])
        svm_results1['sample_full_stp'] = np.squeeze(svm_results1['sample_full_stp'][:,0,:])
        svm_results2['sample_full_stp'] = np.squeeze(svm_results2['sample_full_stp'][:,0,:])
        
    print(svm_results1['sample_full'].shape)
    
    ax = f.add_subplot(num_rows, 2, start_sp+1)
    u = np.mean(svm_results1['sample_full'],axis=1)
    se = np.std(svm_results1['sample_full'],axis=1)
    print(u.shape, se.shape, t.shape)
    ax.plot(t,u,'g')
    ax.fill_between(t,u-se,u+se,facecolor=(0,0.5,0))
    u = np.mean(svm_results2['sample_full'],axis=1)
    se = np.std(svm_results2['sample_full'],axis=1)
    ax.plot(t,u,'m')
    ax.fill_between(t,u-se,u+se,facecolor=(0.5,0,0.5))
    if DMC:
        ax.set_title('Neuronal sample \category decoding')
        cl = 1/2
    else:
        ax.set_title('Neuronal sample decoding')
        cl = 1/8
    na.add_subplot_fixings(ax, chance_level=cl)
    ax.set_ylabel('Decoding accuracy')
    
    ax = f.add_subplot(num_rows, 2, start_sp+2)
    u = np.mean(svm_results2['sample_full_stp'],axis=1)
    se = np.std(svm_results2['sample_full_stp'],axis=1)
    ax.plot(t,u,'m')
    ax.fill_between(t,u-se,u+se,facecolor=(0.5,0,0.5))
    if DMC:
        ax.set_title('Synaptic sample \category decoding')
        cl = 1/2
    else:
        ax.set_title('Synaptic sample decoding')
        cl = 1/8
    na.add_subplot_fixings(ax, chance_level=cl)
    
    
def plot_summary_results(old_format = False):
    
    dt = 20
    t = np.arange(0,2900,dt)
    t -= 900
    t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)
    num_svm_reps = 2
    trial_length = (400+500+500+1000+500)//dt
    N = 11
    data_dir = 'D:/Masse/RNN STP/saved_models/'
    fn = ['DMS_', 'DMS_stp_', 'DMC_stp_', 'DMrS_stp_']
    titles = ['DMS no STP', 'DMS', 'DMC', 'DMrS']
    
    spike_decoding = np.zeros((4, N, trial_length, num_svm_reps))
    synapse_decoding = np.zeros((4, N, trial_length, num_svm_reps))
    spike_decoding_test = np.zeros((4, N, trial_length, num_svm_reps))
    synapse_decoding_test = np.zeros((4, N, trial_length, num_svm_reps))
    perf = np.zeros((4, N))
    perf_shuffled_hidden = np.zeros((4, N))
    perf_shuffled_stp = np.zeros((4, N))
    
    """
    Calculate the spiking and synaptic sample decoding accuracy across all networks
    Calculate the behavioral performance
    """
    for i in range(N):
        print('Group ', i)
        for j in range(1,4):
            if j == 2:
                DMC = [True]
            else:
                DMC = [False]
            f = data_dir + fn[j] + str(i) + '.pkl'
            na = neural_analysis(f, ABBA=False, old_format = old_format)
            perf[j,i] = get_perf(na.desired_outputs, na.model_outputs, na.train_mask, na.rule)
            spike_decode, synapse_decode, spike_decode_test, synapse_decode_test = na.calculate_svms(num_reps = num_svm_reps, DMC = DMC)
            spike_decoding[j,i,:,:] = spike_decode[:,0,:]
            synapse_decoding[j,i,:,:] = synapse_decode[:,0,:]
            spike_decoding_test[j,i,:,:] = spike_decode_test[:,0,:]
            synapse_decoding_test[j,i,:,:] = synapse_decode_test[:,0,:]
            a = contrib_to_behavior.Analysis(f,old_format = old_format)
            perf[j,i], perf_shuffled_hidden[j,i], perf_shuffled_stp[j,i] = a.simulate_network()
            
      
    """
    Calculate the mean decoding accuracy for the last 500 ms of the delay
    """
    d = range(1900//dt,2400//dt)
    delay_accuracy = np.mean(np.mean(spike_decoding[:,:,d,:],axis=3),axis=2)
    ind_example = [0]
    for j in range(1,4):
        ind_good_perf = np.where(perf[j,:] > 0.9)[0]
        ind_sort = np.argsort(delay_accuracy[j,ind_good_perf])[0]
        ind_example.append(ind_good_perf[ind_sort])
        
    """
    Plot decoding accuracy from example models
    Only consider models with performance accuracy above 99.0%
    Will use the model with the lowest spike decoding value during the last 500 ms of the delay
    """     
    print(ind_example)  
    f = plt.figure(figsize=(6,4))
    for j in range(1,4):
        if j == 2:
            chance_level = 1/2
        else:
            chance_level = 1/8
        ax = f.add_subplot(2, 2, j+1)
        u = np.mean(spike_decoding[j,ind_example[j],:,:],axis=1)
        se = np.std(spike_decoding[j,ind_example[j],:,:],axis=1)
        ax.plot(t,u,'g')
        ax.fill_between(t,u-se,u+se,facecolor=(0,1,0,0.5))
        u = np.mean(synapse_decoding[j,ind_example[j],:,:],axis=1)
        se = np.std(synapse_decoding[j,ind_example[j],:,:],axis=1)
        ax.plot(t,u,'m')
        ax.fill_between(t,u-se,u+se,facecolor=(1,0,1,0.5))
        na.add_subplot_fixings(ax, chance_level=chance_level)
        ax.set_title(titles[j])
        ax.set_ylabel('Decoding accuracy')
        ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig('Example models.pdf', format='pdf')    
    plt.show()
        
    """
    Plot mean decoding accuracy across all models
    Only use models with performance accuracy above 85%
    """     
    print(ind_example)  
    f = plt.figure(figsize=(6,4))
    for j in range(1,4):
        if j == 2:
            chance_level = 1/2
        else:
            chance_level = 1/8
        ind_good_models = np.where(perf[j,:] > 0.85)[0]
        ax = f.add_subplot(2, 2, j+1)
        u = np.mean(np.mean(spike_decoding[j,ind_good_models,:,:],axis=2),axis=0)
        se = np.std(np.mean(spike_decoding[j,ind_good_models,:,:],axis=2),axis=0)/np.sqrt(len(ind_good_models))
        ax.plot(t,u,'g')
        ax.fill_between(t,u-se,u+se,facecolor=(0,1,0,0.5))
        u = np.mean(np.mean(synapse_decoding[j,ind_good_models,:,:],axis=2),axis=0)
        se = np.std(np.mean(synapse_decoding[j,ind_good_models,:,:],axis=2),axis=0)/np.sqrt(len(ind_good_models))
        ax.plot(t,u,'m')
        ax.fill_between(t,u-se,u+se,facecolor=(1,0,1,0.5))
        na.add_subplot_fixings(ax, chance_level=chance_level)
        ax.set_title(titles[j])
        ax.set_ylabel('Decoding accuracy')
        ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig('Average models.pdf', format='pdf') 
    plt.show()
        
    """
    Plot decoding accuracy across all models using heatmaps
    Only use models with performance accuracy above 97.5%
    """   
    print(ind_example)  
    f = plt.figure(figsize=(6,4))
    for j in range(1,4):
        if j == 2:
            chance_level = 1/2
        else:
            chance_level = 1/8
        ind_good_models = np.where(perf[j,:] > 0.975)[0]
        ax = f.add_subplot(2, 2, j+1)
        u = np.mean(synapse_decoding[j,ind_good_models,:,:],axis=2)
        im = ax.imshow(u,aspect='auto',interpolation=None)
        f.colorbar(im,orientation='vertical')
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Model number')
        ax.set_xlabel('Time relative to sample onset (ms)')
        ax.spines['top'].set_visible(False)
        ax.set_title(titles[j])
        ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
        ax.set_xticklabels([-500,0,500,1500])
        

    plt.tight_layout()
    plt.savefig('All models.pdf', format='pdf') 
    plt.show()
    

    print(ind_example)        
    return spike_decoding, synapse_decoding, spike_decoding_test, synapse_decoding_test, perf, perf_shuffled_hidden, perf_shuffled_stp

def plot_variable_delay_results():
    
    """
    Plot a model that was trained on a variable delay
    """ 
    data_dir = 'C:/Users/Freedmanlab/Documents/Masse/STP/saved_models/'
    dt = 25
    num_svm_reps = 5
    t = np.arange(0,2900,dt)
    t -= 900
    fn = 'DMS_EI_std_stf_var_delay_1_iter1000.pkl'
    f = data_dir + fn 
    na = neural_analysis(f, ABBA=False)
    perf = get_perf(na.desired_outputs, na.model_outputs, na.train_mask)
    print('Model accuracy = ', perf)
    spike_decode, synapse_decode = na.calculate_svms(num_reps = num_svm_reps, DMC = False)
    
    f = plt.figure(figsize=(3,2))
    chance_level = 1/8
    ax = f.add_subplot(1, 1, 1)
    u = np.mean(spike_decode[:,0,:],axis=1)
    se = np.std(spike_decode[:,0,:],axis=1)
    ax.plot(t,u,'g')
    ax.fill_between(t,u-se,u+se,facecolor=(0,1,0,0.5))
    u = np.mean(synapse_decode[:,0,:],axis=1)
    se = np.std(synapse_decode[:,0,:],axis=1)
    ax.plot(t,u,'m')
    ax.fill_between(t,u-se,u+se,facecolor=(1,0,1,0.5))
    na.add_subplot_fixings(ax, chance_level=chance_level)
    ax.set_ylabel('Decoding accuracy')
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig('Var delay model.pdf', format='pdf') 
    plt.show()

def plot_multiple_delay_results():

    dt = 20
    num_svm_reps = 5
    N = 8
    data_dir = 'D:/Masse/RNN STP/saved_models/'
    delay = [1000,1500,2000]
    num_delays = len(delay)

    mean_decoding = np.zeros((num_delays, N))
    std_decoding = np.zeros((num_delays, N))
    perf = np.zeros((num_delays, N))

    for i in range(N):
        print('Group ', i)
        for j in range(num_delays):
            
            if j==0:
                f = data_dir + 'DMS_stp_' + str(i) + '.pkl'
            else:
                f = data_dir + 'DMS_stp_delay_' + str(delay[j]) + '_' + str(i) + '.pkl'
                
            try:
                na = neural_analysis(f, ABBA=False, old_format = False)
                spike_decode, synapse_decode,_,_ = na.calculate_svms(num_reps = num_svm_reps, DMC = [False])
                perf[j,i] = get_perf(na.desired_outputs, na.model_outputs, na.train_mask, na.rule)
            except:
                na = neural_analysis(f, ABBA=False, old_format = True)
                spike_decode, synapse_decode,_,_ = na.calculate_svms(num_reps = num_svm_reps, DMC = [False])
                perf[j,i] = get_perf(na.desired_outputs, na.model_outputs, na.train_mask, na.rule)
                
            
            # look at last 100 ms of delay epoch
            
            # variable delay
            delay_end = (400+500+500+delay[j])//dt
            delay_start = (400+500+500+delay[j]-100)//dt
            
            # variable tau
            #delay_end = (400+500+500+1000)//dt
            #delay_start = (400+500+500+900)//dt
            
            mean_decoding[j,i] = np.mean(spike_decode[delay_start:delay_end,0,:])
            std_decoding[j,i] = np.std(np.mean(spike_decode[delay_start:delay_end,0,:],axis=0))
            print(i,j,perf[j,i],mean_decoding[j,i],std_decoding[j,i])
            
            
    f = plt.figure(figsize=(3,2))
    chance_level = 1/8
    ax = f.add_subplot(1, 1, 1)
    
    for i, d in enumerate(delay):
        # only use models with over 90% accuracy
        ind_good_model = np.where(perf[i,:]>0.90)[0]
        ax.plot([d]*len(ind_good_model),mean_decoding[i,ind_good_model],'k.')
      
    ax.plot([0,3000],[chance_level,chance_level],'k--')
    ax.set_ylim([0, 1])
    ax.set_xlim([400, 2100])
    ax.set_xticks(delay)
    ax.set_xticklabels(delay)
            
    return mean_decoding, std_decoding, perf


def plot_summary_results_v2(old_format = False):
    
    dt = 20
    t = np.arange(0,2900,dt)
    t -= 900
    t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)
    num_svm_reps = 2
    trial_length = (400+500+500+1000+500)//dt
    N = 20
    data_dir = 'D:/Masse/RNN STP/saved_models/'
    fn = ['DMS_stp_', 'DMC_stp_', 'DMrS_stp_', 'DMS_DMrS_stp_']
    titles = ['DMS', 'DMC', 'DMrS', 'DMS_DMrS']
    num_tasks = len(fn)
    """
    the DMS_DMrS will produce two decoding/accuracy scores, one for each task
    thus, will show num_tasks+1 set of values
    """
    
    spike_decoding = np.zeros((num_tasks+1, N, trial_length, num_svm_reps))
    synapse_decoding = np.zeros((num_tasks+1, N, trial_length, num_svm_reps))
    spike_decoding_test = np.zeros((num_tasks+1, N, trial_length, num_svm_reps))
    synapse_decoding_test = np.zeros((num_tasks+1, N, trial_length, num_svm_reps))
    perf = np.zeros((num_tasks+1, N))
    perf_shuffled_hidden = np.zeros((num_tasks+1, N))
    perf_shuffled_stp = np.zeros((num_tasks+1, N))
    
    """
    Calculate the spiking and synaptic sample decoding accuracy across all networks
    Calculate the behavioral performance
    """
    for i in range(N):
        print('Group ', i)
        for j in range(num_tasks):
            if fn[j] == 'DMC_stp_':
                DMC = [True]
            elif fn[j] == 'DMS_DMrS_stp_':
                DMC = [False, False]
            else:
                DMC = [False]
            f = data_dir + fn[j] + str(i) + '.pkl'
            
            try:
                na = neural_analysis(f, ABBA=False, old_format = old_format)
            except:
                na = neural_analysis(f, ABBA=False, old_format = not old_format)
            #perf_temp = get_perf(na.desired_outputs, na.model_outputs, na.train_mask, na.rule)
            spike_decode, synapse_decode, spike_decode_test, synapse_decode_test = na.calculate_svms(num_reps = num_svm_reps, DMC = DMC)
            try:
                a = contrib_to_behavior.Analysis(f,old_format = old_format)
                perf_temp, perf_shuffled_hidden_temp, perf_shuffled_stp_temp = a.simulate_network()
            except:
                a = contrib_to_behavior.Analysis(f,old_format = not old_format)
                perf_temp, perf_shuffled_hidden_temp, perf_shuffled_stp_temp = a.simulate_network()
            if j<3:
                print(perf_temp)
                perf[j,i] = perf_temp
                perf_shuffled_hidden[j,i] = perf_shuffled_hidden_temp
                perf_shuffled_stp[j,i] = perf_shuffled_stp_temp
                spike_decoding[j,i,:,:] = spike_decode[:,0,:]
                synapse_decoding[j,i,:,:] = synapse_decode[:,0,:]
                spike_decoding_test[j,i,:,:] = spike_decode_test[:,0,:]
                synapse_decoding_test[j,i,:,:] = synapse_decode_test[:,0,:]
            else:
                perf[j:,i] = perf_temp
                perf_shuffled_hidden[j:,i] = perf_shuffled_hidden_temp
                perf_shuffled_stp[j:,i] = perf_shuffled_stp_temp
                spike_decoding[j:,i,:,:] = np.transpose(spike_decode[:,:,:],(1,0,2))
                synapse_decoding[j:,i,:,:] = np.transpose(synapse_decode[:,:,:],(1,0,2))
                spike_decoding_test[j:,i,:,:] = np.transpose(spike_decode_test[:,:,:],(1,0,2))
                synapse_decoding_test[j:,i,:,:] = np.transpose(synapse_decode_test[:,:,:],(1,0,2))
                print(spike_decoding.shape)

     
      
    """
    Calculate the mean decoding accuracy for the last 500 ms of the delay
    """

    dt=20
    d = range(1900//dt,2400//dt)
    delay_accuracy = np.mean(np.mean(spike_decoding[:,:,d,:],axis=3),axis=2)
    fn = ['DMS_stp_', 'DMC_stp_', 'DMrS_stp_', 'DMS_DMrS_stp_']
    titles = ['DMS', 'DMC', 'DMrS', 'DMS + DMrS']  
    # combine the DMS and DMrS trials for the DMS_DMrS task
    delay_accuracy[3,:] = np.mean(delay_accuracy[3:,:],axis=0)
    perf_combined = perf[:num_tasks,:]
    perf_combined[num_tasks-1,:] = np.mean(perf[num_tasks:,:],axis=0)
    
    # will find 2 examples for each task
    ind_example = np.zeros((num_tasks, 3),dtype=np.int8)
    for j in range(num_tasks):
        ind_good_perf = np.where(perf_combined[j,:] > 0.9)[0]
        ind_sort = np.argsort(delay_accuracy[j,ind_good_perf])
        #ind_example[j,0] = ind_good_perf[ind_sort][-1]
        ind_example[j,0]= ind_good_perf[ind_sort][len(ind_sort)//2]
        ind_example[j,1]= ind_good_perf[ind_sort][0]
        
    f = plt.figure(figsize=(6,8.5))
    for j in range(num_tasks):
        if fn[j] == 'DMC_stp_':
            chance_level = 1/2
        else:
            chance_level = 1/8
        for i in range(2):
            ax = f.add_subplot(num_tasks+1, 2, j*2+i+1)
            u = np.mean(spike_decoding[j,ind_example[j,i],:,:],axis=1)
            se = np.std(spike_decoding[j,ind_example[j,i],:,:],axis=1)
            ax.plot(t,u,'g')
            ax.fill_between(t,u-se,u+se,facecolor=(0,1,0,0.5))
            u = np.mean(synapse_decoding[j,ind_example[j,i],:,:],axis=1)
            se = np.std(synapse_decoding[j,ind_example[j,i],:,:],axis=1)
            ax.plot(t,u,'m')
            ax.fill_between(t,u-se,u+se,facecolor=(1,0,1,0.5))
            na.add_subplot_fixings(ax, chance_level=chance_level)
        
        if j == 3:
            # DMS_DMrS task
            u = np.mean(spike_decoding[j+1,ind_example[j,i],:,:],axis=1)
            se = np.std(spike_decoding[j+1,ind_example[j,i],:,:],axis=1)
            ax.plot(t,u,'b')
            ax.fill_between(t,u-se,u+se,facecolor=(0,0,1,0.5))
            u = np.mean(synapse_decoding[j+1,ind_example[j,i],:,:],axis=1)
            se = np.std(synapse_decoding[j+1,ind_example[j,i],:,:],axis=1)
            ax.plot(t,u,'r')
            ax.fill_between(t,u-se,u+se,facecolor=(1,0,0,0.5))
            ax.set_xticks([-500,0,500,1000,1500])
            ax.plot([1000,1000],[-2, 99],'k--')
        
        ax.set_yticks([0,0.5,1])
        ax.set_title(titles[j])
        ax.set_ylabel('Decoding accuracy')
        ax.set_ylim([0, 1])
            
    plt.tight_layout()
    plt.savefig('Summary1.pdf', format='pdf')    
    plt.show()
    col=['b','r','g','c','k']
    marker = ['o','v','^','s','D']


    """
    Normalize delay decoding
    """
    for j in range(num_tasks+1):
        if j == 1:
            delay_accuracy[j,:] = (delay_accuracy[j,:]-0.5)*2
        else:
            delay_accuracy[j,:] = (delay_accuracy[j,:]-1/8)*8/7
        
    f = plt.figure(figsize=(6.5,3))
    ax = f.add_subplot(1, 3, 1)
    for j in range(num_tasks+1):
        ind_good_models = np.where(perf[j,:] > 0.9)[0]
    
        #ax.plot(delay_accuracy[j,ind_good_models], perf_shuffled_hidden[j,ind_good_models] 
        #        -perf[j,ind_good_models],marker[j], color=col[j], markersize=3)
        ax.plot(delay_accuracy[j,ind_good_models], perf_shuffled_hidden[j,ind_good_models] 
                -perf[j,ind_good_models],marker[j], color=col[j], markersize=3)
    ax.set_xlim(-0.1,1.02)    
    ax.set_aspect(1.12/0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks([-0.5,-0.25,0])
    ax.set_xticks([0,0.5,1])
    ax.set_ylabel('Delta acc. shuffled spike rate')
    ax.set_xlabel('Normalized delay decoding acc.')

    ax = f.add_subplot(1, 3, 2)
    for j in range(num_tasks+1):
        ind_good_models = np.where(perf[j,:] > 0.9)[0]
    
        #ax.plot(delay_accuracy[j,ind_good_models], perf_shuffled_hidden[j,ind_good_models] 
        #        -perf[j,ind_good_models],marker[j], color=col[j], markersize=3)
        ax.plot(delay_accuracy[j,ind_good_models], perf_shuffled_stp[j,ind_good_models] 
                -perf[j,ind_good_models],marker[j], color=col[j], markersize=3)
    
    ax.set_xlim(-0.1,1.02)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks([-0.5,-0.25,0])
    ax.set_xticks([0,0.5,1])
    ax.set_aspect(1.12/0.5)
    ax.set_ylabel('Delta acc. shuffled STP')
    ax.set_xlabel('Normalized delay decoding acc.')

    ax = f.add_subplot(1, 3, 3)
    for j in range(num_tasks+1):
        ind_good_models = np.where(perf[j,:] > 0.9)[0]
        ax.plot(perf_shuffled_stp[j,ind_good_models]-perf[j,ind_good_models], perf_shuffled_hidden[j,ind_good_models] 
                -perf[j,ind_good_models],marker[j], color=col[j], markersize=3)
    ax.set_aspect(1)  
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks([-0.5,-0.25,0])
    ax.set_xticks([-0.5,-0.25,0])
    ax.set_ylabel('Delta acc. shuffled spike rate')
    ax.set_xlabel('Delta acc. shuffled STP')
    plt.tight_layout()
    plt.savefig('Summary2.pdf', format='pdf')    
    plt.show()
    
    return spike_decoding, synapse_decoding, spike_decoding_test, synapse_decoding_test, perf, perf_shuffled_hidden, perf_shuffled_stp, ind_example


def get_perf(y, y_hat, mask, rule):

    """
    only examine time points when test stimulus is on
    in another words, when y[0,:,:] is not 0
    """
    print('Neural analysis: get_perf')
    print(y.shape, y_hat.shape, mask.shape)
    mask *= np.logical_or(y[1,:,:]>0,y[2,:,:]>0)
    #mask *= y[0,:,:]==0

    y = np.argmax(y, axis = 0)
    y_hat = np.argmax(y_hat, axis = 0)

    return np.sum(np.float32(y == y_hat)*np.squeeze(mask))/np.sum(mask)
    
   