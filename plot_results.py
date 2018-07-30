import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats
from parameters import *
from itertools import product
import stimulus
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "arial"


def plot_all_figures():

    fig_params = {
        'data_dir'              : './savedir_2000batches/',
        'dt'                    : 10,
        'models_per_task'       : 20,
        'N'                     : 100, # bootstrap iterations
        'accuracy_th'           : 0.9} # minimum accuracy of model required for analysis


    plot_summary_figure(fig_params)



def plot_SF4(fig_params):


    chance_level = 1/8
    #tasks = ['DMS_stp_fast', 'DMS_delay1500', 'DMS_delay2000', 'DMS_delay2500']
    tasks = ['DMS', 'DMS_stp_fast']
    num_tasks = len(tasks)
    model_signficance = np.zeros((num_tasks))
    f = plt.figure(figsize=(6,3.75))
    p_val = 0.025

    for n in range(num_tasks):

        #t = range(-750,2000+n*500,fig_params['dt'])
        #delay_epoch = range((2150+n*500)//fig_params['dt'],(2250)//fig_params['dt'])
        t = range(-750,2000,fig_params['dt'])
        delay_epoch = range((2150)//fig_params['dt'],(2250)//fig_params['dt'])

        # load following results from each task
        delay_accuracy = np.zeros((fig_params['models_per_task']), dtype=np.float32)
        neuronal_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        synaptic_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)

        good_model_count = 0
        count = 0
        while good_model_count < fig_params['models_per_task'] and count < 50:
            task_name = tasks[n] + '_' + str(count)
            try:
                x = pickle.load(open(fig_params['data_dir'] + task_name + '.pkl', 'rb'))
            except:
                #print('not found: ',  task_name + '.pkl')
                count +=1
                continue
            count += 1
            #print(tasks[n], count, np.mean(x['accuracy']))
            if np.mean(x['accuracy']) >  0.9:
                delay_accuracy[good_model_count] = np.mean(x['neuronal_sample_decoding'][0,0,:,delay_epoch])
                neuronal_decoding[good_model_count,:,:] = x['neuronal_sample_decoding'][0,0,:,:]
                synaptic_decoding[good_model_count,:,:] = x['synaptic_sample_decoding'][0,0,:,:]
                good_model_count +=1
        print('number of models ', ' ', n, ' ', good_model_count)
        if good_model_count < fig_params['models_per_task']:
            print('Too few accurately trained models')

        model_signficance[n] = np.sum(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2) \
            >chance_level,axis=1)>1-p_val)

        ax = f.add_subplot(1, 2, n+1)
        for j in range(fig_params['models_per_task']):
            ax.plot(t,np.mean(neuronal_decoding[j,:,:],axis=0),'g')
            ax.plot(t,np.mean(synaptic_decoding[j,:,:],axis=0),'m')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
        #ax.set_xticks([0,500,2000+n*500])
        ax.set_xticks([0,500,2000])
        ax.set_ylim([0,1.02])
        #ax.set_xlim([-500,2000+n*500])
        ax.set_xlim([-500,2000])
        ax.plot([-900,4000],[chance_level,chance_level],'k--')
        ax.plot([0,0],[0,1],'k--')
        ax.plot([500,500],[0,1],'k--')
        #ax.plot([1500+n*500,1500+n*500],[0,1],'k--')
        ax.plot([1500,1500],[0,1],'k--')
        ax.set_ylabel('Decoding accuracy')
        ax.set_xlabel('Time relative to sample onset (ms)')



    plt.tight_layout()
    plt.savefig('FigS4.pdf', format='pdf')
    plt.show()


    print(model_signficance)

def plot_F4(fig_params):

    task = 'DMS+DMRS'
    task = 'DMS+DMC_seq4000'
    fig_params['data_dir'] = './savedir/'
    t = range(-750,2000,fig_params['dt'])
    delay_epoch = range(2150//fig_params['dt'],2250//fig_params['dt'])
    f = plt.figure(figsize=(6,4))
    chance_level = [1/8, 1/2]

    # for each task, we will measure model significance with respect to:
    # dim 1 = 0, neuronal decoding during delay_accuracy
    # dim 1 = 1, shuffled neuronal accuracy > chance
    # dim 1 = 2, shuffled neuronal accuracy < accuracy
    # dim 1 = 3, shuffled synaptic accuracy > chance
    # dim 1 = 4, shuffled synaptic accuracy < accuracy
    model_signficance = np.zeros((2, 5))
    p_val_th = 0.01

    sig_neuronal_delay = np.zeros((2, fig_params['models_per_task']))
    sig_decrease_neuronal_shuffling = np.zeros((2, fig_params['models_per_task']))
    sig_decrease_syn_shuffling = np.zeros((2, fig_params['models_per_task']))
    sig_syn_shuffling = np.zeros((2, fig_params['models_per_task']))

    delay_accuracy = np.zeros((2,fig_params['models_per_task']), dtype=np.float32)
    neuronal_decoding = np.zeros((2,fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
    synaptic_decoding = np.zeros((2,fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
    accuracy = np.zeros((2,fig_params['models_per_task'], fig_params['N']), dtype=np.float32)
    accuracy_neural_shuffled = np.zeros((2,fig_params['models_per_task'], fig_params['N']), dtype=np.float32)
    accuracy_syn_shuffled = np.zeros((2,fig_params['models_per_task'], fig_params['N']), dtype=np.float32)

    corr_decoding_neuronal_shuf = np.zeros((2,2))
    corr_decoding_syn_shuf = np.zeros((2,2))
    corr_neuronal_shuf_syn_shuf = np.zeros((2,2))

    good_model_count = 0
    count = 0
    while good_model_count < fig_params['models_per_task'] and count < 20:
        x = pickle.load(open(fig_params['data_dir'] + task + '_v' + str(count) + '.pkl', 'rb'))
        count += 1
        print(np.mean(x['accuracy'][0,:]), np.mean(x['accuracy'][1,:]))
        if np.mean(x['accuracy']) > fig_params['accuracy_th']:

            for j in range(2):
                delay_accuracy[j, good_model_count] = np.mean(x['neuronal_sample_decoding'][j,0,:,delay_epoch])
                neuronal_decoding[j, good_model_count,:,:] = x['neuronal_sample_decoding'][j,0,:,:]
                synaptic_decoding[j, good_model_count,:,:] = x['synaptic_sample_decoding'][j,0,:,:]
                accuracy[j, good_model_count,:] = x['accuracy'][j,:]
                accuracy_neural_shuffled[j, good_model_count,:] = x['accuracy_neural_shuffled'][j, :]
                accuracy_syn_shuffled[j, good_model_count,:] = x['accuracy_syn_shuffled'][j, :]
            good_model_count +=1

    if good_model_count < fig_params['models_per_task']:
        print('Too few accurately trained models ', good_model_count)

    for j  in range(2):

        chance_level = 1/8 if j == 0 else 1/2

        model_signficance[j, 0] = np.sum(np.mean(np.mean(neuronal_decoding[j,:,:,delay_epoch],axis=2) \
            >chance_level,axis=0)>1-p_val_th)
        model_signficance[j, 1] = np.sum(np.mean(accuracy_neural_shuffled[j,:,:]>0.5,axis=1)>1-p_val_th)
        model_signficance[j, 2] = np.sum(np.mean(accuracy[j,:,:] - accuracy_neural_shuffled[j,:,:]>0.5,axis=1)>1-p_val_th)
        model_signficance[j, 3] = np.sum(np.mean(accuracy_syn_shuffled[j,:,:]>0.5,axis=1)>1-p_val_th)
        model_signficance[j, 4] = np.sum(np.mean(accuracy[j,:,:] - accuracy_syn_shuffled[j,:,:]>0.5,axis=1)>1-p_val_th)

        sig_neuronal_delay[j,:] =np.mean(np.mean(neuronal_decoding[j,:,:,delay_epoch],axis=2)>chance_level,axis=0)>1-p_val_th
        sig_decrease_neuronal_shuffling[j,:] =  np.mean(accuracy[j,:,:]-accuracy_neural_shuffled[j,:,:]>0,axis=1)>1-p_val_th
        sig_decrease_syn_shuffling[j,:] =  np.mean(accuracy[j,:,:]-accuracy_syn_shuffled[j,:,:]>0,axis=1)>1-p_val_th
        sig_syn_shuffling[j,:] = np.mean(accuracy_syn_shuffled[j,:,:]>0.5,axis=1)>1-p_val_th

        corr_decoding_neuronal_shuf[j,:] = scipy.stats.pearsonr(np.mean(np.mean(neuronal_decoding[j,:,:,delay_epoch],axis=2),axis=0), \
            np.mean(accuracy_neural_shuffled[j,:,:],axis=1))
        corr_decoding_syn_shuf[j,:] = scipy.stats.pearsonr(np.mean(np.mean(neuronal_decoding[j,:,:,delay_epoch],axis=2),axis=0), \
            np.mean(accuracy_syn_shuffled[j,:,:],axis=1))
        corr_neuronal_shuf_syn_shuf[j,:] = scipy.stats.pearsonr(np.mean(accuracy_neural_shuffled[j,:,:],axis=1), \
            np.mean(accuracy_syn_shuffled[j,:,:],axis=1))

        ax = f.add_subplot(2, 2, 2*j+1)
        for n in range(fig_params['models_per_task']):
            ax.plot(t,np.mean(synaptic_decoding[j,n,:,:],axis=0),'m')
            ax.plot(t,np.mean(neuronal_decoding[j,n,:,:],axis=0),'g')


        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_xticks([0,500, 1000, 1250, 1500])
        ax.set_xticks([0,500,1500])
        ax.set_ylim([0,1.02])
        ax.set_xlim([-500,2000])
        ax.plot([-900,2000],[chance_level,chance_level],'k--')
        ax.plot([0,0],[0,1],'k--')
        ax.plot([500,500],[0,1],'k--')
        #ax.plot([1000,1000],[0,1],'k--')
        #ax.plot([1250,1250],[0,1],'k--')
        ax.plot([1500,1500],[0,1],'k--')
        ax.set_ylabel('Decoding accuracy')
        ax.set_xlabel('Time relative to sample onset (ms)')


        ax = f.add_subplot(2, 2, 2*j+2)
        ax.plot(delay_accuracy[j,:], np.mean(accuracy[j,:,:],axis=1),'b.')

        ax.plot(delay_accuracy[j,:], np.mean(accuracy_neural_shuffled[j,:,:],axis=1),'r.')
        ax.plot(delay_accuracy[j,:], np.mean(accuracy_syn_shuffled[j,:,:],axis=1),'c.')
        ax.plot([chance_level,chance_level],[0,1],'k--')
        ax.set_aspect(1/0.62)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks([0,0.5,0.6,0.7,0.8,0.9,1])
        ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_ylim([0.4,1.02])
        ax.set_ylabel('Task accuracy')
        ax.set_xlabel('Delay neuronal decoding')
    plt.tight_layout()
    plt.savefig('Fig4_dir_seq.pdf', format='pdf')
    plt.show()

    print('Number of models with delay neuronal decoding accuracy signficantly greater than chance')
    print(model_signficance)
    print('Correlations...')
    print(corr_decoding_neuronal_shuf)
    print(corr_decoding_syn_shuf)
    print(corr_neuronal_shuf_syn_shuf)
    print('Number of models for which neuronal decoding is at chance')
    print(np.sum(1-sig_neuronal_delay,axis=1))
    print('Number of models for which shuffling neuronal activity has no effect')
    print(np.sum(1-sig_decrease_neuronal_shuffling,axis=1))
    print('Number of models for which shuffling STP causes accuracy to fall to chance')
    print(np.sum(1-sig_syn_shuffling,axis=1))
    print('Number of models for which 3 above conditions are satisfied')
    print(np.sum((1-sig_neuronal_delay)*(1-sig_decrease_neuronal_shuffling)*(1-sig_syn_shuffling),axis=1))
    print('Number of models for which shuffling neuronal and synaptic activity decreases accuracy')
    print(np.sum((sig_decrease_neuronal_shuffling)*(sig_decrease_syn_shuffling),axis=1))
    print('Mean accuracy after neuronal shuffling...')
    print(np.mean(np.reshape(accuracy_neural_shuffled,(2,-1)),axis=1))
    print('Mean accuracy after synaptic shuffling...')
    print(np.mean(np.reshape(accuracy_syn_shuffled,(2,-1)),axis=1))



def plot_SF2_v2(fig_params):


    stim = stimulus.Stimulus()

    neuron_ind=[]
    neuron_ind.append(range(0,80,2))
    neuron_ind.append(range(1,80,2))
    neuron_ind.append(range(80,100,2))
    neuron_ind.append(range(81,100,2))
    phases=np.arange(-180,225,45)
    phases2=np.arange(0,225,45)
    delay_epoch = range(2150//fig_params['dt'],2250//fig_params['dt'])

    N = 20
    n_min = np.zeros((N,100,9))
    n_max = np.zeros((N,100,9))
    n_min_late = np.zeros((N,100,9))
    n_max_late = np.zeros((N,100,9))
    s_min = np.zeros((N,100,9))
    s_max = np.zeros((N,100,9))
    s_min_late = np.zeros((N,100,9))
    s_max_late = np.zeros((N,100,9))

    t_max = np.zeros((N,100,9))
    t_min = np.zeros((N,100,9))

    diff_min_n = np.zeros((N,100,5))
    diff_min_s = np.zeros((N,100,5))
    diff_min_t = np.zeros((N,100,5))
    diff_min_s_late = np.zeros((N,100,5))
    diff_max_n = np.zeros((N,100,5))
    diff_max_t = np.zeros((N,100,5))
    diff_max_s = np.zeros((N,100,5))
    diff_max_s_late = np.zeros((N,100,5))

    acc_shuffled = np.zeros((N,5))
    delay_decoding = np.zeros((N))

    for i in range(N):
        #x = pickle.load(open('/media/masse/MySSDataStor1/Short-Term-Synaptic-Plasticity/savedir_2000batches/' + 'DMRS90_grp_shuffle_' + str(i) + '.pkl','rb'))
        x = pickle.load(open('/media/masse/MySSDataStor1/Short-Term-Synaptic-Plasticity/savedir_2000batches_1/' + 'DMRS90ccw_' + str(i) + '.pkl','rb'))
        print(stim.motion_tuning.shape)
        pred_resp = np.dot(x['weights']['w_in'], stim.motion_tuning)
        acc_shuffled[i,1:] = np.mean(x['accuracy_syn_shuffled_grp'],axis=3)
        acc_shuffled[i,0] = np.mean(x['accuracy'])
        delay_decoding[i] = np.mean(x['neuronal_sample_decoding'][0,0,:,delay_epoch])
        print('Accuracy = ', np.mean(x['accuracy']))
        x['neuronal_sample_tuning'] = np.squeeze(x['neuronal_sample_tuning'])
        x['synaptic_sample_tuning'] = np.squeeze(x['synaptic_sample_tuning'])
        for k in range(100):
            s = np.mean(x['neuronal_sample_tuning'][k,:,75:125],axis=1)
            a2 = np.argmin(s)
            a1 = np.argmax(s)
            #s = x['neuronal_sample_tuning'][k,0,:,125]
            #a1 = np.argmin(s)
            #a2 = np.argmax(s)
            for j in range(9):
                n_min[i,k,j] = s[(a1+j+4)%8]
                n_max[i,k,j] = s[(a2+j+4)%8]
                t_min[i,k,j] = pred_resp[k,(a1+j+4)%8]
                t_max[i,k,j] = pred_resp[k,(a2+j+4)%8]
            for j in range(5):
                diff_min_n[i,k,j] = s[(a1-j+4)%8]-s[(a1+j+4)%8]
                diff_max_n[i,k,j] = s[(a2-j+4)%8]-s[(a2+j+4)%8]
                diff_min_t[i,k,j] = pred_resp[k,(a1-j+4)%8]-pred_resp[k,(a1+j+4)%8]
                diff_max_t[i,k,j] = pred_resp[k,(a2-j+4)%8]-pred_resp[k,(a2+j+4)%8]

            s = x['neuronal_sample_tuning'][k,:,225]
            for j in range(9):
                n_min_late[i,k,j] = s[(a1+j+4)%8]
                n_max_late[i,k,j] = s[(a2+j+4)%8]


            s = x['synaptic_sample_tuning'][k,:,125]
            for j in range(9):
                s_min[i,k,j] = s[(a1+j+4)%8]
                s_max[i,k,j] = s[(a2+j+4)%8]

            for j in range(5):
                diff_min_s[i,k,j] = s[(a1-j+4)%8]-s[(a1+j+4)%8]
                diff_max_s[i,k,j] = s[(a2-j+4)%8]-s[(a2+j+4)%8]

            s = x['synaptic_sample_tuning'][k,:,225]
            for j in range(9):
                s_min_late[i,k,j] = s[(a1+j+4)%8]
                s_max_late[i,k,j] = s[(a2+j+4)%8]

            for j in range(5):
                diff_min_s_late[i,k,j] = s[(a1-j+4)%8]-s[(a1+j+4)%8]
                diff_max_s_late[i,k,j] = s[(a2-j+4)%8]-s[(a2+j+4)%8]


    col = [[0,0,1], [1,0,0],[0,1,0], [1,165/255,0]]

    f = plt.figure(figsize=(8,4))
    for i in range(4):
        #ax = f.add_subplot(4, 5, i+1)
        ax = f.add_subplot(2, 4, 1)
        m = np.mean(np.nanmean(n_min[:,neuron_ind[i],:],axis=1),axis=0)
        sd = np.std(np.nanmean(n_min[:,neuron_ind[i],:],axis=1),axis=0)/np.sqrt(N)
        ax.plot(phases,m,color=col[i])
        ax.fill_between(phases,m-sd,m+sd,color=col[i]+[0.5])
        ax.set_ylim([0, 5.5])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlim([-180,180])
        ax.set_xticks([-180,-90,0,90,180])

        ax = f.add_subplot(2, 4, 2)
        m = np.mean(np.nanmean(n_min_late[:,neuron_ind[i],:],axis=1),axis=0)
        sd = np.std(np.nanmean(n_min_late[:,neuron_ind[i],:],axis=1),axis=0)/np.sqrt(N)
        ax.plot(phases,m,color=col[i])
        ax.fill_between(phases,m-sd,m+sd,color=col[i]+[0.5])
        ax.set_ylim([0, 5.5])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlim([-180,180])
        ax.set_xticks([-180,-90,0,90,180])

        #ax = f.add_subplot(4, 5, i+6)
        ax = f.add_subplot(2, 4, 3)
        m = np.mean(np.nanmean(s_min[:,neuron_ind[i],:],axis=1),axis=0)
        sd = np.std(np.nanmean(s_min[:,neuron_ind[i],:],axis=1),axis=0)/np.sqrt(N)
        ax.plot(phases,m,color=col[i])
        ax.fill_between(phases,m-sd,m+sd,color=col[i]+[0.5])
        ax.set_ylim([0.15, 0.45])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlim([-180,180])
        ax.set_xticks([-180,-90,0,90,180])

        #ax = f.add_subplot(4, 5, i+11)
        ax = f.add_subplot(2, 4, 4)
        m = np.mean(np.nanmean(s_min_late[:,neuron_ind[i],:],axis=1),axis=0)
        sd = np.std(np.nanmean(s_min_late[:,neuron_ind[i],:],axis=1),axis=0)/np.sqrt(N)
        ax.plot(phases,m,color=col[i])
        ax.fill_between(phases,m-sd,m+sd,color=col[i]+[0.5])
        ax.set_ylim([0.15, 0.45])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlim([-180,180])
        ax.set_xticks([-180,-90,0,90,180])


        #m = np.mean(diff_min_s[:,neuron_ind[i],:],axis=1)
        #p = scipy.stats.ttest_1samp(m,0)
        #print(' p = ', p)


    #ax = f.add_subplot(4, 5, 16)
    ax = f.add_subplot(2, 4, 5)
    #m = np.mean(acc_shuffled[:,1:],axis=0)
    #sd = np.std(acc_shuffled[:,1:],axis=0)/np.sqrt(20)
    for i in range(4):
        ax.plot((i+1)*np.ones((N)), acc_shuffled[:N,1+i],'.',color=col[i])
    #ax.bar([0,1,2,3], m,  yerr = sd)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim([0.5, 1])

    #m = acc_shuffled[:,1:] - np.tile(np.reshape(acc_shuffled[:,0],(N,1)),(1,4))
    P = np.zeros((5,2))
    for i in range(5):
        P[i,:] = scipy.stats.ttest_1samp(acc_shuffled[:,i] - acc_shuffled[:,-1], 0)
    print('shuffling behavior p = ', P)
    print('shape ', acc_shuffled.shape)

    P = np.zeros((4,2))
    #d = np.zeros((4,2))
    for i in range(4):
        P[i,:] = scipy.stats.ttest_rel(np.mean(diff_min_n[:,neuron_ind[i],2],axis=1), \
            np.mean(diff_min_n[:,neuron_ind[-1],2],axis=1))
        #d[i,j] = np.mean(np.mean(diff_min_n[:,neuron_ind[i],2],axis=1) - \
            #np.mean(diff_min_n[:,neuron_ind[j],2],axis=1))
    print('Assymetry syn late p = ', P)
    #print('Assymetry syn late diff = ', d)



    #ax = f.add_subplot(4, 5, 17)
    ax = f.add_subplot(2, 4, 6)
    plt.plot(delay_decoding, acc_shuffled[:,-1],'k.')
    ax.set_ylim([0.5, 1])
    ax.set_xlim([0.2, 1])
    ax.set_xticks([0.2,0.4,0.6,0.8,1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ind = np.argsort(delay_decoding)


    t_low = t_min[ind[:5], :, :]
    t_high = t_min[ind[15:], :, :]


    for i in range(4):
        #ax = f.add_subplot(4, 5, i+1)
        ax = f.add_subplot(2, 4, 7)
        m = np.mean(np.nanmean(t_high[:,neuron_ind[i],:],axis=1),axis=0)
        sd = np.std(np.nanmean(t_high[:,neuron_ind[i],:],axis=1),axis=0)/np.sqrt(N)
        ax.plot(phases,m,color=col[i])
        ax.fill_between(phases,m-sd,m+sd,color=col[i]+[0.5])
        #ax.set_ylim([0, 5.5])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlim([-180,180])
        ax.set_xticks([-180,-90,0,90,180])

        ax = f.add_subplot(2, 4, 8)
        m = np.mean(np.nanmean(t_low[:,neuron_ind[i],:],axis=1),axis=0)
        sd = np.std(np.nanmean(t_low[:,neuron_ind[i],:],axis=1),axis=0)/np.sqrt(N)
        ax.plot(phases,m,color=col[i])
        ax.fill_between(phases,m-sd,m+sd,color=col[i]+[0.5])
        #ax.set_ylim([0, 5.5])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlim([-180,180])
        ax.set_xticks([-180,-90,0,90,180])

        m = np.mean(diff_min_t[:,neuron_ind[i],:],axis=1)
        p = scipy.stats.ttest_1samp(m,0)[1]
        print('Group ', i, ' Tuning p = ', p)

    r, p = scipy.stats.pearsonr(delay_decoding, acc_shuffled[:,-1])
    print('R = ',r, ' P = ', p)

    plt.tight_layout()
    plt.savefig('FigS2.pdf', format='pdf')
    plt.show()


def plot_SF3(fig_params):

    stim = stimulus.Stimulus()
    delay_epoch = range(2150//fig_params['dt'],2250//fig_params['dt'])
    early_sample = 77
    late_sample = 100
    count = -1

    angles = np.exp(1j*np.arange(8)*2*np.pi/8)

    cat_ind = []
    cat_ind.append([[0,1,2,3],[4,5,6,7]])
    cat_ind.append([[1,2,3, 4],[5,6,7,0]])
    cat_ind.append([[2,3,4,5],[6,7,0,1]])
    cat_ind.append([[3,4,5,6],[7,0,1,2]])

    f = plt.figure(figsize=(6,2))

    neuronal_CTI = np.zeros((7,100,275))
    synaptic_CTI = np.zeros((7,100,275))
    tuning_CTI = np.zeros((7,100))

    exc_ind = range(0,80,1)
    inh_ind = range(80,100,1)

    for i in range(0,20):
        x = pickle.load(open(fig_params['data_dir'] + 'DMC_' + str(i) + '.pkl', 'rb'))
        pred_resp = np.matmul(x['weights']['w_in'], stim.motion_tuning)
        nd = x['neuronal_sample_decoding'][0,0,:,delay_epoch]
        if np.mean(np.mean(nd,axis=0)>1/2)<0.975 and count < 8:
            count += 1
            print(count)
            for n in range(0,par['n_hidden']):
                tuning_CTI[count, n] = calc_CTI(pred_resp[n,:], cat_ind[0], angles)
                for t in range(par['num_time_steps']):
                    neuronal_CTI[count,n,t] = calc_CTI(x['neuronal_sample_tuning'][n,0,:,t], cat_ind[0], angles)
                    synaptic_CTI[count,n,t] = calc_CTI(x['synaptic_sample_tuning'][n,0,:,t], cat_ind[0], angles)

            if count == 1:

                ax = f.add_subplot(1, 4, 1)
                ax.plot([3.5,3.5],[0, 7],'k--')
                ax.plot(x['neuronal_sample_tuning'][inh_ind[11],0,:,early_sample])
                ax.plot(x['neuronal_sample_tuning'][inh_ind[11],0,:,late_sample],'r')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                ax.set_xticks([0,2,4,6])

                ax = f.add_subplot(1, 4, 2)
                ax.plot([3.5,3.5],[0, 0.5],'k--')
                ax.set_xticks([0,2,4,6])
                ax.plot(x['synaptic_sample_tuning'][inh_ind[11],0,:,early_sample])
                ax.plot(x['synaptic_sample_tuning'][inh_ind[11],0,:,late_sample],'r')

                ax.set_ylim([0.25,0.48])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')

    t1 = range(-740,2010,10)
    exc_ind = range(0,100)
    ax = f.add_subplot(1, 4, 3)
    s0 = np.reshape(neuronal_CTI[:,exc_ind,:],(7*len(exc_ind), 275),'F')
    s1 = np.reshape(synaptic_CTI[:,exc_ind,:],(7*len(exc_ind), 275),'F')
    exc_p_val = np.zeros((275))
    for t in range(50,150):
        _,exc_p_val[t] = scipy.stats.ttest_ind(s0[:,t],s1[:,t])
    u0 = np.mean(s0,axis=0)
    u1 = np.mean(s1,axis=0)
    sd0 = np.std(s0,axis=0)/np.sqrt(7*len(exc_ind))
    sd1 = np.std(s1,axis=0)/np.sqrt(7*len(exc_ind))
    ax.plot(t1,u0,color=[0,1,0])
    ax.fill_between(t1, u0-sd0, u0+sd0, color=[0,1,0,0.5])
    ax.plot([-200,400],[0,0],'k--')
    ax.plot([-0,0],[-1,1],'k--')
    ax.plot(t1,u1,color=[1,0,1])
    ax.fill_between(t1, u1-sd1, u1+sd1, color=[1,0,1,0.5])
    for t in range(50,150):
        if exc_p_val[t] < 0.01:
            ax.plot([t1[t]-5, t1[t]+5], [0.28, 0.28],'k-')
    ax.set_xlim([-100, 300])
    ax.set_ylim([-0.05, 0.3])
    ax.set_xticks([-100, 0, 100, 200, 300])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')



    print('exc_p_val', exc_p_val[70:80])
    print('tuning mean')
    print(np.mean(tuning_CTI))
    ax = f.add_subplot(1, 4, 4)

    s0 = np.reshape(tuning_CTI[:,:80],(80*7))
    s1 = np.reshape(tuning_CTI[:,80:],(20*7))
    u = [np.mean(s0), np.mean(s1)]
    sd = [np.std(s0)/np.sqrt(80*7), np.std(s1)/np.sqrt(20*7)]
    #ax.bar([1,2],u,yerr = sd)
    ax.plot(np.ones((7*80)), s0, 'b.')
    ax.plot(2*np.ones((7*20)), s1, 'r.')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    _,p0 = scipy.stats.ttest_1samp(s0,0)
    _,p1 = scipy.stats.ttest_1samp(s1,0)
    print('Tuning CTI p vals ', p0, p1)

    plt.tight_layout()
    plt.savefig('FigS3.pdf', format='pdf')
    plt.show()

def calc_CTI(s, ind, angles):

    N = len(s)
    between = []
    within = []

    for i in range(N-1):
        for j in range(i+1, N):
            ang_diff = int(np.abs(180/np.pi*np.angle(angles[i]/angles[j])))
            value_diff = np.abs(s[i]-s[j])

            # within
            if (i in ind[0] and j in ind[0]) or (i in ind[1] and j in ind[1]):
                within.append([ang_diff, value_diff])
            else:
                between.append([ang_diff, value_diff])

    within = np.array(np.stack(within))
    between = np.array(np.stack(between))

    unique_diffs = np.unique(within[:,0])

    b = 0
    w = 0

    for d in unique_diffs:
        ind0 = np.where(within[:,0]==d)[0]
        ind1 = np.where(between[:,0]==d)[0]
        if len(ind0) > 0 and len(ind1)>0:
            for i in ind0:
                w += within[i,1]/len(ind0)
            for i in ind1:
                b += between[i,1]/len(ind1)

    return (b-w)/(1e-9+b+w)

def plot_summary_figure(fig_params):


    d = '/media/masse/MySSDataStor1/Short-Term-Synaptic-Plasticity/Analysis/'
    tasks = [ 'DMS','DMRS90','DMRS45','DMRS180','ABCA','ABBA','DMS+DMRS','dualDMS']
    """
    """
    d = '/home/masse/Short-term-plasticity-RNN/savedir_spike_cost/'
    sc = '2'
    tasks = [ 'DMS_spike_cost','DMRS90_spike_cost','DMRS45_spike_cost','DMRS180_spike_cost',\
         'ABCA_spike_cost', 'ABBA_spike_cost', 'DMS+DMRS_spike_cost', 'dualDMS_spike_cost']


    num_tasks = len(tasks)
    de1 = range(2150//fig_params['dt'],2250//fig_params['dt']) # DMS, DMRS, DMS+DMRS
    de0 = range(2150//fig_params['dt'],2250//fig_params['dt']) # DMS, DMRS, DMS+DMRS
    delay_epoch = [de1] * num_tasks
    delay_epoch_man = [de0] * num_tasks

    #delay_epoch = [de1,de1,de1,de3,]
    neuron_ind = [range(0,100,2), range(1,100,2)]
    fig = plt.figure(figsize=(4,4))

    delay_accuracy = np.zeros((num_tasks+2, fig_params['models_per_task']))
    manipulation = np.zeros((num_tasks+2, fig_params['models_per_task']))

    fn = os.listdir(d)

    for n in range(num_tasks):
        count = 0
        for f in fn:

            if f.startswith(tasks[n] + sc + '_'):
            #if f.startswith(tasks[n] + '_'):
                print(f)
                x = pickle.load(open(d + f, 'rb'))
                #print(x.keys())
                if not 'neuronal_pev' in x.keys():
                    continue
                neuronal = np.sqrt(1e-9+x['neuronal_pev'])*np.exp(1j*x['neuronal_pref_dir'])
                synaptic = np.sqrt(1e-9+x['synaptic_pev'])*np.exp(1j*x['synaptic_pref_dir'])

                if tasks[n].startswith('DMS+DMRS'):
                    delay_accuracy[n,count] = np.mean(x['neuronal_sample_decoding'][0,0,:,delay_epoch[n]])
                    delay_accuracy[n+1,count] = np.mean(x['neuronal_sample_decoding'][1,0,:,delay_epoch[n]])
                    s1 = np.mean(synaptic[:,:,0,delay_epoch_man[n]],axis=2)
                    #n1 = np.mean(neuronal[:,:,0, 75:85],axis=2)
                    n1 = np.mean(neuronal[:,:,0, 75:85],axis=2)
                    for r,j in product(range(2), range(2)):
                        manipulation[n+r, count] += ((-1)**j)*np.mean(np.real(s1[neuron_ind[j],r]*np.conj(n1[neuron_ind[j],r])))/ \
                            np.mean(np.abs(s1[neuron_ind[j],r])*np.abs(n1[neuron_ind[j],r]))/2



                elif tasks[n].startswith('dualDMS'):
                    delay_accuracy[n+1,count] = np.mean(x['neuronal_sample_decoding'][0,0,:,delay_epoch[n]])/2 + \
                        np.mean(x['neuronal_sample_decoding'][3,1,:,delay_epoch[n]])/2
                    delay_accuracy[n+2,count] = np.mean(x['neuronal_sample_decoding'][1,0,:,delay_epoch[n]])/2 + \
                        np.mean(x['neuronal_sample_decoding'][2,1,:,delay_epoch[n]])/2

                    print(delay_accuracy[n+1,count], delay_accuracy[n+2,count])


                    c = np.zeros((2,4))
                    for m in range(4):
                        for k in range(2):
                            s1 = np.mean(synaptic[:,m,k,delay_epoch_man[n]],axis=1)
                            n1 = np.mean(neuronal[:,m,k, 75:85],axis=1)
                            for j in range(2):
                                c[k,m] += ((-1)**j)*np.mean(np.real(s1[neuron_ind[j]]*np.conj(n1[neuron_ind[j]])))/ \
                                    np.mean(np.abs(s1[neuron_ind[j]])*np.abs(n1[neuron_ind[j]]))/2

                    manipulation[n+1, count] = (c[0,0] +  c[1,3])/2
                    manipulation[n+2, count] = (c[0,1] +  c[1,2])/2
                else:
                    delay_accuracy[n,count] = np.mean(x['neuronal_sample_decoding'][:,:,:,delay_epoch[n]])
                    s1 = np.mean(synaptic[:,:,:,delay_epoch_man[n]],axis=3, keepdims = True)
                    n1 = np.mean(neuronal[:,:, :,75:85],axis=3, keepdims = True)
                    for j in range(2):
                        manipulation[n, count] += ((-1)**j)*np.mean(np.real(s1[neuron_ind[j],:]*np.conj(n1[neuron_ind[j],:])))/ \
                            np.mean(np.abs(s1[neuron_ind[j],:])*np.abs(n1[neuron_ind[j],:]))/2
                        #manipulation[n, count] += ((-1)**j)*np.mean(np.real(s1[neuron_ind[j],:]*np.conj(n1[neuron_ind[j],:]))/np.abs(s1[neuron_ind[j],:])/np.abs(n1[neuron_ind[j],:]))/2
                count += 1

    u0 = np.zeros((num_tasks+2))
    u1 = np.zeros((num_tasks+2))
    sd0 = np.zeros((num_tasks+2))
    sd1 = np.zeros((num_tasks+2))


    for n in range(num_tasks+2):
        ind = np.where(delay_accuracy[n,:] > 0)[0]
        print(n, ind)
        u0[n] = np.mean(delay_accuracy[n,ind])
        u1[n] = np.mean(manipulation[n,ind])
        sd0[n] = np.std(delay_accuracy[n,ind])/np.sqrt(len(ind))
        sd1[n] = np.std(manipulation[n,ind])/np.sqrt(len(ind))


    print(u0)
    print(u1)
    print(sd0)
    print(sd1)

    r, p = scipy.stats.spearmanr(u0, 1-u1)
    print('Spearman Corr R = ', r, ' P = ',p)
    r, p = scipy.stats.pearsonr(u0, 1-u1)
    print('Pearson Corr R = ', r, ' P = ',p)
    """
    ax = fig.add_subplot(1,3,1)
    ax.errorbar(u0[:3],u1[:3],xerr = sd0[:3],yerr = sd1[:3])
    ax = fig.add_subplot(1,3,2)
    ax.errorbar(u0[3:5],u1[3:5],xerr = sd0[3:5],yerr = sd1[3:5])
    ax = fig.add_subplot(1,3,3)
    ax.errorbar(u0[5:],u1[5:],xerr = sd0[5:],yerr = sd1[5:])
    plt.show()
    """

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    col = ['b','y','r','r','k','g','b','k','m','c','y']
    #col = ['b','y','r','k','g','b','k','m','c']

    print('FINAL RESULTS')
    print('decoding ', u0)
    print('manipulation ', 1-u1)
    for i in range(num_tasks+2):
        ax.errorbar(u0[i],1-u1[i],xerr = sd0[i],yerr = sd1[i], fmt='o', color = col[i])

    ax.set_xlabel('Delay decoding accuracy')
    ax.set_ylabel('Manipulation')
    #ax.legend(['DMS','DMC','DMRS','DMRS45','DMRS180','DMS+DMRS','ABCA','ABBA','Attended','Unattended'])
    ax.legend(['DMS','DMRS','DMRS45','DMRS180','ABCA','ABBA','DMS+DMRS (DMS)','DMS+DMRS (DMRS)','Attended','Unattended'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.plot([0.125,0.125],[0,1],'k--')
    #ax.set_xlim([0,0.7])
    #ax.set_ylim([0,0.32])
    #ax.set_aspect(0.7/0.32)
    plt.tight_layout()
    plt.savefig('Fig_summary.pdf', format='pdf')
    plt.show()



def plot_F3(fig_params):

    tasks = [ 'DMS','DMRS90_grp_shuffle','DMC']
    num_tasks = len(tasks)

    #t = range(-900,2000,fig_params['dt'])
    t = range(-750,2000,fig_params['dt'])
    #delay_epoch = range(2300//fig_params['dt'],2400//fig_params['dt'])
    delay_epoch = range(2150//fig_params['dt'],2250//fig_params['dt'])

    f = plt.figure(figsize=(6,6))

    p_neuronal_delay = np.zeros((num_tasks, fig_params['models_per_task']))
    p_decrease_neuronal_shuffling = np.zeros((num_tasks, fig_params['models_per_task'])) # comparison to no shuffling
    p_decrease_synaptic_shuffling = np.zeros((num_tasks, fig_params['models_per_task'])) # comparison to  no shuffling
    p_neuronal_shuffling = np.zeros((num_tasks, fig_params['models_per_task'])) # comparison to chance
    p_synaptic_shuffling = np.zeros((num_tasks, fig_params['models_per_task'])) # comparison to chance

    accuracy_suppression = np.zeros((num_tasks, fig_params['models_per_task'], 17))
    delay_accuracy = np.zeros((num_tasks, fig_params['models_per_task']))
    accuracy = np.zeros((num_tasks, fig_params['models_per_task'], fig_params['N']))
    accuracy_neural_shuffled = np.zeros((num_tasks, fig_params['models_per_task'], fig_params['N']))
    accuracy_syn_shuffled = np.zeros((num_tasks, fig_params['models_per_task'], fig_params['N']))

    # correlation between neuronal decoding during the delay, accuracy after sig_syn_shuffling
    # neuronal activity, and accuracy after shuffling synaptic activity
    corr_decoding_neuronal_shuf =  np.zeros((num_tasks,2))
    corr_decoding_syn_shuf =  np.zeros((num_tasks,2))
    corr_neuronal_shuf_syn_shuf =  np.zeros((num_tasks,2))

    decoding_p_val =  np.zeros((num_tasks, 2))


    p_val_th = 0.025

    # will use DMS decoding results for comparison
    neuronal_decoding_DMS  = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)

    for n in range(num_tasks):

        if tasks[n] == 'DMC':
            chance_level = 1/2
        else:
            chance_level = 1/8

        # load following results from each task
        neuronal_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        synaptic_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        mean_resp = np.zeros((fig_params['models_per_task'], len(t)), dtype=np.float32)

        good_model_count = 0
        count = 0
        while good_model_count < fig_params['models_per_task'] and count<24:
            x = pickle.load(open(fig_params['data_dir'] + tasks[n] + '_' + str(count) + '.pkl', 'rb'))
            count += 1
            #print(count, np.mean(x['accuracy']))
            if np.mean(x['accuracy']) >  fig_params['accuracy_th']:

                delay_accuracy[n,good_model_count] = np.mean(x['neuronal_sample_decoding'][0,0,:,delay_epoch])
                neuronal_decoding[good_model_count,:,:] = x['neuronal_sample_decoding'][0,0,:,:]

                #mean_resp[good_model_count,:] = np.mean(np.reshape(x['neuronal_sample_tuning'],(-1,len(t))),axis=0)

                if tasks[n] == 'DMS':
                    neuronal_decoding_DMS[good_model_count,:,:] = x['neuronal_sample_decoding'][0,0,:,:]
                    #neuronal_decoding_DMS[good_model_count,:,:] = x['neuronal_decoding'][0,:,:]

                synaptic_decoding[good_model_count,:,:] = x['synaptic_sample_decoding'][0,0,:,:]
                #synaptic_decoding[good_model_count,:,:] = x['synaptic_decoding'][0,:,:]


                accuracy[n, good_model_count,:] = x['accuracy']
                accuracy_neural_shuffled[n, good_model_count,:] = x['accuracy_neural_shuffled']
                accuracy_syn_shuffled[n, good_model_count,:] = x['accuracy_syn_shuffled']
                if tasks[n] == 'DMS':
                    accuracy_suppression[n, good_model_count, :]  = x['accuracy_suppression'][0,:,0]
                good_model_count +=1


        if good_model_count < fig_params['models_per_task']:
            print('Too few accurately trained models')

        print(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2)>chance_level,axis=1))

        p_neuronal_delay[n,:] =np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2)>chance_level,axis=1)>1-p_val_th
        p_decrease_neuronal_shuffling[n,:] =  np.mean(accuracy[n,:,:]-accuracy_neural_shuffled[n,:,:]>0,axis=1)>1-p_val_th
        p_decrease_synaptic_shuffling[n,:] =  np.mean(accuracy[n,:,:]-accuracy_syn_shuffled[n,:,:]>0,axis=1)>1-p_val_th
        p_neuronal_shuffling[n,:] = np.mean(accuracy_neural_shuffled[n,:,:]>0.5,axis=1)>1-p_val_th
        p_synaptic_shuffling[n,:] = np.mean(accuracy_syn_shuffled[n,:,:]>0.5,axis=1)>1-p_val_th

        corr_decoding_neuronal_shuf[n,:] = scipy.stats.pearsonr(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2),axis=1), \
            np.mean(accuracy_neural_shuffled[n,:,:],axis=1))
        corr_decoding_syn_shuf[n,:] = scipy.stats.pearsonr(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2),axis=1), \
            np.mean(accuracy_syn_shuffled[n,:,:],axis=1))
        corr_neuronal_shuf_syn_shuf[n,:] = scipy.stats.pearsonr(np.mean(accuracy_neural_shuffled[n,:,:],axis=1), \
            np.mean(accuracy_syn_shuffled[n,:,:],axis=1))

        decoding_p_val[n,:] = scipy.stats.ttest_ind(np.mean(np.mean(neuronal_decoding_DMS[:,:,delay_epoch],axis=2),axis=1), \
            np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2),axis=1))


        ax = f.add_subplot(num_tasks, 2, 2*n+1)
        for j in range(fig_params['models_per_task']):
            ax.plot(t,np.mean(neuronal_decoding[j,:,:],axis=0),'g')
            ax.plot(t,np.mean(synaptic_decoding[j,:,:],axis=0),'m')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_xticks([0,500,2000])
        ax.set_ylim([0,1.02])
        ax.set_xlim([-500,2000])
        ax.plot([-900,2000],[chance_level,chance_level],'k--')
        ax.plot([0,0],[0,1],'k--')
        ax.plot([500,500],[0,1],'k--')
        ax.plot([1500,1500],[0,1],'k--')
        ax.set_ylabel('Decoding accuracy')
        ax.set_xlabel('Time relative to sample onset (ms)')


        ax = f.add_subplot(num_tasks, 2, 2*n+2)
        ax.plot(delay_accuracy[n,:], np.mean(accuracy[n,:,:],axis=1),'b.')
        ax.plot(delay_accuracy[n,:], np.mean(accuracy_neural_shuffled[n,:,:],axis=1),'r.')
        ax.plot(delay_accuracy[n,:], np.mean(accuracy_syn_shuffled[n,:,:],axis=1),'c.')
        ax.plot([chance_level,chance_level],[0,1],'k--')
        ax.set_aspect(1.02/0.62)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks([0,0.5,0.6,0.7,0.8,0.9,1])
        ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_ylim([0.4,1.02])
        ax.set_xlim([0,1.02])
        ax.set_ylabel('Behavioral accuracy')
        ax.set_xlabel('Delay neuronal decoding')
    plt.tight_layout()
    plt.savefig('Fig3.pdf', format='pdf')
    plt.show()

    num_tasks = 1

    for n in range(num_tasks):
        r1, p1 = scipy.stats.pearsonr(delay_accuracy[n,:], accuracy_suppression[n,:,1])
        print(tasks[n], ' Corr suppression - neuronal decoding ',r1 , p1)
        r1, p1 = scipy.stats.pearsonr(np.mean(accuracy_neural_shuffled[n,:,:],axis=1), accuracy_suppression[n,:,1])
        print(tasks[n], ' Corr suppression - accuracy shuffled ',r1 , p1)


    f = plt.figure(figsize=(6,2.5))
    for n in range(num_tasks):
        """
        if n == num_tasks-1:
            chance_level = 1/2
        else:
            chance_level = 1/8
        """
        chance_level = 1/8
        ax = f.add_subplot(num_tasks, 1, 1*n+1)
        ax.plot(delay_accuracy[n,:], accuracy_suppression[n,:,1],'k.')
        ax.plot([chance_level,chance_level],[0,1],'k--')
        ax.set_aspect(1.02)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks([0.5,0.6,0.7,0.8,0.9,1])
        ax.set_xticks([0,0.1,0.2,0.3,0.4,0.5])
        ax.set_ylim([0.5,1])
        ax.set_xlim([0,0.5])
        if n == num_tasks-1:
            ax.set_ylabel('Behavioral accuracy')
            ax.set_xlabel('Delay neuronal decoding')



    plt.tight_layout()
    plt.savefig('FigSx.pdf', format='pdf')
    plt.show()


    print('Number of models with delay neuronal decoding accuracy signficantly greater than chance')
    #print(model_signficance)

    print('Number of models for which neuronal decoding is at chance')
    print(np.sum(1-p_neuronal_delay,axis=1))
    print('Number of models for which shuffling neuronal activity has no effect')
    print(np.sum(1-p_decrease_neuronal_shuffling,axis=1))
    print('Number of models for which shuffling STP causes accuracy to fall to chance')
    print(np.sum(1-p_synaptic_shuffling,axis=1))
    print('Number of models for which 3 above conditions are satisfied')
    print(np.sum((1-p_neuronal_delay)*(1-p_decrease_neuronal_shuffling)*(1-p_synaptic_shuffling),axis=1))
    print('Number of models for which shuffling neuronal and synaptic activity decreases accuracy')
    print(np.sum((p_decrease_neuronal_shuffling)*(p_decrease_synaptic_shuffling),axis=1))

    print('Correlations...')
    print(corr_decoding_neuronal_shuf)
    print(corr_decoding_syn_shuf)
    print(corr_neuronal_shuf_syn_shuf)
    print('T-test neuroanl decoding p-val compared to DMS')
    print(decoding_p_val)



def plot_S_learning(fig_params):

    N = 20
    m = 5
    v = np.ones((m))/m
    acc0 = []
    acc1 = []
    loss0 = []
    loss1 = []
    f = plt.figure(figsize=(6,5))
    for i in range(N):
        x = pickle.load(open(fig_params['data_dir']+'DMS_' + str(i) + '.pkl','rb'))
        s1 = np.convolve(x['model_performance']['accuracy'],v,'valid')
        acc0.append(s1)
        s1 = np.convolve(x['model_performance']['perf_loss'],v,'valid')
        loss0.append(s1)
        x = pickle.load(open(fig_params['data_dir']+'DMS_no_stp_' + str(i) + '.pkl','rb'))
        s1 = np.convolve(x['model_performance']['accuracy'],v,'valid')
        acc1.append(s1)
        s1 = np.convolve(x['model_performance']['perf_loss'],v,'valid')
        loss1.append(s1)
    acc0 = np.stack(acc0)
    acc1 = np.stack(acc1)
    loss0 = np.stack(loss0)
    loss1 = np.stack(loss1)


    ax = f.add_subplot(2,2,1)
    t = np.arange(0,(2000+1-m)*1024,1024)/1000
    t = np.transpose(np.tile(t, (20,1)))
    #print(np.transpose(acc0).shape)
    ax.plot(t,np.transpose(np.log10(loss0)))
    ax.set_xlim([0,2000*1024])
    ax.set_xlim([0,2000])
    ax.set_ylim([-2.5,1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Training batch number')

    ax = f.add_subplot(2,2,2)
    ax.plot(t,np.transpose(np.log10(loss1)))
    ax.set_xlim([0,2000*1024])
    ax.set_xlim([0,2000])
    ax.set_ylim([-2.5,1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Trial number')

    ax = f.add_subplot(2,2,3)
    #print(np.transpose(acc0).shape)
    ax.plot(t,np.transpose(acc0))
    ax.set_xlim([0,2000*1024])
    ax.set_xlim([0,2000])
    ax.set_ylim([0,1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel('Task accuracy')
    ax.set_xlabel('Training batch number')

    ax = f.add_subplot(2,2,4)
    ax.plot(t,np.transpose(acc1))
    ax.set_xlim([0,2000*1024])
    ax.set_xlim([0,2000])
    ax.set_ylim([0,1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel('Task accuracy')
    ax.set_xlabel('Trial number')

    plt.tight_layout()
    plt.savefig('FigS_learning.pdf', format='pdf')
    plt.show()

def plot_F6_v2(fig_params):

    task = 'dualDMS_fixed'
    t = range(-750,3500,fig_params['dt'])
    p_val_th = 0.025
    chance_level = 1/8

    example_ind = 99

    delay_accuracy = np.zeros((4, 2, fig_params['models_per_task']), dtype=np.float32)
    delay_sig_attended = np.zeros((2, 2, fig_params['models_per_task']), dtype=np.float32)
    delay_sig_unattended = np.zeros((2, 2, fig_params['models_per_task']), dtype=np.float32)
    neuronal_decoding = np.zeros((fig_params['models_per_task'], 4, fig_params['N'], len(t)), dtype=np.float32)
    synaptic_decoding = np.zeros((fig_params['models_per_task'], 4, fig_params['N'], len(t)), dtype=np.float32)
    neuronal_rule_decoding = np.zeros((fig_params['models_per_task'], 2, fig_params['N'], len(t)), dtype=np.float32)
    synaptic_rule_decoding = np.zeros((fig_params['models_per_task'], 2, fig_params['N'], len(t)), dtype=np.float32)
    accuracy = np.zeros((fig_params['models_per_task']), dtype=np.float32)
    accuracy_neural_shuffled = np.zeros((fig_params['models_per_task']), dtype=np.float32)
    accuracy_syn_shuffled = np.zeros((fig_params['models_per_task']), dtype=np.float32)

    delay_epoch_post_cue = []
    delay_epoch_post_cue.append(range(25+50+50+90, 25+50+50+100)) # last 100 ms of first delay period , dt=10ms
    delay_epoch_post_cue.append(range(25+50+50+100+50+90, 25+50+50+100+50+100)) # last 100 ms of second delay period , dt=10ms

    delay_epoch_pre_cue = []
    delay_epoch_pre_cue.append(range(25+50+50+40, 25+50+50+50)) # last 100 ms of first delay period , dt=10ms
    delay_epoch_pre_cue.append(range(25+50+50+100+50+40, 25+50+50+100+50+50))

    neuronal_rule_decoding = np.zeros((fig_params['models_per_task'], 2, fig_params['N'], len(t)), dtype=np.float32)
    neuronal_decoding = np.zeros((2*fig_params['models_per_task'], 6, fig_params['N'], len(t)), dtype=np.float32)
    #neuronal_decoding_probe = np.zeros((2*fig_params['models_per_task'], 4, fig_params['N'], len(t)), dtype=np.float32)
    synaptic_rule_decoding = np.zeros((fig_params['models_per_task'], 2, fig_params['N'], len(t)), dtype=np.float32)
    synaptic_decoding = np.zeros((2*fig_params['models_per_task'], 6, fig_params['N'], len(t)), dtype=np.float32)
    sig_pre_cue = np.zeros((2*fig_params['models_per_task'], 6, 2))
    sig_post_cue = np.zeros((2*fig_params['models_per_task'], 6, 2))

    good_model_count = -1
    count = -1

    while good_model_count < fig_params['models_per_task'] and count < 19:
        count += 1
        try:
            x = pickle.load(open(fig_params['data_dir'] + task + '_' + str(count) + '.pkl', 'rb'))
            #x1 = pickle.load(open(fig_params['data_dir'] + 'probe_' + task + '_' + str(count) + '.pkl', 'rb'))
            #print(fig_params['data_dir'] + task + '_' + str(count) + '.pkl', np.mean(x['accuracy']))
        except:
            #print('not found: ',  fig_params['data_dir'] + task + '_' + str(count) + '.pkl')
            continue
        good_model_count += 1

        # rule decoding
        #print(count, x.keys())
        neuronal_rule_decoding[good_model_count,:,:,:] = x['neuronal_rule_decoding']
        synaptic_rule_decoding[good_model_count,:,:,:] = x['synaptic_rule_decoding']

        # first stim cued
        neuronal_decoding[2*good_model_count,0,:,:] = x['neuronal_sample_decoding'][0,0,:,:]/2 + x['neuronal_sample_decoding'][2,0,:,:]/2
        neuronal_decoding[2*good_model_count+1,0,:,:] = x['neuronal_sample_decoding'][1,1,:,:]/2 + x['neuronal_sample_decoding'][3,1,:,:]/2
        synaptic_decoding[2*good_model_count,0,:,:] = x['synaptic_sample_decoding'][0,0,:,:]/2 + x['synaptic_sample_decoding'][2,0,:,:]/2
        synaptic_decoding[2*good_model_count+1,0,:,:] = x['synaptic_sample_decoding'][1,1,:,:]/2 + x['synaptic_sample_decoding'][3,1,:,:]/2

        # first stim not cued
        neuronal_decoding[2*good_model_count,1,:,:] = x['neuronal_sample_decoding'][1,0,:,:]/2 + x['neuronal_sample_decoding'][3,0,:,:]/2
        neuronal_decoding[2*good_model_count+1,1,:,:] = x['neuronal_sample_decoding'][0,1,:,:]/2 +  x['neuronal_sample_decoding'][2,1,:,:]/2
        synaptic_decoding[2*good_model_count,1,:,:] = x['synaptic_sample_decoding'][1,0,:,:]/2 + x['synaptic_sample_decoding'][3,0,:,:]/2
        synaptic_decoding[2*good_model_count+1,1,:,:] = x['synaptic_sample_decoding'][0,1,:,:]/2 + x['synaptic_sample_decoding'][2,1,:,:]/2

        # first stim not cued, second stim cued
        neuronal_decoding[2*good_model_count,2,:,:] = x['neuronal_sample_decoding'][1,0,:,:]
        neuronal_decoding[2*good_model_count+1,2,:,:] = x['neuronal_sample_decoding'][2,1,:,:]
        synaptic_decoding[2*good_model_count,2,:,:] = x['synaptic_sample_decoding'][1,0,:,:]
        synaptic_decoding[2*good_model_count+1,2,:,:] = x['synaptic_sample_decoding'][2,1,:,:]

        # first stim not cued, second stim not cued
        neuronal_decoding[2*good_model_count,3,:,:] = x['neuronal_sample_decoding'][3,0,:,:]
        neuronal_decoding[2*good_model_count+1,3,:,:] = x['neuronal_sample_decoding'][0,1,:,:]
        synaptic_decoding[2*good_model_count,3,:,:] = x['synaptic_sample_decoding'][3,0,:,:]
        synaptic_decoding[2*good_model_count+1,3,:,:] = x['synaptic_sample_decoding'][0,1,:,:]

        # second stim cued
        neuronal_decoding[2*good_model_count,4,:,:] = x['neuronal_sample_decoding'][1,0,:,:]/2 + x['neuronal_sample_decoding'][0,0,:,:]/2
        neuronal_decoding[2*good_model_count+1,4,:,:] = x['neuronal_sample_decoding'][2,1,:,:]/2 + x['neuronal_sample_decoding'][3,1,:,:]/2
        synaptic_decoding[2*good_model_count,4,:,:] = x['synaptic_sample_decoding'][1,0,:,:]/2 + x['synaptic_sample_decoding'][0,0,:,:]/2
        synaptic_decoding[2*good_model_count+1,4,:,:] = x['synaptic_sample_decoding'][2,1,:,:]/2 + x['synaptic_sample_decoding'][3,1,:,:]/2

        # second stim not cued
        neuronal_decoding[2*good_model_count,5,:,:] = x['neuronal_sample_decoding'][2,0,:,:]/2 + x['neuronal_sample_decoding'][3,0,:,:]/2
        neuronal_decoding[2*good_model_count+1,5,:,:] = x['neuronal_sample_decoding'][0,1,:,:]/2 + x['neuronal_sample_decoding'][1,1,:,:]/2
        synaptic_decoding[2*good_model_count,5,:,:] = x['synaptic_sample_decoding'][2,0,:,:]/2 + x['synaptic_sample_decoding'][3,0,:,:]/2
        synaptic_decoding[2*good_model_count+1,5,:,:] = x['synaptic_sample_decoding'][0,1,:,:]/2 + x['synaptic_sample_decoding'][1,1,:,:]/2



    for j in range(2):
        s = np.mean(neuronal_decoding[:,:,:,delay_epoch_pre_cue[j]],axis=3)
        sig_pre_cue[:, :, j] = np.mean(s > chance_level,axis=2) > 1 - p_val_th
        s = np.mean(neuronal_decoding[:,:,:,delay_epoch_post_cue[j]],axis=3)
        sig_post_cue[:, :, j] = np.mean(s > chance_level,axis=2) > 1 - p_val_th
        if j == 1:
            ind_uattended = np.where(sig_pre_cue[:, 2, 1]==0)[0]
            print('ind_uattended',ind_uattended)


    print(np.sum(sig_pre_cue,axis=0))
    print(np.sum(sig_post_cue,axis=0))

    f = plt.figure(figsize=(6,4))

    n_mean = np.mean(np.mean(neuronal_decoding[:,:,:,:],axis=2),axis=0)
    n_sd = np.std(np.mean(neuronal_decoding[:,:,:,:],axis=2),axis=0)/np.sqrt(40)
    #n_probe_mean = np.mean(np.mean(neuronal_decoding_probe[:,:,:,:],axis=2),axis=0)
    #n_probe_sd = np.std(np.mean(neuronal_decoding_probe[:,:,:,:],axis=2),axis=0)/np.sqrt(40)
    s_mean = np.mean(np.mean(synaptic_decoding[:,:,:,:],axis=2),axis=0)
    s_sd = np.std(np.mean(synaptic_decoding[:,:,:,:],axis=2),axis=0)/np.sqrt(40)

    print('n_mean', n_mean.shape)

    ax = f.add_subplot(2, 2, 1)
    ax.plot(t,n_mean[0,:],color=[0,0,1])
    ax.fill_between(t, n_mean[0,:]-n_sd[0,:], n_mean[0,:]+n_sd[0,:], color=[0,0,1,0.2])
    ax.plot(t,n_mean[1,:],color=[1,0,0])
    ax.fill_between(t, n_mean[1,:]-n_sd[1,:], n_mean[1,:]+n_sd[1,:], color=[1,0,0,0.2])
    ax.plot(t,s_mean[0,:],color=[1,1,0])
    ax.fill_between(t, s_mean[0,:]-s_sd[0,:], s_mean[0,:]+s_sd[0,:], color=[1,1,0,0.2])
    ax.plot(t,s_mean[1,:],color=[0,0,0])
    ax.fill_between(t, s_mean[1,:]-s_sd[1,:], s_mean[1,:]+s_sd[1,:], color=[0,0,0,0.2])
    add_dualDMS_subplot_details(ax, chance_level)
    ax.set_xlim([-500,2000])

    ax = f.add_subplot(2, 2, 2)
    ax.plot(t,n_mean[4,:],color=[0,0,1])
    ax.fill_between(t, n_mean[4,:]-n_sd[4,:], n_mean[4,:]+n_sd[4,:], color=[0,0,1,0.2])
    ax.plot(t,n_mean[5,:],color=[1,0,0])
    ax.fill_between(t, n_mean[5,:]-n_sd[5,:], n_mean[5,:]+n_sd[5,:], color=[1,0,0,0.2])
    ax.plot(t,s_mean[4,:],color=[1,1,0])
    ax.fill_between(t, s_mean[4,:]-s_sd[4,:], s_mean[4,:]+s_sd[4,:], color=[1,1,0,0.2])
    ax.plot(t,s_mean[5,:],color=[0,0,0])
    ax.fill_between(t, s_mean[5,:]-s_sd[5,:], s_mean[5,:]+s_sd[5,:], color=[0,0,0,0.2])
    add_dualDMS_subplot_details(ax, chance_level)
    ax.set_xlim([2000,3500])

    j = 1
    n_mean_pre = np.mean(np.mean(neuronal_decoding[:,:,:,delay_epoch_pre_cue[j]],axis=3),axis=2)
    n_mean_post = np.mean(np.mean(neuronal_decoding[:,:,:,delay_epoch_post_cue[j]],axis=3),axis=2)

    #p_val = scipy.stats.ttest_rel(n_mean_pre,n_mean_post)[1]
    p_val = [scipy.stats.ttest_rel(n_mean_post[:,0],n_mean_post[:,1])[1], scipy.stats.ttest_rel(n_mean_post[:,2],n_mean_post[:,3])[1], scipy.stats.ttest_rel(n_mean_post[:,4],n_mean_post[:,5])[1]]
    t_stats = [scipy.stats.ttest_rel(n_mean_post[:,0],n_mean_post[:,1])[0], scipy.stats.ttest_rel(n_mean_post[:,2],n_mean_post[:,3])[0], scipy.stats.ttest_rel(n_mean_post[:,4],n_mean_post[:,5])[0]]
    post = np.mean(n_mean_post, axis=0)
    pre = np.mean(n_mean_pre, axis=0)
    post_sd = np.std(n_mean_post, axis=0)
    pre_sd = np.std(n_mean_pre, axis=0)
    print('post = ', post, 'SD ',post_sd)
    print('pre = ', pre, 'SD ', pre_sd)
    print('p_val = ', p_val)
    print('t_stats = ', t_stats)
    ax = f.add_subplot(2, 2, 3)
    ax.plot([0,1],[0,1],'k--')
    ax.plot([chance_level,chance_level],[0,1],'k--')
    ax.plot([0,1],[chance_level,chance_level],'k--')
    ax.plot(n_mean_pre[:,0+2*j], n_mean_post[:,0+2*j],'b.')
    ax.plot(n_mean_pre[:,1+2*j], n_mean_post[:,1+2*j],'r.')
    #ax.plot(n_mean_pre[ind_uattended,0+2*j], n_mean_post[ind_uattended,0+2*j],'c.')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.set_aspect('equal')
    #ax.set_yticks([0,0.5,0.6,0.7,0.8,0.9,1])
    #ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])


    n_mean = np.mean(np.mean(neuronal_rule_decoding[:,:,:,:],axis=2),axis=0)
    n_sd = np.std(np.mean(neuronal_rule_decoding[:,:,:,:],axis=2),axis=0)/np.sqrt(20)
    s_mean = np.mean(np.mean(synaptic_rule_decoding[:,:,:,:],axis=2),axis=0)
    s_sd = np.std(np.mean(synaptic_rule_decoding[:,:,:,:],axis=2),axis=0)/np.sqrt(20)

    ax = f.add_subplot(2, 2, 4)
    ax.plot(t,n_mean[0,:],linestyle = '--', color=[0,1,0], linewidth = 1)
    ax.fill_between(t, n_mean[0,:]-n_sd[0,:], n_mean[0,:]+n_sd[0,:], color=[0,1,0,0.2])
    ax.plot(t,s_mean[0,:],linestyle = '--', color=[1,0,1], linewidth = 1)
    ax.fill_between(t, s_mean[0,:]-s_sd[0,:], s_mean[0,:]+s_sd[0,:], color=[1,0,1,0.2])


    #ax = f.add_subplot(3, 2, 6)
    ax.plot(t,n_mean[1,:],color=[0,1,0])
    ax.fill_between(t, n_mean[1,:]-n_sd[1,:], n_mean[1,:]+n_sd[1,:], color=[0,1,0,0.2])
    ax.plot(t,s_mean[1,:],color=[1,0,1])
    ax.fill_between(t, s_mean[1,:]-s_sd[1,:], s_mean[1,:]+s_sd[1,:], color=[1,0,1,0.2])
    add_dualDMS_subplot_details(ax, 0.5)
    ax.set_xlim([-500,3500])
    ax.set_ylim([0.45,1.02])


    plt.tight_layout()
    plt.savefig('Fig6.pdf', format='pdf')
    plt.show()




def plot_F6(fig_params):

    task = 'dualDMS'
    t = range(-750,3500,fig_params['dt'])
    p_val_th = 0.025
    chance_level = 1/8

    example_ind = 99

    delay_accuracy = np.zeros((4, 2, fig_params['models_per_task']), dtype=np.float32)
    delay_sig_attended = np.zeros((2, 2, fig_params['models_per_task']), dtype=np.float32)
    delay_sig_unattended = np.zeros((2, 2, fig_params['models_per_task']), dtype=np.float32)
    neuronal_decoding = np.zeros((fig_params['models_per_task'], 4, fig_params['N'], len(t)), dtype=np.float32)
    synaptic_decoding = np.zeros((fig_params['models_per_task'], 4, fig_params['N'], len(t)), dtype=np.float32)
    neuronal_rule_decoding = np.zeros((fig_params['models_per_task'], 2, fig_params['N'], len(t)), dtype=np.float32)
    synaptic_rule_decoding = np.zeros((fig_params['models_per_task'], 2, fig_params['N'], len(t)), dtype=np.float32)
    accuracy = np.zeros((fig_params['models_per_task']), dtype=np.float32)
    accuracy_neural_shuffled = np.zeros((fig_params['models_per_task']), dtype=np.float32)
    accuracy_syn_shuffled = np.zeros((fig_params['models_per_task']), dtype=np.float32)

    delay_epoch = []
    delay_epoch.append(range(25+50+50+90, 25+50+50+100)) # last 100 ms of first delay period , dt=10ms
    delay_epoch.append(range(25+50+50+100+50+90, 25+50+50+100+50+100)) # last 100 ms of second delay period , dt=10ms

    #delay_epoch = []
    #delay_epoch.append(range(25+50+50+40, 25+50+50+50)) # last 100 ms of first delay period , dt=10ms
    #delay_epoch.append(range(25+50+50+100+50+40, 25+50+50+100+50+50))

    good_model_count = 0
    count = 0
    f = plt.figure(figsize=(5,6.5))

    while good_model_count < fig_params['models_per_task'] and count < 21:
        count += 1
        try:
            x = pickle.load(open(fig_params['data_dir'] + task + '_' + str(count) + '.pkl', 'rb'))
            print(fig_params['data_dir'] + task + '_' + str(count) + '.pkl', np.mean(x['accuracy']))
        except:
            print('not found: ',  fig_params['data_dir'] + task + '_' + str(count) + '.pkl')
            continue

        if np.mean(x['accuracy']) >  0.7:

            if good_model_count == example_ind:
                for j in range(4):
                    ax = f.add_subplot(4, 2, j+1)
                    ax.plot(t, np.mean(x['neuronal_sample_decoding'][j,0,:,:],axis=0),color=[0,0,1])
                    ax.plot(t, np.mean(x['neuronal_sample_decoding'][j,1,:,:],axis=0),color=[1,165/255,0])
                    ax.plot(t, np.mean(x['synaptic_sample_decoding'][j,0,:,:],axis=0),color=[0,1,1])
                    ax.plot(t, np.mean(x['synaptic_sample_decoding'][j,1,:,:],axis=0),color=[1,0,0])
                    add_dualDMS_subplot_details(ax, chance_level)
                plt.show()

            # rule decoding
            neuronal_rule_decoding[good_model_count,:,:,:] = x['neuronal_rule_decoding']
            synaptic_rule_decoding[good_model_count,:,:,:] = x['synaptic_rule_decoding']

            # cued and cued
            neuronal_decoding[good_model_count,0,:,:] = (x['neuronal_sample_decoding'][0,0,:,:] + \
                x['neuronal_sample_decoding'][3,1,:,:])/2
            synaptic_decoding[good_model_count,0,:,:] = (x['synaptic_sample_decoding'][0,0,:,:] + \
                x['synaptic_sample_decoding'][3,1,:,:])/2

            # cued and not-cued
            neuronal_decoding[good_model_count,1,:,:] = (x['neuronal_sample_decoding'][2,0,:,:] + \
                x['neuronal_sample_decoding'][1,1,:,:])/2
            synaptic_decoding[good_model_count,1,:,:] = (x['synaptic_sample_decoding'][2,0,:,:] + \
                x['synaptic_sample_decoding'][1,1,:,:])/2

            # not-cued and cued
            neuronal_decoding[good_model_count,2,:] = (x['neuronal_sample_decoding'][1,0,:,:] + \
                x['neuronal_sample_decoding'][2,1,:,:])/2
            synaptic_decoding[good_model_count,2,:] = (x['synaptic_sample_decoding'][1,0,:,:] + \
                x['synaptic_sample_decoding'][2,1,:,:])/2

            # not-cued and not-cued
            neuronal_decoding[good_model_count,3,:,:] = (x['neuronal_sample_decoding'][3,0,:,:] + \
                x['neuronal_sample_decoding'][0,1,:,:])/2
            synaptic_decoding[good_model_count,3,:,:] = (x['synaptic_sample_decoding'][3,0,:,:] + \
                x['synaptic_sample_decoding'][0,1,:,:])/2






            # attended, delay epoch 1
            s1_attn_d1 = np.mean(x['neuronal_sample_decoding'][0,0,:,delay_epoch[0]]/2 + \
                x['neuronal_sample_decoding'][2,0,:,delay_epoch[0]]/2, axis=0)
            s2_attn_d1 =np.mean(x['neuronal_sample_decoding'][1,1,:,delay_epoch[0]]/2 \
                + x['neuronal_sample_decoding'][3,1,:,delay_epoch[0]]/2, axis=0)
            delay_sig_attended[0,0,good_model_count] = np.mean(s1_attn_d1 > chance_level) > 1 - p_val_th
            delay_sig_attended[0,1,good_model_count] = np.mean(s2_attn_d1 > chance_level) > 1 - p_val_th

            # attended, delay epoch 2
            s1_attn_d2 = np.mean(x['neuronal_sample_decoding'][0,0,:,delay_epoch[1]]/2 + \
                x['neuronal_sample_decoding'][1,0,:,delay_epoch[1]]/2, axis=0)
            s2_attn_d2 =np.mean(x['neuronal_sample_decoding'][2,1,:,delay_epoch[1]]/2 \
                + x['neuronal_sample_decoding'][3,1,:,delay_epoch[1]]/2, axis=0)
            delay_sig_attended[1,0,good_model_count] = np.mean(s1_attn_d2 > chance_level) > 1 - p_val_th
            delay_sig_attended[1,1,good_model_count] = np.mean(s2_attn_d2 > chance_level) > 1 - p_val_th

            # unattended, delay epoch 1
            s1_uattn_d1 = np.mean(x['neuronal_sample_decoding'][1,0,:,delay_epoch[0]]/2 + \
                x['neuronal_sample_decoding'][3,0,:,delay_epoch[0]]/2, axis=0)
            s2_uattn_d1 =np.mean(x['neuronal_sample_decoding'][0,1,:,delay_epoch[0]]/2 \
                + x['neuronal_sample_decoding'][2,1,:,delay_epoch[0]]/2, axis=0)
            delay_sig_unattended[0,0,good_model_count] = np.mean(s1_uattn_d1 > chance_level) > 1 - p_val_th
            delay_sig_unattended[0,1,good_model_count] = np.mean(s2_uattn_d1 > chance_level) > 1 - p_val_th

            # unattended, delay epoch 2
            s1_uattn_d2 = np.mean(x['neuronal_sample_decoding'][2,0,:,delay_epoch[1]]/2 + \
                x['neuronal_sample_decoding'][3,0,:,delay_epoch[1]]/2, axis=0)
            s2_uattn_d2 =np.mean(x['neuronal_sample_decoding'][0,1,:,delay_epoch[1]]/2 \
                + x['neuronal_sample_decoding'][1,1,:,delay_epoch[1]]/2, axis=0)
            print('s1_uattn_d2', s1_uattn_d2.shape)
            delay_sig_unattended[1,0,good_model_count] = np.mean(s1_uattn_d2 > chance_level) > 1 - p_val_th
            delay_sig_unattended[1,1,good_model_count] = np.mean(s2_uattn_d2 > chance_level) > 1 - p_val_th

            accuracy[good_model_count] = np.mean(x['accuracy'])
            accuracy_neural_shuffled[good_model_count] = np.mean(x['accuracy_neural_shuffled'])
            accuracy_syn_shuffled[good_model_count] = np.mean(x['accuracy_syn_shuffled'])


            if good_model_count == example_ind:
                print('example session, RF 1 delay 1')
                print(np.mean(s1_attn_d1), np.mean(s1_uattn_d1), ' p = ', calc_p_val_compare(s1_attn_d1, s1_uattn_d1))
                print('abov chance ', np.mean(s1_attn_d1>chance_level), np.mean(s1_uattn_d1>chance_level))
                print('example session, RF 1, delay 2')
                print(np.mean(s1_attn_d2), np.mean(s1_uattn_d2), ' p = ', calc_p_val_compare(s1_attn_d2, s1_uattn_d2))
                print('abov chance ', np.mean(s1_attn_d2>chance_level), np.mean(s1_uattn_d2>chance_level))
                print('example session, RF 2 delay 1')
                print(np.mean(s2_attn_d1), np.mean(s2_uattn_d1), ' p = ', calc_p_val_compare(s2_attn_d1, s2_uattn_d1))
                print('abov chance ', np.mean(s2_attn_d1>chance_level), np.mean(s2_uattn_d1>chance_level))
                print('example session, RF 2, delay 2')
                print(np.mean(s2_attn_d2), np.mean(s2_uattn_d2), ' p = ', calc_p_val_compare(s2_attn_d2, s2_uattn_d2))
                print('abov chance ', np.mean(s2_attn_d2>chance_level), np.mean(s2_uattn_d2>chance_level))

                print(np.mean(np.mean(neuronal_decoding[example_ind,:,:,delay_epoch[1]], axis=2),axis=1))

            good_model_count += 1

    print('sh',neuronal_decoding.shape)
    f = plt.figure(figsize=(8,8))
    ax = f.add_subplot(2, 2, 1)
    for j in range(20):
        s0 = neuronal_decoding[j,0,:,:]/2+neuronal_decoding[j,1,:,:]/2
        s1 = neuronal_decoding[j,2,:,:]/2+neuronal_decoding[j,3,:,:]/2
        ax.plot(t,np.mean(s0,axis=0),'b')
        ax.plot(t,np.mean(s1,axis=0),'r')
    add_dualDMS_subplot_details(ax, chance_level)
    ax.set_xlim([-500,2000])
    ax = f.add_subplot(2, 2, 2)
    for j in range(20):
        s0 = neuronal_decoding[j,0,:,:]/2+neuronal_decoding[j,1,:,:]/2
        s1 = neuronal_decoding[j,2,:,:]/2+neuronal_decoding[j,3,:,:]/2
        ax.plot([1,2],[np.mean(s1[:,delay_epoch[0]]),np.mean(s0[:,delay_epoch[0]])],'k')

    ax = f.add_subplot(2, 2, 3)
    for j in range(20):
        ax.plot(t,np.mean(neuronal_decoding[j,2,:,:],axis=0),'b')
        ax.plot(t,np.mean(neuronal_decoding[j,3,:,:],axis=0),'r')
    add_dualDMS_subplot_details(ax, chance_level)
    ax.set_xlim([2000,3500])
    ax = f.add_subplot(2, 2, 4)
    for j in range(20):
        ax.plot([1,2],[np.mean(neuronal_decoding[j,3,:,delay_epoch[1]]),np.mean(neuronal_decoding[j,2,:,delay_epoch[1]])],'k')
    plt.show()


    # first epoch
    #d1 = np.mean(delay_accuracy[:2,0,:good_model_count] - delay_accuracy[2:,0,:good_model_count],axis=0)
    # second epoch
    #d2 = delay_accuracy[2,1,:good_model_count] - delay_accuracy[3,1,:good_model_count]
    u1 = np.mean(delay_accuracy[:2,0,:good_model_count],axis=0)
    u2 = np.mean(delay_accuracy[2:,0,:good_model_count],axis=0)
    print('Delay 1')
    print('Attended ', np.mean(u1))
    print('Unattended ', np.mean(u2))
    p_val = scipy.stats.ttest_rel(u1,u2)[1]
    print('P-Val ', p_val)

    u1 = delay_accuracy[2,1,:good_model_count]
    u2 = delay_accuracy[3,1,:good_model_count]
    print('Delay 2')
    print('Attended ', np.mean(u1))
    print('Unattended ', np.mean(u2))
    p_val = scipy.stats.ttest_rel(u1,u2)[1]
    print('P-Val ',p_val)


    delay_sig_unattended = np.reshape(delay_sig_unattended[:,:,:good_model_count],[2, 2*good_model_count])
    delay_sig_attended = np.reshape(delay_sig_attended[:,:,:good_model_count],[2, 2*good_model_count])
    print('Sig attended', np.mean(delay_sig_attended,axis=1))
    print('Sig unattended', np.mean(delay_sig_unattended,axis=1))

    1/0

    # example network
    ax = f.add_subplot(3, 2, 1)
    plot_mean_with_err_bars(ax, t, np.mean(neuronal_decoding[example_ind,:2,:,:],axis=0), good_model_count, col = [0,0,1])
    plot_mean_with_err_bars(ax, t, np.mean(neuronal_decoding[example_ind,2:,:,:],axis=0), good_model_count, col = [1,165/255,0])
    plot_mean_with_err_bars(ax, t, np.mean(synaptic_decoding[example_ind,:2,:,:],axis=0), good_model_count, col = [0,1,1])
    plot_mean_with_err_bars(ax, t, np.mean(synaptic_decoding[example_ind,2:,:,:],axis=0), good_model_count, col = [1,0,0])
    add_dualDMS_subplot_details(ax, chance_level)
    ax.set_xlim([-500,2000])
    ax = f.add_subplot(3, 2, 2)
    plot_mean_with_err_bars(ax, t, neuronal_decoding[example_ind,2,:,:], good_model_count, col = [0,0,1])
    plot_mean_with_err_bars(ax, t, neuronal_decoding[example_ind,3,:,:], good_model_count, col = [1,165/255,0])
    plot_mean_with_err_bars(ax, t, synaptic_decoding[example_ind,2,:,:], good_model_count, col = [0,1,1])
    plot_mean_with_err_bars(ax, t, synaptic_decoding[example_ind,3,:,:], good_model_count, col = [1,0,0])
    add_dualDMS_subplot_details(ax, chance_level)


    #f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(3, 2, 3)
    plot_mean_with_err_bars(ax, t, np.mean(neuronal_decoding[:good_model_count,:2,:,:],axis=1), good_model_count, col = [0,0,1])
    plot_mean_with_err_bars(ax, t, np.mean(neuronal_decoding[:good_model_count,2:,:,:],axis=1), good_model_count, col = [1,165/255,0])
    plot_mean_with_err_bars(ax, t, np.mean(synaptic_decoding[:good_model_count,:2,:,:],axis=1), good_model_count, col = [0,1,1])
    plot_mean_with_err_bars(ax, t, np.mean(synaptic_decoding[:good_model_count,2:,:,:],axis=1), good_model_count, col = [1,0,0])
    add_dualDMS_subplot_details(ax, chance_level)
    ax.set_xlim([-500,2000])
    ax = f.add_subplot(3, 2, 4)
    plot_mean_with_err_bars(ax, t, neuronal_decoding[:good_model_count,2,:,:], good_model_count, col = [0,0,1])
    plot_mean_with_err_bars(ax, t, neuronal_decoding[:good_model_count,3,:,:], good_model_count, col = [1,165/255,0])
    plot_mean_with_err_bars(ax, t, synaptic_decoding[:good_model_count,2,:,:], good_model_count, col = [0,1,1])
    plot_mean_with_err_bars(ax, t, synaptic_decoding[:good_model_count,3,:,:], good_model_count, col = [1,0,0])
    add_dualDMS_subplot_details(ax, chance_level)
    ax.set_xlim([2000,3500])
    ax = f.add_subplot(3, 2, 5)
    plot_mean_with_err_bars(ax, t, neuronal_rule_decoding[:good_model_count,0,:,:], good_model_count, col = [0,0,1])
    plot_mean_with_err_bars(ax, t, neuronal_rule_decoding[:good_model_count,1,:,:], good_model_count, col = [1,165/255,0])
    plot_mean_with_err_bars(ax, t, synaptic_rule_decoding[:good_model_count,0,:,:], good_model_count, col = [0,1,1])
    plot_mean_with_err_bars(ax, t, synaptic_rule_decoding[:good_model_count,1,:,:], good_model_count, col = [1,0,0])
    add_dualDMS_subplot_details(ax, 0.5)
    ax = f.add_subplot(3, 2, 6)
    ax.plot(np.mean(delay_accuracy[:,1,:good_model_count],axis=0), accuracy[:good_model_count],'b.')
    ax.plot(np.mean(delay_accuracy[:,1,:good_model_count],axis=0), accuracy_neural_shuffled[:good_model_count],'r.')
    ax.plot(np.mean(delay_accuracy[:,1,:good_model_count],axis=0), accuracy_syn_shuffled[:good_model_count],'c.')
    ax.plot([chance_level,chance_level],[0,1],'k--')
    ax.set_aspect(1.02/0.62)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks([0,0.5,0.6,0.7,0.8,0.9,1])
    ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_ylim([0.4,1.02])
    ax.set_xlim([0,1.02])
    ax.set_ylabel('Task accuracy')
    ax.set_xlabel('Delay neuronal decoding')


    plt.tight_layout()
    plt.savefig('Fig6.pdf', format='pdf')
    plt.show()


def calc_p_val_compare(x,y):

    x = np.reshape(x,[-1,1])
    y = np.reshape(y,[1,-1])
    x = np.tile(x,[1,x.shape[0]])
    y = np.tile(y,[y.shape[1],1])

    return np.mean(x > y)

def plot_mean_with_err_bars(ax, t, x, N, col):

    print('SHAPE ',x.shape)
    if x.ndim > 2:
        u = np.mean(np.mean(x,axis=1),axis=0)
        sd = np.std(np.mean(x,axis=1),axis=0)/np.sqrt(N)
        ax.plot(t,u,color=col)
        ax.fill_between(t, u-sd, u+sd, color=col+[0.5])
    else:
        u = np.mean(x,axis=0)
        ax.plot(t,u,color=col)


def add_dualDMS_subplot_details(ax, chance_level):

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_xticks([0,500,1500,3000])
    ax.set_ylim([0,1.02])
    ax.set_xlim([-500,3500])
    ax.plot([-900,4000],[chance_level,chance_level],'k--')
    ax.plot([0,0],[0,1],'k--')
    ax.plot([500,500],[0,1],'k--')
    ax.plot([1000,1000],[0,1],'k--')
    ax.plot([1500,1500],[0,1],'k--')
    ax.plot([2000,2000],[0,1],'k--')
    ax.plot([2500,2500],[0,1],'k--')
    ax.plot([3000,3000],[0,1],'k--')
    ax.set_ylabel('Decoding accuracy')
    ax.set_xlabel('Time relative to sample onset (ms)')

def plot_F5(fig_params):

    tasks = ['ABCA_v2','ABBA_v2']
    #tasks_test = ['ABCA_test_decode','ABBA_test_decode']
    num_tasks = len(tasks)

    #t = range(-900,4000,fig_params['dt'])
    t = range(-750,3300,fig_params['dt'])
    #delay_epoch = range(4300//fig_params['dt'],4400//fig_params['dt'])
    #delay_epoch = range(3050//fig_params['dt'],3150//fig_params['dt'])

    test_epoch = [range((750+800+800*i)//fig_params['dt'],(750+1200+800*i)//fig_params['dt']) for i in range(3)]
    delay_epoch = [range((750+700+800*i)//fig_params['dt'],(750+800+800*i)//fig_params['dt']) for i in range(3)]

    f = plt.figure(figsize=(6,8))
    chance_level = 1/8

    # for each task, we will measure model significance with respect to:
    # dim 1 = 0, neuronal decoding during delay_accuracy
    # dim 1 = 1, shuffled neuronal accuracy > chance
    # dim 1 = 2, shuffled neuronal accuracy < accuracy
    # dim 1 = 3, shuffled synaptic accuracy > chance
    # dim 1 = 4, shuffled synaptic accuracy < accuracy
    model_signficance = np.zeros((num_tasks, 5))

    #sig_neuronal_delay = np.zeros((num_tasks, fig_params['models_per_task']))
    sig_decrease_neuronal_shuffling = np.zeros((num_tasks, fig_params['models_per_task']))
    sig_decrease_syn_shuffling = np.zeros((num_tasks, fig_params['models_per_task']))
    sig_syn_shuffling = np.zeros((num_tasks, fig_params['models_per_task']))

    sig_neuronal_test = np.zeros((num_tasks, fig_params['models_per_task'], 3))
    sig_neuronal_delay = np.zeros((num_tasks, fig_params['models_per_task'], 3))
    neuronal_test = np.zeros((num_tasks, fig_params['models_per_task'], 3))
    neuronal_delay = np.zeros((num_tasks, fig_params['models_per_task'], 3))

    # correlation between neuronal decoding during the delay, accuracy after sig_syn_shuffling
    # neuronal activity, and accuracy after shuffling synaptic activity
    corr_decoding_neuronal_shuf =  np.zeros((num_tasks,2))
    corr_decoding_syn_shuf =  np.zeros((num_tasks,2))
    corr_neuronal_shuf_syn_shuf =  np.zeros((num_tasks,2))

    decoding_p_val =  np.zeros((num_tasks))
    p_val_th = 0.025
    chance_level = 1/8


    for n in range(num_tasks):

        # load following results from each task
        delay_accuracy = np.zeros((fig_params['models_per_task']), dtype=np.float32)
        neuronal_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        synaptic_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        neuronal_test_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        synaptic_test_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        synaptic_test_decoding_shuffled = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        accuracy = np.zeros((fig_params['models_per_task'], fig_params['N']), dtype=np.float32)
        accuracy_neural_shuffled = np.zeros((fig_params['models_per_task'], fig_params['N']), dtype=np.float32)
        accuracy_syn_shuffled = np.zeros((fig_params['models_per_task'], fig_params['N']), dtype=np.float32)

        tuning_sim = np.zeros((fig_params['models_per_task'], len(t)), dtype=np.float32)
        tuning_sim_shuffled = np.zeros((4, fig_params['models_per_task'], len(t)), dtype=np.float32)
        #neuron_ind = [range(100),range(80),range(80,100),range(80,100,2),range(81,100,2)]
        #neuron_ind = [range(100),range(0,80,2),range(1,80,2),range(80,100,2),range(81,100,2)] # all neurons, EXC neurons, INH neurons
        #neuron_ind = [range(100)]*5

        accuracy_test2_shuffled = np.zeros((fig_params['models_per_task'], 4))
        accuracy_test2 = np.zeros((fig_params['models_per_task']))

        good_model_count = 0
        count = -1
        while good_model_count < fig_params['models_per_task'] and count < 20:
            count += 1
            try:
                x = pickle.load(open(fig_params['data_dir'] + tasks[n] + '_' + str(count) + '.pkl', 'rb'))
                #x_neural = pickle.load(open(fig_params['data_dir'] + 'neural_behavior_' + tasks[n] + '_' + str(count) + '.pkl', 'rb'))
            except:
                print('not found: ',  tasks[n] + '_' + str(count) + '.pkl')
                continue

            print(tasks[n] + '_' + str(count) + '.pkl', np.mean(x['accuracy']))

            if np.mean(x['accuracy']) >  0.9:

                delay_accuracy[good_model_count] = np.mean(x['neuronal_sample_decoding'][0,0,:,delay_epoch])
                neuronal_decoding[good_model_count,:,:] = x['neuronal_sample_decoding'][0,0,:,:]
                synaptic_decoding[good_model_count,:,:] = x['synaptic_sample_decoding'][0,0,:,:]

                neuronal_test_decoding[good_model_count,:,:] = x['neuronal_test_decoding'][0,0,:,:]
                synaptic_test_decoding[good_model_count,:,:] = x['synaptic_test_decoding'][0,0,:,:]

                accuracy[good_model_count,:] = x['accuracy'][0,0,:]
                accuracy_neural_shuffled[good_model_count,:] = x['accuracy_neural_shuffled'][0,1,:]
                accuracy_syn_shuffled[good_model_count,:] = x['accuracy_syn_shuffled'][0,1,:]

                syn_sample_tuning = x['synaptic_pev'][:,0,:]*np.exp(1j*x['synaptic_pref_dir'][:,0,:])
                syn_test_tuning = x['synaptic_pev_test'][:,0,:]*np.exp(1j*x['synaptic_pref_dir_test'][:,0,:])

                #print(x['accuracy_syn_shuffled'][0,0,:])
                #quit()

                if n == 1:
                    syn_test_tuning_shuffled = np.zeros((4, par['n_hidden'], len(t)), dtype=np.complex64)
                    accuracy_test2[good_model_count]= x['accuracy_no_suppression'][0]
                    #accuracy_test2[good_model_count]=  x['accuracy_neural_shuffled_grp']

                    for j in range(4):
                        syn_test_tuning_shuffled[j,:,:] = x['synaptic_pev_test_shuffled'][0,0,j,:,:]*\
                            np.exp(1j*x['synaptic_pref_dir_test_shuffled'][0,0,j,:,:])
                        accuracy_test2_shuffled[good_model_count, j] = x['accuracy_suppression'][0,0,j,0]
                        print('Acc suppression ',j, x['accuracy_suppression'][0,0,j,0])

                        """
                        syn_test_tuning_shuffled[j,:,:] = np.mean(x['synaptic_pev_test_shuffled'][0,1,j,:,:,:]*\
                            np.exp(1j*x['synaptic_pref_dir_test_shuffled'][0,1,j,:,:,:]),axis=0)
                        accuracy_test2_shuffled[good_model_count, j] = np.mean(x['accuracy_syn_shuffled_grp'][0,1,j,:])
                        print('acc ', np.mean(x['accuracy_syn_shuffled_grp'][0,1,j,:]))
                        """

                for t1 in range(75+81,len(t)):
                    tuning_sim[good_model_count,t1] = np.real(np.sum(syn_sample_tuning[:,t1]*\
                        np.conj(syn_test_tuning[:,t1])))/(0.001+np.sum(np.abs(syn_sample_tuning[:,t1])\
                        *np.abs(syn_test_tuning[:,t1])))

                    if n == 1:
                        for j in range(4):
                            tuning_sim_shuffled[j,good_model_count,t1] = np.real(np.sum(syn_test_tuning_shuffled[j,:,t1]*\
                                np.conj(syn_sample_tuning[:,t1])))/(0.001+np.sum(np.abs(syn_sample_tuning[:,t1])\
                                *np.abs(syn_test_tuning_shuffled[j,:,t1])))

                good_model_count +=1

        if good_model_count < fig_params['models_per_task']:
            print('Too few accurately trained models, good models = ', good_model_count)

        model_signficance[n, 0] = np.sum(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2) \
            >chance_level,axis=1)>1-p_val_th)

        model_signficance[n, 1] = np.sum(np.mean(accuracy_neural_shuffled>0.5,axis=1)>1-p_val_th)
        model_signficance[n, 2] = np.sum(np.mean(accuracy-accuracy_neural_shuffled>0,axis=1)>1-p_val_th)
        model_signficance[n, 3] = np.sum(np.mean(accuracy_syn_shuffled>0.5,axis=1)>1-p_val_th)
        model_signficance[n, 4] =  np.sum(np.mean(accuracy-accuracy_syn_shuffled>0,axis=1)>1-p_val_th)
        #sig_neuronal_delay[n,:] =np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2)>chance_level,axis=1)>1-p_val_th

        for j in range(3):
            sig_neuronal_test[n,:,j] =np.mean(np.mean(neuronal_decoding[:,:,test_epoch[j]],axis=2)>chance_level,axis=1)>1-p_val_th
            sig_neuronal_delay[n,:,j] =np.mean(np.mean(neuronal_decoding[:,:,delay_epoch[j]],axis=2)>chance_level,axis=1)>1-p_val_th
            neuronal_test[n,:,j] =np.mean(np.mean(neuronal_decoding[:,:,test_epoch[j]],axis=2),axis=1)
            neuronal_delay[n,:,j] =np.mean(np.mean(neuronal_decoding[:,:,delay_epoch[j]],axis=2),axis=1)


        sig_decrease_neuronal_shuffling[n,:] =  np.mean(accuracy-accuracy_neural_shuffled>0,axis=1)>1-p_val_th
        sig_decrease_syn_shuffling[n,:] =  np.mean(accuracy-accuracy_syn_shuffled>0,axis=1)>1-p_val_th
        sig_syn_shuffling[n,:] = np.mean(accuracy_syn_shuffled>0.5,axis=1)>1-p_val_th


        #ind = np.where(np.mean(accuracy_neural_shuffled,axis=1)<0.9)[0]
        #print(tasks[n], ind)
        #print(tasks[n], np.mean(np.mean(accuracy_neural_shuffled[ind,:],axis=1)))
        #print(tasks[n], np.mean(np.mean(accuracy_syn_shuffled[ind,:],axis=1)))


        corr_decoding_neuronal_shuf[n,:] = scipy.stats.pearsonr(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch[2]],axis=2),axis=1), \
            np.mean(accuracy_neural_shuffled,axis=1))
        corr_decoding_syn_shuf[n,:] = scipy.stats.pearsonr(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch[2]],axis=2),axis=1), \
            np.mean(accuracy_syn_shuffled,axis=1))
        corr_neuronal_shuf_syn_shuf[n,:] = scipy.stats.pearsonr(np.mean(accuracy_neural_shuffled,axis=1), \
            np.mean(accuracy_syn_shuffled,axis=1))


        ax = f.add_subplot(4,2,n+1)
        add_ABBA_subplot_details(ax, 'decode')
        for j in range(fig_params['models_per_task']):
            ax.plot(t,np.mean(neuronal_decoding[j,:,:],axis=0),'g')
            ax.plot(t,np.mean(synaptic_decoding[j,:,:],axis=0),'m')


        ax = f.add_subplot(4,2,n+3)
        add_ABBA_subplot_details(ax, 'decode')
        for j in range(fig_params['models_per_task']):
            ax.plot(t,np.mean(neuronal_test_decoding[j,:,:],axis=0),'g')
            ax.plot(t,np.mean(synaptic_test_decoding[j,:,:],axis=0),'m')


        #ax = f.add_subplot(4,2,n+5)
        #ax.plot(delay_accuracy, np.mean(accuracy,axis=1),'b.')
        #ax.plot(delay_accuracy, np.mean(accuracy_neural_shuffled,axis=1),'r.')
        #ax.plot(delay_accuracy, np.mean(accuracy_syn_shuffled,axis=1),'c.')
        #print('Scatter')
        #print(delay_accuracy,np.mean(accuracy,axis=1))
        #print(delay_accuracy,np.mean(accuracy_neural_shuffled,axis=1))
        #print(delay_accuracy,np.mean(accuracy_syn_shuffled,axis=1))
        #add_ABBA_subplot_details(ax, 'shuffle')

        ax = f.add_subplot(4,2,n+5)
        u = np.mean(tuning_sim[:good_model_count,:],axis=0)
        add_ABBA_subplot_details(ax, 'tuning')
        sd = np.std(tuning_sim[:good_model_count,:],axis=0)/np.sqrt(good_model_count)
        ax.plot(t, u,'k')
        ax.fill_between(t, u-sd, u+sd, color=[0,0,0,0.5])



        if n == 0:
            test2_time = range(25+50+40*4, 25+50+40*5)
            tuning_sim_ABCA = np.mean(tuning_sim[:good_model_count, test2_time], axis=1)
        elif n == 1:
            tuning_sim_ABBA_base = np.mean(tuning_sim[:good_model_count, test2_time], axis=1)
            tuning_sim_ABBA_supp = np.mean(tuning_sim_shuffled[2,:good_model_count, test2_time], axis=0)
            print('ABBA supp ', tuning_sim_ABBA_supp)
            print('ABBA base ', tuning_sim_ABBA_base)


        if n == 1:
            ax = f.add_subplot(4,2,7)
            u = np.mean(tuning_sim[:good_model_count,:],axis=0)
            sd = np.std(tuning_sim[:good_model_count,:],axis=0)/np.sqrt(good_model_count)
            add_ABBA_subplot_details(ax, 'tuning')
            ax.plot(t, u,'k')
            ax.fill_between(t, u-sd, u+sd, color=[0,0,0,0.5])
            #col = [[0,0,1],[1,0,0],[0,1,0],[0,1,1]]
            col = [[0,0,1], [1,0,0],[0,1,0], [1,165/255,0]]
            for j in range(4):
                u = np.mean(tuning_sim_shuffled[j,:good_model_count,:],axis=0)
                sd = np.std(tuning_sim_shuffled[j,:good_model_count,:],axis=0)/np.sqrt(good_model_count)
                ax.plot(t, u, color=col[j])
                ax.fill_between(t, u-sd, u+sd, color=col[j]+[0.5])



            acc = np.zeros((5))
            acc_se = np.zeros((5))
            acc[0] = np.mean(accuracy_test2[:good_model_count])
            acc[1:] = np.mean(accuracy_test2_shuffled[:good_model_count, :],axis=0)
            acc_se[0] = np.std(accuracy_test2[:good_model_count])/np.sqrt(good_model_count)
            acc_se[1:] = np.std(accuracy_test2_shuffled[:good_model_count, :],axis=0)/np.sqrt(good_model_count)
            ax = f.add_subplot(4,2,8)
            ax.plot(np.ones((good_model_count)), accuracy_test2[:good_model_count],'k.')
            for i in range(2,6):
                ax.plot(i*np.ones((good_model_count)), accuracy_test2_shuffled[:good_model_count, i-2],'.', color=col[i-2])

            #p0, p1, p2, p3, p4 = ax.bar([0,1,2,3,4], acc, yerr=acc_se)
            #p0.set_facecolor('k')
            #p1.set_facecolor(col[0])
            #p2.set_facecolor(col[1])
            #p3.set_facecolor(col[2])
            #p4.set_facecolor(col[3])
            ax.set_ylabel('Behavioral accuracy')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylim([0.8, 1])


    print('Mean accuracy after suppression')
    print(acc)
    print('accuracy after suppression p-vals compared to basline')
    p = np.zeros((4))
    for i in range(4):
        _, p[i] = scipy.stats.ttest_rel(accuracy_test2[:good_model_count], accuracy_test2_shuffled[:good_model_count, i])
    print('p-vals ', p)
    print('accuracy after suppression lowest compared to otehr three')
    p0 = scipy.stats.ttest_rel(accuracy_test2_shuffled[:good_model_count,0], accuracy_test2_shuffled[:good_model_count, 2])
    p1 = scipy.stats.ttest_rel(accuracy_test2_shuffled[:good_model_count,1], accuracy_test2_shuffled[:good_model_count, 2])
    p2 = scipy.stats.ttest_rel(accuracy_test2_shuffled[:good_model_count,3], accuracy_test2_shuffled[:good_model_count, 2])
    print('p-vals ', p0, p1, p2)

    plt.tight_layout()
    plt.savefig('Fig5.pdf', format='pdf')
    plt.show()

    p_val = scipy.stats.ttest_ind(tuning_sim_ABCA,tuning_sim_ABBA_base)
    print('Mean ABCA sim ', np.mean(tuning_sim_ABCA), '+/-', np.std(tuning_sim_ABCA), ' Mean ABBA sim ', np.mean(tuning_sim_ABBA_base) , \
        '+/-',  np.std(tuning_sim_ABBA_base) , 'p = ', p_val)

    p_val = scipy.stats.ttest_rel(tuning_sim_ABBA_supp ,tuning_sim_ABBA_base)
    print('Mean ABBA sim supp', np.mean(tuning_sim_ABBA_supp), '+/-', np.std(tuning_sim_ABBA_supp), ' Mean ABBA sim ', \
        np.mean(tuning_sim_ABBA_base) , '+/-', np.std(tuning_sim_ABBA_base),' p = ', p_val)


    print('Number of models with delay neuronal decoding accuracy signficantly greater than chance')
    print(model_signficance)

    #print('Number of models for which neuronal decoding is at chance')
    #print(np.sum(1-sig_neuronal_delay,axis=1))
    print('Number of models for which shuffling neuronal activity has no effect')
    print(np.sum(1-sig_decrease_neuronal_shuffling,axis=1))
    print('Number of models for which shuffling STP causes accuracy to fall to chance')
    print(np.sum(1-sig_syn_shuffling,axis=1))
    #print('Number of models for which 3 above conditions are satisfied')
    #print(np.sum((1-sig_neuronal_delay)*(1-sig_decrease_neuronal_shuffling)*(1-sig_syn_shuffling),axis=1))
    print('Number of models for which shuffling neuronal and synaptic activity decreases accuracy')
    print(np.sum((sig_decrease_neuronal_shuffling)*(sig_decrease_syn_shuffling),axis=1))

    print('Correlations...')
    print(corr_decoding_neuronal_shuf)
    print(corr_decoding_syn_shuf)
    print(corr_neuronal_shuf_syn_shuf)

    print('Number of models for which neuronal decoding for test 1 is above chance')
    print(np.sum(sig_neuronal_test[:,:,0],axis=1))
    print('Number of models for which neuronal decoding for test 2 is above chance')
    print(np.sum(sig_neuronal_test[:,:,1],axis=1))
    print('Number of models for which neuronal decoding for test 3 is above chance')
    print(np.sum(sig_neuronal_test[:,:,2],axis=1))

    print('Number of models for which neuronal decoding for delay 1 is above chance')
    print(np.sum(sig_neuronal_delay[:,:,0],axis=1))
    print('Number of models for which neuronal decoding for delay 2 is above chance')
    print(np.sum(sig_neuronal_delay[:,:,1],axis=1))
    print('Number of models for which neuronal decoding for delay 3 is above chance')
    print(np.sum(sig_neuronal_delay[:,:,2],axis=1))

    test1_p_val = scipy.stats.ttest_ind(neuronal_test[0,:,0],neuronal_test[1,:,0])[1]
    test2_p_val = scipy.stats.ttest_ind(neuronal_test[0,:,1],neuronal_test[1,:,1])[1]
    print('Test 1')
    print(np.mean(neuronal_test[0,:,0]), np.mean(neuronal_test[1,:,0]), test1_p_val)
    print('Test 2')
    print(np.mean(neuronal_test[0,:,1]), np.mean(neuronal_test[1,:,1]), test2_p_val)

def add_ABBA_subplot_details(ax, plot_type):

    chance_level = 1/8
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if plot_type == 'decode':

        ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_xticks([0,400,800,1200,1600,2000,2400])
        ax.set_ylim([0,1.02])
        ax.set_xlim([-500,2800])
        ax.plot([-900,2800],[chance_level,chance_level],'k--')
        ax.plot([0,0],[-1,1],'k--')
        ax.plot([400,400],[-1,1],'k--')
        ax.plot([800,800],[-1,1],'k--')
        ax.plot([1200,1200],[-1,1],'k--')
        ax.plot([1600,1600],[-1,1],'k--')
        ax.plot([2000,2000],[-1,1],'k--')
        ax.plot([2400,2400],[-1,1],'k--')
        ax.set_ylabel('Decoding accuracy')
        ax.set_xlabel('Time relative to sample onset (ms)')

    elif plot_type == 'shuffle':

        ax.plot([chance_level,chance_level],[0,1],'k--')
        ax.set_aspect(1/0.62)
        ax.set_yticks([0,0.5,0.6,0.7,0.8,0.9,1])
        ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_ylim([0.4,1.02])
        ax.set_ylabel('Task accuracy')
        ax.set_xlabel('Delay neuronal decoding')

    elif plot_type == 'tuning':
        ax.set_xticks([0,400,800,1200,1600,2000,2400])
        ax.set_ylim([-0.15,1.02])
        ax.set_xlim([-500,2800])
        ax.plot([0,0],[-1,1],'k--')
        ax.plot([400,400],[-1,1],'k--')
        ax.plot([800,800],[-1,1],'k--')
        ax.plot([1200,1200],[-1,1],'k--')
        ax.plot([1600,1600],[-1,1],'k--')
        ax.plot([2000,2000],[-1,1],'k--')
        ax.plot([2400,2400],[-1,1],'k--')
        ax.plot([-1000, 3000],[0,0],'k--')
        ax.set_ylabel('Tuning similarity')
        ax.set_xlabel('Time relative to sample onset (ms)')
