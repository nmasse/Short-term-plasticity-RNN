import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats
from parameters import *
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "arial"


def plot_all_figures():

    fig_params = {
        'data_dir'              : 'C:/Users/nicol/Projects/RNN_analysis_nov1/',
        'dt'                    : 10,
        'models_per_task'       : 20,
        'N'                     : 100, # bootstrap iterations
        'accuracy_th'           : 0.9} # minimum accuracy of model required for analysis
    #plot_SF1(fig_params)
    #plot_SF2(fig_params)
    plot_F3(fig_params)
    #plot_F4(fig_params)
    #plot_F5(fig_params)
    #plot_F6(fig_params)

def plot_SF2(fig_params):

    task_name = 'DMRS45_51'
    x = pickle.load(open(fig_params['data_dir'] + task_name + '.pkl', 'rb'))
    t = range(-900,2000,fig_params['dt'])

    early_sample_time = 40+50+5 # 100 ms into sample epoch
    late_sample_time = 40+50+50 # 500 ms into sample epoch
    test_time = 40+50+50+100+5

    f = plt.figure(figsize=(7,5))
    chance_level = 1/8

    # plot neuronal decoding accuracy of the DMS and DMRs90 tasks
    ax = f.add_subplot(2, 2, 1)
    ax.plot(t,np.mean(x['neuronal_decoding'][0,0,:,:],axis=0),'g')
    ax.plot(t,np.mean(x['synaptic_decoding'][0,0,:,:],axis=0),'m')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_xticks([0,500,1500-10])
    ax.set_ylim([0,1.02])
    ax.set_xlim([-500,1500-10])
    ax.plot([-900,2000],[chance_level,chance_level],'k--')
    ax.plot([0,0],[0,1],'k--')
    ax.plot([500,500],[0,1],'k--')
    ax.set_ylabel('Decoding accuracy')
    ax.set_xlabel('Time relative to sample onset (ms)')

    # select neurons that are selective for both tasks
    ind = np.where((x['neuronal_pev'][:,0,early_sample_time]>0.1)*(x['neuronal_pev'][:,0,early_sample_time]<1))[0]

    # calculate the angular differences between the preferred directiosn measured in each tasks
    diff_early = np.angle(np.exp(1j*x['neuronal_pref_dir'][ind,0,early_sample_time] \
        -1j*x['neuronal_pref_dir'][ind,0,late_sample_time]))/np.pi*180
    #diff_late = np.angle(np.exp(1j*x['neuronal_pref_dir'][ind_late,0,late_sample_time] \
    #    -1j*x['neuronal_pref_dir'][ind_late,1,late_sample_time]))/np.pi*180

    diff_early_test = np.angle(np.exp(1j*x['neuronal_pref_dir'][ind,0,early_sample_time] \
        -1j*x['neuronal_pref_dir_test'][ind,0,test_time]))/np.pi*180
    diff_late_test = np.angle(np.exp(1j*x['neuronal_pref_dir'][ind,0,late_sample_time] \
        -1j*x['neuronal_pref_dir_test'][ind,0,test_time]))/np.pi*180

    bins = np.arange(-180,180,30)
    ax = f.add_subplot(2, 2, 2)
    ax.hist(diff_early, bins = bins)
    ax.set_xticks([-180,-90,0,90,180])
    ax.set_ylim([0,20])
    ax.set_ylabel('Count')
    ax.set_xlabel('Difference in preferred direction')

    ax = f.add_subplot(2, 2, 3)
    ax.hist(diff_early_test, bins = bins)
    ax.set_xticks([-180,-90,0,90,180])
    ax.set_ylim([0,20])
    ax.set_ylabel('Count')

    ax = f.add_subplot(2, 2, 4)
    ax.hist(diff_late_test, bins = bins)
    ax.set_xticks([-180,-90,0,90,180])
    ax.set_ylim([0,20])
    ax.set_ylabel('Count')
    ax.set_xlabel('Difference in preferred direction')

    #x = f.add_subplot(2, 3, 5)
    #ax.hist(diff_late_test, bins = bins)
    #ax.set_xticks([-180,-90,0,90,180])
    #ax.set_ylim([0,20])
    #ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('FigS2.pdf', format='pdf')
    plt.show()


def plot_SF1(fig_params):

    num_tasks = 3
    chance_level = 1/8
    model_signficance = np.zeros((num_tasks))
    f = plt.figure(figsize=(3,4.25))

    for n in range(num_tasks):

        t = range(-750,2000+n*500,fig_params['dt'])
        delay_epoch = range((2150+n*500)//fig_params['dt'],(2250+n*500)//fig_params['dt'])

        # load following results from each task
        delay_accuracy = np.zeros((fig_params['models_per_task']), dtype=np.float32)
        neuronal_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        synaptic_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)

        good_model_count = 0
        count = 0
        while good_model_count < fig_params['models_per_task'] and count < 25:
            if n == 0:
                task_name = 'DMS_' + str(count)
            elif n == 1:
                task_name = 'DMS_' + str(count) + '_delay1500'
            elif n == 2:
                task_name = 'DMS_delay2000_sc02_' + str(count)
            try:
                x = pickle.load(open(fig_params['data_dir'] + task_name + '.pkl', 'rb'))
            except:
                #print('not found: ',  task_name + '.pkl')
                count +=1
                continue
            count += 1
            if np.mean(x['accuracy']) >  0.9:
                delay_accuracy[good_model_count] = np.mean(x['neuronal_decoding'][0,0,:,delay_epoch])
                neuronal_decoding[good_model_count,:,:] = x['neuronal_decoding'][0,0,:,:]
                synaptic_decoding[good_model_count,:,:] = x['synaptic_decoding'][0,0,:,:]
                good_model_count +=1
        print('number of models ', ' ', n, ' ', good_model_count)
        if good_model_count < fig_params['models_per_task']:
            print('Too few accurately trained models')

        model_signficance[n] = np.sum(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2) \
            >chance_level,axis=1)>0.95)

        ax = f.add_subplot(num_tasks, 1, n+1)
        plt.hold(True)
        for j in range(fig_params['models_per_task']):
            ax.plot(t,np.mean(neuronal_decoding[j,:,:],axis=0),'g')
            ax.plot(t,np.mean(synaptic_decoding[j,:,:],axis=0),'m')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_xticks([0,500,1500-10+n*500])
        ax.set_ylim([0,1.02])
        ax.set_xlim([-500,1500-10+n*500])
        ax.plot([-900,4000],[chance_level,chance_level],'k--')
        ax.plot([0,0],[0,1],'k--')
        ax.plot([500,500],[0,1],'k--')
        ax.set_ylabel('Decoding accuracy')
        ax.set_xlabel('Time relative to sample onset (ms)')



    plt.tight_layout()
    plt.savefig('FigS1.pdf', format='pdf')
    plt.show()


    print(model_signficance)

def plot_F4(fig_params):

    task = 'DMS+DMRS'
    t = range(-750,2000,fig_params['dt'])
    delay_epoch = range(2150//fig_params['dt'],2250//fig_params['dt'])
    f = plt.figure(figsize=(6,3.75))
    chance_level = 1/8

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
    while good_model_count < fig_params['models_per_task'] and count < 24:
        x = pickle.load(open(fig_params['data_dir'] + task + '_' + str(count) + '.pkl', 'rb'))
        count += 1
        print(np.mean(x['accuracy'][0,:]), np.mean(x['accuracy'][1,:]))
        if np.mean(x['accuracy']) > fig_params['accuracy_th']:

            for j in range(2):
                delay_accuracy[j, good_model_count] = np.mean(x['neuronal_decoding'][j,0,:,delay_epoch])
                neuronal_decoding[j, good_model_count,:,:] = x['neuronal_decoding'][j,0,:,:]
                synaptic_decoding[j, good_model_count,:,:] = x['synaptic_decoding'][j,0,:,:]
                accuracy[j, good_model_count,:] = x['accuracy'][j,:]
                accuracy_neural_shuffled[j, good_model_count,:] = x['accuracy_neural_shuffled'][j, :]
                accuracy_syn_shuffled[j, good_model_count,:] = x['accuracy_syn_shuffled'][j, :]
            good_model_count +=1

    if good_model_count < fig_params['models_per_task']:
        print('Too few accurately trained models ', good_model_count)

    for j  in range(2):

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
        plt.hold(True)
        for n in range(fig_params['models_per_task']):
            ax.plot(t,np.mean(neuronal_decoding[j,n,:,:],axis=0),'g')
            ax.plot(t,np.mean(synaptic_decoding[j,n,:,:],axis=0),'m')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_xticks([0,500,1000, 1250, 1500-10])
        ax.set_ylim([0,1.02])
        ax.set_xlim([-500,1500-10])
        ax.plot([-900,2000],[chance_level,chance_level],'k--')
        ax.plot([0,0],[0,1],'k--')
        ax.plot([500,500],[0,1],'k--')
        ax.plot([1000,1000],[0,1],'k--')
        ax.plot([1250,1250],[0,1],'k--')
        ax.set_ylabel('Decoding accuracy')
        ax.set_xlabel('Time relative to sample onset (ms)')


        ax = f.add_subplot(2, 2, 2*j+2)
        plt.hold(True)
        ax.plot(delay_accuracy[j,:], np.mean(accuracy[j,:,:],axis=1),'b.')
        ax.plot(delay_accuracy[j,:], np.mean(accuracy_syn_shuffled[j,:,:],axis=1),'c.')
        ax.plot(delay_accuracy[j,:], np.mean(accuracy_neural_shuffled[j,:,:],axis=1),'r.')
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
    plt.savefig('Fig4.pdf', format='pdf')
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


def plot_F3(fig_params):

    tasks = [ 'DMS','DMRS180','DMRS90','DMC']
    num_tasks = len(tasks)

    #t = range(-900,2000,fig_params['dt'])
    t = range(-750,2000,fig_params['dt'])
    #delay_epoch = range(2300//fig_params['dt'],2400//fig_params['dt'])
    delay_epoch = range(2150//fig_params['dt'],2250//fig_params['dt'])

    f = plt.figure(figsize=(6,7.5))

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

    decoding_p_val =  np.zeros((num_tasks))

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
        while good_model_count < fig_params['models_per_task'] and count<21:
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
                accuracy_suppression[n, good_model_count, :]  = x['accuracy_suppression'][0,:,0]
                good_model_count +=1


        if good_model_count < fig_params['models_per_task']:
            print('Too few accurately trained models')

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

        decoding_p_val[n] = scipy.stats.ttest_ind(np.mean(np.mean(neuronal_decoding_DMS[:,:,delay_epoch],axis=2),axis=1), \
            np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2),axis=1))[1]


        ax = f.add_subplot(num_tasks, 2, 2*n+1)
        plt.hold(True)

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
        plt.hold(True)
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


    f = plt.figure(figsize=(6,7.5))
    for n in range(num_tasks):
        if n == 3:
            chance_level = 1/2
        else:
            chance_level = 1/8
        ax = f.add_subplot(num_tasks, 2, 2*n+1)
        ax.plot(delay_accuracy[n,:], accuracy_suppression[n,:,1],'k.')
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
        if n == 3:
            ax.set_ylabel('Behavioral accuracy')
            ax.set_xlabel('Delay neuronal decoding')

        ax = f.add_subplot(num_tasks, 2, 2*n+2)
        ax.plot(np.mean(accuracy_neural_shuffled[n,:,:],axis=1), accuracy_suppression[n,:,1],'k.')
        #ax.plot([chance_level,chance_level],[0,1],'k--')
        ax.set_aspect(1.02/0.62)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_yticks([0,0.5,0.6,0.7,0.8,0.9,1])
        ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_ylim([0.4,1.02])
        ax.set_xlim([0,1.02])
        if n == 3:
            ax.set_ylabel('Behavioral accuracy')
            ax.set_xlabel('Behavioral accuracy - neuronal activity shuffled')

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

    print(p_neuronal_delay.shape, accuracy_suppression.shape)
    print('XXXX 0.9')
    print(np.sum((1-p_neuronal_delay)*(1-p_decrease_neuronal_shuffling)*(1-p_synaptic_shuffling)*(accuracy_suppression[:,:,1]>0.9),axis=1))
    print('XXXX 0.925')
    print(np.sum((1-p_neuronal_delay)*(1-p_decrease_neuronal_shuffling)*(1-p_synaptic_shuffling)*(accuracy_suppression[:,:,1]>0.925),axis=1))
    print('XXXX 0.95')
    print(np.sum((1-p_neuronal_delay)*(1-p_decrease_neuronal_shuffling)*(1-p_synaptic_shuffling)*(accuracy_suppression[:,:,1]>0.95),axis=1))

    print('Correlations...')
    print(corr_decoding_neuronal_shuf)
    print(corr_decoding_syn_shuf)
    print(corr_neuronal_shuf_syn_shuf)
    print('T-test neuroanl decoding p-val compared to DMS')
    print(decoding_p_val)


def plot_F6(fig_params):

    task = 'dualDMS'
    t = range(-750,3500,fig_params['dt'])
    p_val_th = 0.025
    chance_level = 1/8

    example_ind = 0

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

    delay_epoch = []
    delay_epoch.append(range(25+50+50+40, 25+50+50+50)) # last 100 ms of first delay period , dt=10ms
    delay_epoch.append(range(25+50+50+100+50+40, 25+50+50+100+50+50))

    good_model_count = 0
    count = 0
    f = plt.figure(figsize=(5,6.5))

    while good_model_count < fig_params['models_per_task'] and count < 11:
        count += 1
        try:
            x = pickle.load(open(fig_params['data_dir'] + task + '_' + str(count+12) + '.pkl', 'rb'))
        except:
            print('not found: ',  fig_params['data_dir'] + task + '_' + str(count+12) + '.pkl')
            continue

        print(task + '_' + str(count+12) + '.pkl', np.mean(x['accuracy']))
        if np.mean(x['accuracy']) >  0.9:

            if good_model_count == 999+example_ind:
                for j in range(4):
                    ax = f.add_subplot(4, 2, j+1)
                    ax.plot(t, np.mean(x['neuronal_sample_decoding'][j,0,:,:],axis=0),color=[0,0,1])
                    ax.plot(t, np.mean(x['neuronal_sample_decoding'][j,1,:,:],axis=0),color=[1,165/255,0])
                    ax.plot(t, np.mean(x['synaptic_sample_decoding'][j,0,:,:],axis=0),color=[0,1,1])
                    ax.plot(t, np.mean(x['synaptic_sample_decoding'][j,1,:,:],axis=0),color=[1,0,0])
                    add_dualDMS_subplot_details(ax, chance_level)


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

            for j in range(2):
                for i in range(4):
                    delay_accuracy[i,j,good_model_count] = np.mean(neuronal_decoding[good_model_count,i,:,delay_epoch[j]])


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

    tasks = ['ABCA','ABBA']
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

                accuracy[good_model_count,:] = x['accuracy']
                accuracy_neural_shuffled[good_model_count,:] = x['accuracy_neural_shuffled']
                accuracy_syn_shuffled[good_model_count,:] = x['accuracy_syn_shuffled']

                syn_sample_tuning = x['synaptic_pev'][:,0,:]*np.exp(1j*x['synaptic_pref_dir'][:,0,:])
                syn_test_tuning = x['synaptic_pev_test'][:,0,:]*np.exp(1j*x['synaptic_pref_dir_test'][:,0,:])


                if n == 1:
                    syn_test_tuning_shuffled = np.zeros((4, par['n_hidden'], len(t)), dtype=np.complex64)
                    accuracy_test2[good_model_count]= x['ABBA_test2_acc'][0]

                    for j in range(4):
                        syn_test_tuning_shuffled[j,:,:] = x['synaptic_pev_test_shuffled'][3+j,5,:,0,:]*\
                            np.exp(1j*x['synaptic_pref_dir_test_shuffled'][3+j,5,:,0,:])
                        accuracy_test2_shuffled[good_model_count, j] = x['ABBA_test2_acc_shuffled'][3+j,5,0]


                for t1 in range(len(t)):
                    tuning_sim[good_model_count,t1] = np.real(np.sum(syn_sample_tuning[:,t1]*\
                        np.conj(syn_test_tuning[:,t1])))/(0.01+np.sum(np.abs(syn_sample_tuning[:,t1])\
                        *np.abs(syn_test_tuning[:,t1])))

                    if n == 1:
                        for j in range(4):
                            tuning_sim_shuffled[j,good_model_count,t1] = np.real(np.sum(syn_test_tuning_shuffled[j,:,t1]*\
                                np.conj(syn_sample_tuning[:,t1])))/(0.01+np.sum(np.abs(syn_sample_tuning[:,t1])\
                                *np.abs(syn_test_tuning_shuffled[j,:,t1])))

                good_model_count +=1

        print('COUNT ', count)
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


        ax = f.add_subplot(5,2,n+1)
        for j in range(fig_params['models_per_task']):
            ax.plot(t,np.mean(neuronal_decoding[j,:,:],axis=0),'g')
            ax.plot(t,np.mean(synaptic_decoding[j,:,:],axis=0),'m')
        add_ABBA_subplot_details(ax, 'decode')

        ax = f.add_subplot(5,2,n+3)
        for j in range(fig_params['models_per_task']):
            ax.plot(t,np.mean(neuronal_test_decoding[j,:,:],axis=0),'g')
            ax.plot(t,np.mean(synaptic_test_decoding[j,:,:],axis=0),'m')
        add_ABBA_subplot_details(ax, 'decode')

        ax = f.add_subplot(5,2,n+5)
        ax.plot(delay_accuracy, np.mean(accuracy,axis=1),'b.')
        ax.plot(delay_accuracy, np.mean(accuracy_neural_shuffled,axis=1),'r.')
        ax.plot(delay_accuracy, np.mean(accuracy_syn_shuffled,axis=1),'c.')
        add_ABBA_subplot_details(ax, 'shuffle')

        ax = f.add_subplot(5,2,n+7)
        u = np.mean(tuning_sim[:good_model_count,:],axis=0)
        sd = np.std(tuning_sim[:good_model_count,:],axis=0)/np.sqrt(good_model_count)
        ax.plot(t, u,'k')
        ax.fill_between(t, u-sd, u+sd, color=[0,0,0,0.5])

        add_ABBA_subplot_details(ax, 'tuning')

        if n == 0:
            test2_time = range(25+50+40*4, 25+50+40*5)
            tuning_sim_ABCA = np.mean(tuning_sim[:good_model_count, test2_time], axis=1)
        elif n == 1:
            tuning_sim_ABBA_base = np.mean(tuning_sim[:good_model_count, test2_time], axis=1)
            tuning_sim_ABBA_supp = np.mean(tuning_sim_shuffled[2,:good_model_count, test2_time], axis=0)
            print('ABBA supp ', tuning_sim_ABBA_supp)
            print('ABBA base ', tuning_sim_ABBA_base)


        if n == 1:
            ax = f.add_subplot(5,2,9)
            u = np.mean(tuning_sim[:good_model_count,:],axis=0)
            sd = np.std(tuning_sim[:good_model_count,:],axis=0)/np.sqrt(good_model_count)
            ax.plot(t, u,'k')
            ax.fill_between(t, u-sd, u+sd, color=[0,0,0,0.5])
            col = [[0,0,1],[1,0,0],[0,1,0],[0,1,1]]
            for j in range(4):
                u = np.mean(tuning_sim_shuffled[j,:good_model_count,:],axis=0)
                sd = np.std(tuning_sim_shuffled[j,:good_model_count,:],axis=0)/np.sqrt(good_model_count)
                ax.plot(t, u, color=col[j])
                ax.fill_between(t, u-sd, u+sd, color=col[j]+[0.5])

            add_ABBA_subplot_details(ax, 'tuning')

            acc = np.zeros((5))
            acc_se = np.zeros((5))
            acc[0] = np.mean(accuracy_test2[:good_model_count])
            acc[1:] = np.mean(accuracy_test2_shuffled[:good_model_count, :],axis=0)
            acc_se[0] = np.std(accuracy_test2[:good_model_count])/np.sqrt(good_model_count)
            acc_se[1:] = np.std(accuracy_test2_shuffled[:good_model_count, :],axis=0)/np.sqrt(good_model_count)
            ax = f.add_subplot(5,2,10)
            p0, p1, p2, p3, p4 = ax.bar([0,1,2,3,4], acc, yerr=acc_se)
            p0.set_facecolor('k')
            p1.set_facecolor('b')
            p2.set_facecolor('r')
            p3.set_facecolor('g')
            p4.set_facecolor('c')
            ax.set_ylabel('Behavioral accuracy')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)




    plt.tight_layout()
    plt.savefig('Fig5.pdf', format='pdf')
    plt.show()

    p_val = scipy.stats.ttest_ind(tuning_sim_ABCA,tuning_sim_ABBA_base)[1]
    print('Mean ABCA sim ', np.mean(tuning_sim_ABCA), ' Mean ABBA sim ', np.mean(tuning_sim_ABBA_base) , ' p = ', p_val)

    p_val = scipy.stats.ttest_rel(tuning_sim_ABBA_supp ,tuning_sim_ABBA_base)[1]
    print('Mean ABBA sim supp', np.mean(tuning_sim_ABBA_supp), ' Mean ABBA sim ', np.mean(tuning_sim_ABBA_base) , ' p = ', p_val)


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
        ax.plot([0,0],[0,1],'k--')
        ax.plot([400,400],[0,1],'k--')
        ax.plot([800,800],[0,1],'k--')
        ax.plot([1200,1200],[0,1],'k--')
        ax.plot([1600,1600],[0,1],'k--')
        ax.plot([2000,2000],[0,1],'k--')
        ax.plot([2400,2400],[0,1],'k--')
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
        ax.set_ylim([-0.3,1.02])
        ax.set_xlim([-500,2800])
        ax.plot([0,0],[0,1],'k--')
        ax.plot([400,400],[0,1],'k--')
        ax.plot([800,800],[0,1],'k--')
        ax.plot([1200,1200],[0,1],'k--')
        ax.plot([1600,1600],[0,1],'k--')
        ax.plot([2000,2000],[0,1],'k--')
        ax.plot([2400,2400],[0,1],'k--')
        ax.set_ylabel('Tuning similarity')
        ax.set_xlabel('Time relative to sample onset (ms)')
