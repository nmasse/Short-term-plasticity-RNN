import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "arial"


def plot_all_figures():

    fig_params = {
        'data_dir'              : 'C:/Users/nicol/Projects/RNN_STP_analysis/',
        'dt'                    : 10,
        'models_per_task'       : 20,
        'N'                     : 100, # bootstrap iterations
        'accuracy_th'           : 0.9} # minimum accuracy of model required for analysis
    #plot_supp_figure(fig_params)
    #plot_figure3(fig_params)
    #plot_figure4(fig_params)
    plot_figure5(fig_params)

def plot_supp_figure(fig_params):

    num_tasks = 3
    chance_level = 1/8
    model_signficance = np.zeros((num_tasks))
    f = plt.figure(figsize=(3,4.25))

    for n in range(num_tasks):

        t = range(-900,2000+n*500,fig_params['dt'])
        delay_epoch = range((2300+n*500)//fig_params['dt'],(2400+n*500)//fig_params['dt'])

        # load following results from each task
        delay_accuracy = np.zeros((fig_params['models_per_task']), dtype=np.float32)
        neuronal_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        synaptic_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)

        good_model_count = 0
        count = 1
        while good_model_count < fig_params['models_per_task']:
            if n == 0:
                task_name = 'DMS_' + str(count)
            elif n == 1:
                task_name = 'DMS_' + str(count) + '_1500'
            elif n == 2:
                task_name = 'DMS_' + str(count) + '_2000'
            x = pickle.load(open(fig_params['data_dir'] + task_name + '.pkl', 'rb'))
            count += 1
            if np.mean(x['accuracy']) >  fig_params['accuracy_th']:
                delay_accuracy[good_model_count] = np.mean(x['neuronal_decoding'][0,:,delay_epoch])
                neuronal_decoding[good_model_count,:,:] = x['neuronal_decoding'][0,:,:]
                synaptic_decoding[good_model_count,:,:] = x['synaptic_decoding'][0,:,:]
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

def plot_figure4(fig_params):

    task = 'DMS+DMRS'
    t = range(-900,2000,fig_params['dt'])
    delay_epoch = range(2300//fig_params['dt'],2400//fig_params['dt'])
    f = plt.figure(figsize=(6,4.25))
    chance_level = 1/8

    # for each task, we will measure model significance with respect to:
    # dim 1 = 0, neuronal decoding during delay_accuracy
    # dim 1 = 1, shuffled neuronal accuracy > chance
    # dim 1 = 2, shuffled neuronal accuracy < accuracy
    # dim 1 = 3, shuffled synaptic accuracy > chance
    # dim 1 = 4, shuffled synaptic accuracy < accuracy
    model_signficance = np.zeros((2, 5))
    p_val_th = 0.05

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
    while good_model_count < fig_params['models_per_task']:
        x = pickle.load(open(fig_params['data_dir'] + task + '_' + str(count+1) + '.pkl', 'rb'))
        count += 1
        if np.mean(x['accuracy'][0,:]) > fig_params['accuracy_th'] and np.mean(x['accuracy'][1,:]) > fig_params['accuracy_th']:
            for j in  range(2):
                delay_accuracy[j, good_model_count] = np.mean(x['neuronal_decoding'][j,:,delay_epoch])
                neuronal_decoding[j, good_model_count,:,:] = x['neuronal_decoding'][j,:,:]
                synaptic_decoding[j, good_model_count,:,:] = x['synaptic_decoding'][j,:,:]
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


def plot_figure3(fig_params):

    tasks = ['DMS', 'DMRS180','DMRS45','DMRS90','DMC']
    num_tasks = len(tasks)

    t = range(-900,2000,fig_params['dt'])
    delay_epoch = range(2300//fig_params['dt'],2400//fig_params['dt'])

    f = plt.figure(figsize=(6,8.5))
    chance_level = 1/8

    # for each task, we will measure model significance with respect to:
    # dim 1 = 0, neuronal decoding during delay_accuracy
    # dim 1 = 1, shuffled neuronal accuracy > chance
    # dim 1 = 2, shuffled neuronal accuracy < accuracy
    # dim 1 = 3, shuffled synaptic accuracy > chance
    # dim 1 = 4, shuffled synaptic accuracy < accuracy
    model_signficance = np.zeros((num_tasks, 5))

    sig_neuronal_delay = np.zeros((num_tasks, fig_params['models_per_task']))
    sig_decrease_neuronal_shuffling = np.zeros((num_tasks, fig_params['models_per_task']))
    sig_decrease_syn_shuffling = np.zeros((num_tasks, fig_params['models_per_task']))
    sig_syn_shuffling = np.zeros((num_tasks, fig_params['models_per_task']))

    # correlation between neuronal decoding during the delay, accuracy after sig_syn_shuffling
    # neuronal activity, and accuracy after shuffling synaptic activity
    corr_decoding_neuronal_shuf =  np.zeros((num_tasks,2))
    corr_decoding_syn_shuf =  np.zeros((num_tasks,2))
    corr_neuronal_shuf_syn_shuf =  np.zeros((num_tasks,2))

    decoding_p_val =  np.zeros((num_tasks))

    p_val_th = 0.01

    # will use DMS decoding results for comparison
    neuronal_decoding_DMS  = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)

    for n in range(num_tasks):

        if tasks[n] == 'DMC':
            chance_level = 1/2
        else:
            chance_level = 1/8

        # load following results from each task
        delay_accuracy = np.zeros((fig_params['models_per_task']), dtype=np.float32)
        neuronal_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        synaptic_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        accuracy = np.zeros((fig_params['models_per_task'], fig_params['N']), dtype=np.float32)
        accuracy_neural_shuffled = np.zeros((fig_params['models_per_task'], fig_params['N']), dtype=np.float32)
        accuracy_syn_shuffled = np.zeros((fig_params['models_per_task'], fig_params['N']), dtype=np.float32)


        good_model_count = 0
        count = 0
        while good_model_count < fig_params['models_per_task']:
            x = pickle.load(open(fig_params['data_dir'] + tasks[n] + '_' + str(count+1) + '.pkl', 'rb'))
            count += 1
            if np.mean(x['accuracy']) >  fig_params['accuracy_th']:
                delay_accuracy[good_model_count] = np.mean(x['neuronal_decoding'][0,:,delay_epoch])
                neuronal_decoding[good_model_count,:,:] = x['neuronal_decoding'][0,:,:]
                if tasks[n] == 'DMS':
                    neuronal_decoding_DMS[good_model_count,:,:] = x['neuronal_decoding'][0,:,:]
                synaptic_decoding[good_model_count,:,:] = x['synaptic_decoding'][0,:,:]
                accuracy[good_model_count,:] = x['accuracy']
                accuracy_neural_shuffled[good_model_count,:] = x['accuracy_neural_shuffled']
                accuracy_syn_shuffled[good_model_count,:] = x['accuracy_syn_shuffled']
                good_model_count +=1

        if good_model_count < fig_params['models_per_task']:
            print('Too few accurately trained models')

        model_signficance[n, 0] = np.sum(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2) \
            >chance_level,axis=1)>1-p_val_th)


        model_signficance[n, 1] = np.sum(np.mean(accuracy_neural_shuffled>0.5,axis=1)>1-p_val_th)
        model_signficance[n, 2] = np.sum(np.mean(accuracy-accuracy_neural_shuffled>0,axis=1)>1-p_val_th)
        model_signficance[n, 3] = np.sum(np.mean(accuracy_syn_shuffled>0.5,axis=1)>1-p_val_th)
        model_signficance[n, 4] =  np.sum(np.mean(accuracy-accuracy_syn_shuffled>0,axis=1)>1-p_val_th)
        sig_neuronal_delay[n,:] =np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2)>chance_level,axis=1)>1-p_val_th
        sig_decrease_neuronal_shuffling[n,:] =  np.mean(accuracy-accuracy_neural_shuffled>0,axis=1)>1-p_val_th
        sig_decrease_syn_shuffling[n,:] =  np.mean(accuracy-accuracy_syn_shuffled>0,axis=1)>1-p_val_th
        sig_syn_shuffling[n,:] = np.mean(accuracy_syn_shuffled>0.5,axis=1)>1-p_val_th


        N = 100
        a = np.reshape(np.tile(accuracy,(1,1,N)),(20,N**2))
        b = np.tile(accuracy_neural_shuffled,(1,1,N))
        b = np.reshape(np.transpose(b,(0,2,1)),(20,N**2))
        #print(tasks[n], np.mean(a-b>0,axis=1))
        ind = np.where(np.mean(accuracy_neural_shuffled,axis=1)<0.9)[0]
        print(tasks[n], ind)
        print(tasks[n], np.mean(np.mean(accuracy_neural_shuffled[ind,:],axis=1)))
        print(tasks[n], np.mean(np.mean(accuracy_syn_shuffled[ind,:],axis=1)))


        corr_decoding_neuronal_shuf[n,:] = scipy.stats.pearsonr(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2),axis=1), \
            np.mean(accuracy_neural_shuffled,axis=1))
        corr_decoding_syn_shuf[n,:] = scipy.stats.pearsonr(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2),axis=1), \
            np.mean(accuracy_syn_shuffled,axis=1))
        corr_neuronal_shuf_syn_shuf[n,:] = scipy.stats.pearsonr(np.mean(accuracy_neural_shuffled,axis=1), \
            np.mean(accuracy_syn_shuffled,axis=1))

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
        ax.set_xticks([0,500,1500-10])
        ax.set_ylim([0,1.02])
        ax.set_xlim([-500,1500-10])
        ax.plot([-900,2000],[chance_level,chance_level],'k--')
        ax.plot([0,0],[0,1],'k--')
        ax.plot([500,500],[0,1],'k--')
        ax.set_ylabel('Decoding accuracy')
        ax.set_xlabel('Time relative to sample onset (ms)')


        ax = f.add_subplot(num_tasks, 2, 2*n+2)
        plt.hold(True)
        ax.plot(delay_accuracy, np.mean(accuracy,axis=1),'b.')
        ax.plot(delay_accuracy, np.mean(accuracy_neural_shuffled,axis=1),'r.')
        ax.plot(delay_accuracy, np.mean(accuracy_syn_shuffled,axis=1),'c.')
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
    plt.savefig('Fig3.pdf', format='pdf')
    plt.show()

    print('Number of models with delay neuronal decoding accuracy signficantly greater than chance')
    print(model_signficance)

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

    print('Correlations...')
    print(corr_decoding_neuronal_shuf)
    print(corr_decoding_syn_shuf)
    print(corr_neuronal_shuf_syn_shuf)
    print('T-test neuroanl decoding p-val compared to DMS')
    print(decoding_p_val)


def plot_figure5(fig_params):

    tasks = ['ABCA', 'ABBA']
    num_tasks = len(tasks)

    t = range(-900,4000,fig_params['dt'])
    delay_epoch = range(4300//fig_params['dt'],4400//fig_params['dt'])

    f = plt.figure(figsize=(6,4))
    chance_level = 1/8

    # for each task, we will measure model significance with respect to:
    # dim 1 = 0, neuronal decoding during delay_accuracy
    # dim 1 = 1, shuffled neuronal accuracy > chance
    # dim 1 = 2, shuffled neuronal accuracy < accuracy
    # dim 1 = 3, shuffled synaptic accuracy > chance
    # dim 1 = 4, shuffled synaptic accuracy < accuracy
    model_signficance = np.zeros((num_tasks, 5))

    sig_neuronal_delay = np.zeros((num_tasks, fig_params['models_per_task']))
    sig_decrease_neuronal_shuffling = np.zeros((num_tasks, fig_params['models_per_task']))
    sig_decrease_syn_shuffling = np.zeros((num_tasks, fig_params['models_per_task']))
    sig_syn_shuffling = np.zeros((num_tasks, fig_params['models_per_task']))

    # correlation between neuronal decoding during the delay, accuracy after sig_syn_shuffling
    # neuronal activity, and accuracy after shuffling synaptic activity
    corr_decoding_neuronal_shuf =  np.zeros((num_tasks,2))
    corr_decoding_syn_shuf =  np.zeros((num_tasks,2))
    corr_neuronal_shuf_syn_shuf =  np.zeros((num_tasks,2))

    decoding_p_val =  np.zeros((num_tasks))
    p_val_th = 0.01
    chance_level = 1/8


    for n in range(num_tasks):

        # load following results from each task
        delay_accuracy = np.zeros((fig_params['models_per_task']), dtype=np.float32)
        neuronal_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        synaptic_decoding = np.zeros((fig_params['models_per_task'], fig_params['N'], len(t)), dtype=np.float32)
        accuracy = np.zeros((fig_params['models_per_task'], fig_params['N']), dtype=np.float32)
        accuracy_neural_shuffled = np.zeros((fig_params['models_per_task'], fig_params['N']), dtype=np.float32)
        accuracy_syn_shuffled = np.zeros((fig_params['models_per_task'], fig_params['N']), dtype=np.float32)

        good_model_count = 0
        count = 0
        while good_model_count < fig_params['models_per_task'] and count < 49:
            count += 1
            try:
                x = pickle.load(open(fig_params['data_dir'] + tasks[n] + '_' + str(count) + '_mask.pkl', 'rb'))
            except:
                continue

            if np.mean(x['accuracy']) >  fig_params['accuracy_th']:
            #if 1 >  fig_params['accuracy_th']:
                delay_accuracy[good_model_count] = np.mean(x['neuronal_decoding'][0,:,delay_epoch])
                neuronal_decoding[good_model_count,:,:] = x['neuronal_decoding'][0,:,:]
                if tasks[n] == 'DMS':
                    neuronal_decoding_DMS[good_model_count,:,:] = x['neuronal_decoding'][0,:,:]
                synaptic_decoding[good_model_count,:,:] = x['synaptic_decoding'][0,:,:]
                accuracy[good_model_count,:] = x['accuracy']
                accuracy_neural_shuffled[good_model_count,:] = x['accuracy_neural_shuffled']
                accuracy_syn_shuffled[good_model_count,:] = x['accuracy_syn_shuffled']
                good_model_count +=1

        if good_model_count < fig_params['models_per_task']:
            print('Too few accurately trained models, good models = ', good_model_count)

        model_signficance[n, 0] = np.sum(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2) \
            >chance_level,axis=1)>1-p_val_th)

        model_signficance[n, 1] = np.sum(np.mean(accuracy_neural_shuffled>0.5,axis=1)>1-p_val_th)
        model_signficance[n, 2] = np.sum(np.mean(accuracy-accuracy_neural_shuffled>0,axis=1)>1-p_val_th)
        model_signficance[n, 3] = np.sum(np.mean(accuracy_syn_shuffled>0.5,axis=1)>1-p_val_th)
        model_signficance[n, 4] =  np.sum(np.mean(accuracy-accuracy_syn_shuffled>0,axis=1)>1-p_val_th)
        sig_neuronal_delay[n,:] =np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2)>chance_level,axis=1)>1-p_val_th
        sig_decrease_neuronal_shuffling[n,:] =  np.mean(accuracy-accuracy_neural_shuffled>0,axis=1)>1-p_val_th
        sig_decrease_syn_shuffling[n,:] =  np.mean(accuracy-accuracy_syn_shuffled>0,axis=1)>1-p_val_th
        sig_syn_shuffling[n,:] = np.mean(accuracy_syn_shuffled>0.5,axis=1)>1-p_val_th


        N = 100
        a = np.reshape(np.tile(accuracy,(1,1,N)),(20,N**2))
        b = np.tile(accuracy_neural_shuffled,(1,1,N))
        b = np.reshape(np.transpose(b,(0,2,1)),(20,N**2))
        #print(tasks[n], np.mean(a-b>0,axis=1))
        ind = np.where(np.mean(accuracy_neural_shuffled,axis=1)<0.9)[0]
        print(tasks[n], ind)
        print(tasks[n], np.mean(np.mean(accuracy_neural_shuffled[ind,:],axis=1)))
        print(tasks[n], np.mean(np.mean(accuracy_syn_shuffled[ind,:],axis=1)))


        corr_decoding_neuronal_shuf[n,:] = scipy.stats.pearsonr(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2),axis=1), \
            np.mean(accuracy_neural_shuffled,axis=1))
        corr_decoding_syn_shuf[n,:] = scipy.stats.pearsonr(np.mean(np.mean(neuronal_decoding[:,:,delay_epoch],axis=2),axis=1), \
            np.mean(accuracy_syn_shuffled,axis=1))
        corr_neuronal_shuf_syn_shuf[n,:] = scipy.stats.pearsonr(np.mean(accuracy_neural_shuffled,axis=1), \
            np.mean(accuracy_syn_shuffled,axis=1))


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
        ax.set_xticks([0,500,1500,2500,3500])
        ax.set_ylim([0,1.02])
        ax.set_xlim([-500,4000-10])
        ax.plot([-900,4000],[chance_level,chance_level],'k--')
        ax.plot([0,0],[0,1],'k--')
        ax.plot([500,500],[0,1],'k--')
        ax.set_ylabel('Decoding accuracy')
        ax.set_xlabel('Time relative to sample onset (ms)')


        ax = f.add_subplot(num_tasks, 2, 2*n+2)
        plt.hold(True)
        ax.plot(delay_accuracy, np.mean(accuracy,axis=1),'b.')
        ax.plot(delay_accuracy, np.mean(accuracy_neural_shuffled,axis=1),'r.')
        ax.plot(delay_accuracy, np.mean(accuracy_syn_shuffled,axis=1),'c.')
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
    plt.savefig('Fig5.pdf', format='pdf')
    plt.show()

    print('Number of models with delay neuronal decoding accuracy signficantly greater than chance')
    print(model_signficance)

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

    print('Correlations...')
    print(corr_decoding_neuronal_shuf)
    print(corr_decoding_syn_shuf)
    print(corr_neuronal_shuf_syn_shuf)
