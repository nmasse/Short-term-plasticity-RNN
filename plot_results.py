import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "arial"

def plot_figure3():

    data_dir = 'C:/Users/nicol/Projects/RNN_STP_analysis/'
    tasks = ['DMS', 'DMRS180','DMRS90','DMC']
    num_tasks = len(tasks)
    accuracy_th = 0.9 # require all networks to have at least this performance accuracy
    models_per_task = 20
    N = 100 # number of bootstrap repeats for each value

    # will calculate neuronal decoding accuracy during the last 500 ms of delay
    dt=10
    t = range(-900,2000,dt)
    delay_epoch = range(2300//dt,2400//dt)

    f = plt.figure(figsize=(6,8.5))
    chance_level = 1/8

    for n in range(num_tasks):

        if tasks[n] == 'DMC':
            chance_level = 1/2
        else:
            chance_level = 1/8

        # load following results from each task
        delay_accuracy = np.zeros((models_per_task), dtype=np.float32)
        neuronal_decoding = np.zeros((models_per_task, N, len(t)), dtype=np.float32)
        synaptic_decoding = np.zeros((models_per_task, N, len(t)), dtype=np.float32)
        accuracy = np.zeros((models_per_task, N), dtype=np.float32)
        accuracy_neural_shuffled = np.zeros((models_per_task, N), dtype=np.float32)
        accuracy_syn_shuffled = np.zeros((models_per_task, N), dtype=np.float32)

        good_model_count = 0
        count = 0
        while good_model_count < models_per_task:
            x = pickle.load(open(data_dir + tasks[n] + '_' + str(count+1) + '.pkl', 'rb'))
            count += 1
            if np.mean(x['accuracy']) >  accuracy_th:
                delay_accuracy[good_model_count] = np.mean(x['neuronal_decoding'][:,:,delay_epoch])
                neuronal_decoding[good_model_count,:,:] = x['neuronal_decoding'][0,:,:]
                synaptic_decoding[good_model_count,:,:] = x['synaptic_decoding'][0,:,:]
                accuracy[good_model_count,:] = x['accuracy']
                accuracy_neural_shuffled[good_model_count,:] = x['accuracy_neural_shuffled']
                accuracy_syn_shuffled[good_model_count,:] = x['accuracy_syn_shuffled']
                good_model_count +=1

        if good_model_count < models_per_task:
            print('Too few accurately trained models')

        ax = f.add_subplot(num_tasks, 2, 2*n+1)
        plt.hold(True)
        for j in range(models_per_task):
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
    plt.savefig('Summary.pdf', format='pdf')
    plt.show()
