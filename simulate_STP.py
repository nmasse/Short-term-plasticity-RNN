import numpy as np
import matplotlib.pyplot as plt

def run_simulation():

    delta_t = 0.05
    time_pts = np.arange(50000)
    x = np.zeros((len(time_pts)+1))
    u = np.zeros((len(time_pts)+1))
    s = np.zeros((len(time_pts)+1))
    
    alpha_std, alpha_stf, U = create_stp_constants(synapse_type='std', delta_t=delta_t)
    u[0] = U
    x[0] = 1
    for t in time_pts:
        if t >= 500//delta_t and t<600//delta_t and t%(20//delta_t)==0:
            spike = 1
            s[t] = 1
        else:
            spike = 0
        x[t+1], u[t+1] = run_sim_step(x[t], u[t], alpha_std, alpha_stf, U, spike)
      
    t = (time_pts -  500//delta_t)*delta_t 
    f = plt.figure(figsize=(8,6))
    ax = f.add_subplot(3, 1, 1)

    ax.plot(t,s[1:],'k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ax = f.add_subplot(3, 1, 2)
    ax.hold(True)
    ax.plot(t,x[1:])
    ax.plot(t,u[1:],'r')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ax = f.add_subplot(3, 1, 3)
    ax.plot(t,u[1:]*x[1:]/U,'k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    plt.savefig('std.pdf', format='pdf')
    plt.show()

def run_sim_step(x, u, alpha_std, alpha_stf, U, spike):
    

    x += alpha_std*(1-x) - u*x*spike
    u += alpha_stf*(U-u) + U*(1-u)*spike
    x = np.minimum(np.float32(1), np.maximum(np.float32(0), x))
    u = np.minimum(np.float32(1), np.maximum(np.float32(0), u))
    
    return x, u


def create_stp_constants(synapse_type='std', delta_t=1):
        
    # synapses can either be stp or std 
    
    if synapse_type == 'std': # make them all depressing
        tau_f = 200 # in milliseconds
        tau_d = 1500
        U = 0.45
            
    elif synapse_type == 'stf': # make them all facilitating
        tau_f = 1500
        tau_d = 200
        U = 0.15
        
    else:
        print('Wrong STP specification')
            
    # convert time constants into decay rates    
    alpha_std = delta_t/tau_d
    alpha_stf = delta_t/tau_f

    return alpha_std, alpha_stf, U