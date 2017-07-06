from parameters import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import threading
import model
import psutil
import sys
import os

global data
data = {
    'trials'    : [],
    'perf_loss' : [],
    'spike_loss': [],
    'activity'  : [],
    'accuracy'  : []
}

def start():
    par['df_num'] = '0004'
    model.main(switch, update)

def stop():
    sys.exit()

def switch(iteration, prev_switch_iteration, savename):
    if iteration == (prev_switch_iteration + 10):
        if par['allowed_categories'] == [0]:
            par['allowed_categories'] = [1]
            print("Switching to category 1.\n")
            with open(savename, 'a') as f:
                f.write('Switching to category 1.\n')
            return iteration
        elif par['allowed_categories'] == [1]:
            par['allowed_categories'] = [0]
            print("Switching to category 0.\n")
            with open(savename, 'a') as f:
                f.write('Switching to category 0.\n')
            return iteration
        else:
            print("ERROR: Bad category.")
            quit()
    else:
        return prev_switch_iteration


###############################
### GUI and update handlers ###
###############################

white_label = "QLabel { color : white; }"

def update(trial, perf, spike, act, acc):
    data['trials'].append(trial)
    data['perf_loss'].append(perf)
    data['spike_loss'].append(spike)
    data['activity'].append(act)
    data['accuracy'].append(acc)

def GUI():
    app = QtGui.QApplication([])
    w   = pg.GraphicsWindow()

    w.setWindowTitle('Realtime Network Behavior')

    start_btn = QtGui.QPushButton('Run Model')
    stop_btn  = QtGui.QPushButton('Quit')
    title     = QtGui.QLabel('Realtime Network Behavior')
    title.setStyleSheet(white_label)

    pars_lab_text = \
    "Using dendrites:\n\nUsing EI network:\n\nSynaptic configuration:\n\n"\
    + "Stimulus type:\n\nAllowed fields:\n\nAllowed categories:\n\n"\
    + "Dendrite function:\n\nDendrites per neuron:\n\nRun length:"

    pars_resp_lab_text = \
    str(par['use_dendrites']) + "\n\n" + str(par['EI']) + "\n\n" + str(par['synapse_config']) + "\n\n"\
    + str(par['stimulus_type']) + "\n\n" + str(par['allowed_fields']) + "\n\n" + str(par['allowed_categories']) + "\n\n"\
    + str(par['df_num']) + "\n\n" + str(par['den_per_unit']) + "\n\n" + str(par['projected_num_trials'])

    par1      = QtGui.QLabel(pars_lab_text)
    par1_resp = QtGui.QLabel(pars_resp_lab_text)

    par1.setStyleSheet(white_label)
    par1_resp.setStyleSheet(white_label)

    p1 = pg.PlotWidget()


    layout = QtGui.QGridLayout()
    w.setLayout(layout)

    layout.addWidget(start_btn, 0, 0)
    layout.addWidget(stop_btn, 0, 1)
    layout.addWidget(title, 0, 2, 1, 3)

    layout.addWidget(par1, 1, 0)
    layout.addWidget(par1_resp, 1, 1)

    layout.addWidget(p1, 1, 3, 1, 3)

    w.show()

    t_start = threading.Thread(target=start)


    start_btn.clicked.connect(t_start.start)
    stop_btn.clicked.connect(stop)

    p1.addLegend()
    p1.setYRange(0, 1, padding=0.02)

    curve1 = p1.plot(x=data['trials'], y=data['accuracy'], name='Accuracy', pen=(255,0,0))
    curve2 = p1.plot(x=data['trials'], y=data['perf_loss'], name='Perf. Loss', pen=(0,255,0))
    curve3 = p1.plot(x=data['trials'], y=data['activity'], name='Activity', pen=(0,0,255))
    curve4 = p1.plot(x=data['trials'], y=data['spike_loss'], name='SpikeLoss', pen=(50,50,200))

    def updateGUI():
        pars_resp_lab_text = \
        str(par['use_dendrites']) + "\n\n" + str(par['EI']) + "\n\n" + str(par['synapse_config']) + "\n\n"\
        + str(par['stimulus_type']) + "\n\n" + str(par['allowed_fields']) + "\n\n" + str(par['allowed_categories']) + "\n\n"\
        + str(par['df_num']) + "\n\n" + str(par['den_per_unit']) + "\n\n" + str(par['projected_num_trials'])

        par1_resp.setText(pars_resp_lab_text)

        curve1.setData(x=data['trials'], y=data['accuracy'])
        curve2.setData(x=data['trials'], y=data['perf_loss'])
        curve3.setData(x=data['trials'], y=data['activity'])
        curve4.setData(x=data['trials'], y=data['spike_loss'])
        app.processEvents()

    timer = QtCore.QTimer()
    timer.timeout.connect(updateGUI)
    timer.start(50)

    app.exec_()

t_GUI = threading.Thread(target=GUI)
t_GUI.start()
