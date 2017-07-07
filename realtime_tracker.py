from parameters import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import threading
import model
import time
import sys

global data
data = {
    'trials'    : [],
    'perf_loss' : [],
    'spike_loss': [],
    'activity'  : [],
    'accuracy'  : []
}

white_label = "QLabel { color : white; }"

def update_date(trial, perf, spike, act, acc):
    data['trials'].append(trial)
    data['perf_loss'].append(perf)
    data['spike_loss'].append(spike)
    data['activity'].append(act)
    data['accuracy'].append(acc)

def run_GUI(switch_func):
    app = QtGui.QApplication([])
    w   = pg.GraphicsWindow()
    w.setWindowTitle('Realtime Network Behavior')

    start_btn   = QtGui.QPushButton('Run Model')
    stop_btn    = QtGui.QPushButton('Quit')
    title       = QtGui.QLabel('Realtime Network Tracking and Analysis')
    title.setStyleSheet(white_label)

    pars_label_text1 = \
    "Using dendrites:\n\nUsing EI network:\n\nSynaptic configuration:\n\n"\
    + "Stimulus type:\n\nAllowed fields:\n\nAllowed categories:\n\n"\
    + "Dendrite function:\n\nDendrites per neuron:\n\nRun length:"

    pars_label_text2 = \
    str(par['use_dendrites']) + "\n\n" + str(par['EI']) + "\n\n" + str(par['synapse_config']) + "\n\n"\
    + str(par['stimulus_type']) + "\n\n" + str(par['allowed_fields']) + "\n\n" + str(par['allowed_categories']) + "\n\n"\
    + str(par['df_num']) + "\n\n" + str(par['den_per_unit']) + "\n\n" + str(par['projected_num_trials']) + ' trials'

    par1 = QtGui.QLabel(pars_label_text1)
    par2 = QtGui.QLabel(pars_label_text2)

    par1.setStyleSheet(white_label)
    par2.setStyleSheet(white_label)

    p1 = pg.PlotWidget()

    layout = QtGui.QGridLayout()
    w.setLayout(layout)

    layout.addWidget(start_btn, 0, 0)
    layout.addWidget(stop_btn, 0, 1)
    layout.addWidget(title, 0, 2, 1, 3)

    layout.addWidget(par1, 1, 0)
    layout.addWidget(par2, 1, 1)

    layout.addWidget(p1, 1, 3, 1, 3)

    w.show()

    t_start = threading.Thread(target=run_model, args=(switch_func,))
    start_btn.clicked.connect(t_start.start)
    stop_btn.clicked.connect(stop)

    p1.addLegend()
    p1.setYRange(0, 1, padding=0.1)

    curve1 = p1.plot(x=data['trials'], y=data['accuracy'], name='Accuracy', pen=(255,0,0))
    curve2 = p1.plot(x=data['trials'], y=data['perf_loss'], name='Perf. Loss', pen=(0,255,0))
    curve3 = p1.plot(x=data['trials'], y=data['activity'], name='Activity', pen=(0,0,255))
    curve4 = p1.plot(x=data['trials'], y=data['spike_loss'], name='SpikeLoss', pen=(50,50,200))

    def updateGUI():
        new_pars = \
        str(par['use_dendrites']) + "\n\n" + str(par['EI']) + "\n\n" + str(par['synapse_config']) + "\n\n"\
        + str(par['stimulus_type']) + "\n\n" + str(par['allowed_fields']) + "\n\n" + str(par['allowed_categories']) + "\n\n"\
        + str(par['df_num']) + "\n\n" + str(par['den_per_unit']) + "\n\n" + str(par['projected_num_trials']) + ' trials'

        par2.setText(new_pars)

        curve1.setData(x=data['trials'], y=data['accuracy'])
        curve2.setData(x=data['trials'], y=data['perf_loss'])
        curve3.setData(x=data['trials'], y=data['activity'])
        curve4.setData(x=data['trials'], y=data['spike_loss'])
        app.processEvents()

    timer = QtCore.QTimer()
    timer.timeout.connect(updateGUI)
    timer.start(50)

    app.exec_()

def run_model(switch_func):
    model.main(switch_func)


def stop():
    print("\nQuitting...\n")
    sys.exit()


def main(switch_func):

    t_GUI = threading.Thread(target=run_GUI, args=(switch_func,))
    t_GUI.start()

    sys.exit()
