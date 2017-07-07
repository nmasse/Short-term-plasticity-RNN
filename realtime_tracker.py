from parameters import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
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

def run_GUI():
    app = QtGui.QApplication([])
    w   = pg.GraphicsWindow()
    w.setWindowTitle('Realtime Network Behavior')

    start_btn   = QtGui.QPushButton('Run Model')
    stop_btn    = QtGui.QPushButton('Quit')
    title       = QtGui.QLabel('Realtime Network Tracking and Analysis')
    title.setStyleSheet(white_label)

def stop():
    sys.exit()


def main():

    sys.exit()
