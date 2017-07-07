from parameters import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import threading
import model
import time
import sys

global data
data = {
    'trials'    : [0],
    'perf_loss' : [0.],
    'spike_loss': [0.],
    'activity'  : [0.],
    'accuracy'  : [0.]
}

white_label = "QLabel { color : white; }"

def update_data(trial, acc):
    data['trials'].append(trial)
    #data['perf_loss'].append(perf)
    #data['spike_loss'].append(spike)
    #data['activity'].append(act)
    data['accuracy'].append(acc)


def stop():
    print("Quitting model.")
    sys.exit()


def main(switch_func):

    t_start = threading.Thread(target=model.main, args=(switch_func,))
    t_start.start()

    app = QtGui.QApplication([])
    w   = pg.GraphicsWindow()
    w.setWindowIcon(QtGui.QIcon('./resources/other/hud_icon.png'))
    w.setWindowTitle('Network HUD')
    w.setFixedSize(400,350)
    #w.setWindowFlags(QtCore.Qt.FramelessWindowHint)
    #w.move(0, 0)

    stop_btn    = QtGui.QPushButton('Quit Model')
    par1        = QtGui.QLabel('')
    par1.setStyleSheet(white_label)

    p1 = pg.PlotWidget()

    layout = QtGui.QGridLayout()
    w.setLayout(layout)

    layout.addWidget(stop_btn, 0, 1)
    layout.addWidget(par1, 0, 0)
    layout.addWidget(p1, 1, 0, 1, 2)

    w.show()

    stop_btn.clicked.connect(stop)

    p1.addLegend()
    p1.setYRange(0, 1, padding=0.1)

    curve1 = p1.plot(x=data['trials'], y=data['accuracy'], name='Accuracy', pen=(255,0,0))

    def updateHUD():
        new_text1 = 'Current trial: {:d}'.format(data['trials'][-1]) + \
                    '\n\nCurrent accuracy: {:.4f}'.format(data['accuracy'][-1])

        par1.setText(new_text1)
        curve1.setData(x=data['trials'], y=data['accuracy'])

        app.processEvents()

    timer = QtCore.QTimer()
    timer.timeout.connect(updateHUD)
    timer.start(50)

    sys.exit(app.exec_())
