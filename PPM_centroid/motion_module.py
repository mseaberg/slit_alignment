import numpy as np
import time
from pyqtgraph.Qt import QtCore
from ophyd import EpicsSignalRO as SignalRO


class Calibration(QtCore.QThread):

    def __init__(self, data_handler):
        #super(Calibration, self).__init__()
        QtCore.QThread.__init__(self)
        self.data_handler = data_handler

    def run(self):
        for i in range(100):
            print('running calibration')
            time.sleep(.1)

