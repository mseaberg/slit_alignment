import numpy as np
import time
from pyqtgraph.Qt import QtCore
from ophyd import EpicsSignalRO as SignalRO
from ophyd import EpicsSignal as Signal


class Calibration(QtCore.QThread):

    def __init__(self, data_handler):
        #super(Calibration, self).__init__()
        QtCore.QThread.__init__(self)
        self.data_handler = data_handler
        self.mr2k4 = KBMirror('MR2K4:KBO')
        self.mr3k4 = KBMirror('MR3K4:KBO')

    def run(self):
        starting_point = self.mirror.pitch.get()

        for i in range(10):
            self.mirror.pitch.mvr(1, wait=True)
            time.sleep(2)
        self.mirror.pitch.mv(starting_point)
        print('calibration complete')


class Motor():
    def __init__(self, pv_name):
        self.setpoint = Signal(pv_name)
        self.rbv = SignalRO(pv_name+'.RBV', auto_monitor=True)

    def mv(self, target, wait=False, tol=0.1):
        self.set(target)
        if wait:
            while np.abs(self.get() - target) > tol:
                time.sleep(0.2)

    def mvr(self, adjustment, wait=False, tol=0.1):
        target = self.get() + adjustment
        self.mv(target, wait=wait, tol=tol)

    def get(self):
        return self.rbv.value

    def set(self, target):
        self.setpoint.set(target)



class KBMirror():

    def __init__(self, mirror_base):
        # initialize attributes
        self.mirror_base = mirror_base
        self.motor_base = self.mirror_base + ':MMS'
        # initialize epics signals
        self.pitch = Motor(self.motor_base+':PITCH')
        self.x = Motor(self.motor_base+':X')
        self.y = Motor(self.motor_base+':Y')
        self.us = Motor(self.motor_base+':BEND:US')
        self.ds = Motor(self.motor_base+':BEND:DS')

