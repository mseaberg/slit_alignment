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
        starting_point = self.mr2k4.pitch.get()

        for i in range(10):
            self.mr2k4.pitch.mvr(1, wait=True)
            time.sleep(2)
        self.mr2k4.pitch.mv(starting_point)
        print('calibration complete')


class Alignment(QtCore.QThread):

    def __init__(self, data_handler, goals):
        QtCore.QThread.__init__(self)
        self.data_handler = data_handler
        self.mr2k4 = KBMirror('MR2K4:KBO')
        self.mr3k4 = KBMirror('MR3K4:KBO')

        # first column is upstream bender, second column is downstream bender
        # first row is focus distance, second column is third order
        Ax_inverse = np.zeros((2, 2))

        # this is the calibration matrix for MR2K4
        # units of top row are z[mm]/bend[mm]
        # units of bottom row are coeff/bend[mm]
        # eventually need to scale out the wavelength for the third order coefficients but this should work at 870 eV
        Ax_inverse[:, 0] = np.array([15, -142e6])
        Ax_inverse[:, 1] = np.array([23, 115e6])

        self.Ax = np.linalg.inv(Ax_inverse)

        # this is the calibration matrix for MR3K4. Details are the same as above for MR2K4
        Ay_inverse = np.zeros((2, 2))

        Ay_inverse[:, 0] = np.array([13.3, -120e6])
        Ay_inverse[:, 1] = np.array([16.7, 83e6])

        self.Ay = np.linalg.inv(Ay_inverse)

        # goals is just a dictionary. Each entry in the dictionary is a 1D array. The second entry will probably
        # always be zero since this corresponds to undesirable 3rd order
        self.x_goals = goals['x']
        self.y_goals = goals['y']

    def run(self):
        # need to get some updates from the RunProcessing object to see where we are currently. We also need to

        # need a while loop here to collect some data
        counter = 0

        z_x = 0
        z_y = 0
        coma_x = 0
        coma_y = 0

        while counter < 3:

            print('counter %d' % counter)

            data_dict = self.data_handler.data_dict
            # wait for at least 3 shots in a row of the incoming data to be valid
            counter = np.sum(data_dict['wavefront_is_valid'][-3:])
            z_x = np.mean(data_dict['z_x'][-3:])
            z_y = np.mean(data_dict['z_y'][-3:])
            coma_x = np.mean(data_dict['coma_x'][-3:])
            coma_y = np.mean(data_dict['coma_y'][-3:])

            # wait for a couple seconds before checking again
            time.sleep(2)

        # calculate desired move
        current_x = np.array([z_x, coma_x])
        current_y = np.array([z_y, coma_y])

        delta_x = self.x_goals - current_x
        delta_y = self.y_goals - current_y

        motion_x = np.dot(self.Ax, delta_x)
        motion_y = np.dot(self.Ay, delta_y)

        # move the mirrors
        # self.mr2k4.us.mvr(motion_x[0])
        # self.mr2k4.ds.mvr(motion_x[1])
        # self.mr3k4.us.mvr(motion_y[0])
        # self.mr3k4.ds.mvr(motion_y[1])


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

