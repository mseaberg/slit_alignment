from epics import PV
import numpy as np
import scipy.ndimage.interpolation as interpolate
import scipy.ndimage as ndimage
import time
from pyqtgraph.Qt import QtCore
from pcdsdevices.areadetector.detectors import PCDSAreaDetector
from lcls_beamline_toolbox.xraybeamline2d import optics
import sys
import pandas as pd


class RunProcessing(QtCore.QObject):
    sig = QtCore.pyqtSignal(dict)

    def __init__(self, imager_prefix, data_dict, wfs_name=None):
        super(RunProcessing, self).__init__()

        # get wavefront sensor (may be None)
        self.wfs_name = wfs_name

        if wfs_name is not None:
            self.WFS_object = optics.WFS_Device(wfs_name)
        else:
            self.WFS_object = None

        # PPM object for image acquisition and processing
        self.PPM_object = optics.PPM_Device(imager_prefix, threshold=0.1)

        self.running = True

        # frame rate initialization
        self.fps = 0.
        self.lastupdate = time.time()

        # initialize dictionary to pass for plotting
        self.data_dict = data_dict

        self.counter = self.data_dict['counter']

        #### Start  #####################
        self._update()
        
    def get_FOV(self):
        width = self.PPM_object.FOV
        height = np.copy(width)
        return width, height

    def update_1d_data(self, dict_key, new_value):
        self.data_dict[dict_key] = np.roll(self.data_dict[dict_key], -1)
        self.data_dict[dict_key][-1] = new_value

    def running_average(self, source_key, dest_key):
        self.data_dict[dest_key] = pd.Series(self.data_dict[source_key]).rolling(10, min_periods=1).mean().values

    def _update(self):

        if self.running:
            # get latest image
            self.PPM_object.get_image()

            # update dictionary
            self.update_1d_data('cx', self.PPM_object.cx)
            self.update_1d_data('cy', self.PPM_object.cy)
            self.update_1d_data('wx', self.PPM_object.wx)
            self.update_1d_data('wy', self.PPM_object.wy)
            self.running_average('cx', 'cx_smooth')
            self.running_average('cy', 'cy_smooth')
            self.running_average('wx', 'wx_smooth')
            self.running_average('wy', 'wy_smooth')
            self.update_1d_data('timestamps', self.PPM_object.time_stamp)

            # get lineouts
            lineout_x = self.PPM_object.x_lineout
            lineout_y = self.PPM_object.y_lineout

            # gaussian fits
            fit_x = np.exp(-(self.PPM_object.x - self.PPM_object.cx) ** 2 / 2 / (self.PPM_object.wx / 2.355) ** 2)
            fit_y = np.exp(-(self.PPM_object.y - self.PPM_object.cy) ** 2 / 2 / (self.PPM_object.wy / 2.355) ** 2)

            # update dictionary
            self.data_dict['im1'] = self.PPM_object.profile
            self.data_dict['lineout_x'] = lineout_x/np.max(lineout_x)
            self.data_dict['lineout_y'] = lineout_y/np.max(lineout_y)
            self.data_dict['fit_x'] = fit_x
            self.data_dict['fit_y'] = fit_y
            self.data_dict['x'] = self.PPM_object.x
            self.data_dict['y'] = self.PPM_object.y


            # wavefront sensing
            if self.WFS_object is not None:
                wfs_data, wfs_param = self.PPM_object.retrieve_wavefront(self.WFS_object)

                self.update_1d_data('z_x', wfs_data['z2x'])
                self.update_1d_data('z_y', wfs_data['z2y'])
                self.data_dict['x_res'] = wfs_data['x_res']
                self.data_dict['y_res'] = wfs_data['y_res']
                self.data_dict['x_prime'] = wfs_data['x_prime']
                self.data_dict['y_prime'] = wfs_data['y_prime']


            # frame rate code
            now = time.time()
            dt = (now - self.lastupdate)
            if dt <= 0:
                dt = 0.000000000001
            fps2 = 1.0 / dt
            self.lastupdate = now
            self.fps = self.fps * 0.9 + fps2 * 0.1
            tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps)
            self.data_dict['tx'] = tx

            # send data
            self.sig.emit(self.data_dict)

            # keep running unless the stop button is pressed
            if self.running:
                QtCore.QTimer.singleShot(100, self._update)
                self.counter += 1
            else:
                self.PPM_object.reset_camera()

    def stop(self):
        self.running = False
        self.PPM_object.stop()


class RunRegistration(QtCore.QObject):
    sig = QtCore.pyqtSignal(dict)

    def __init__(self, yag, data_dict, imager=None):
        super(RunRegistration, self).__init__()

        # self.my_signal = QtCore.Signal()

        # self.gui = gui

        self.yag1 = yag
        self.epics_name = ''
        #if len(sys.argv) > 1:
        if imager is not None:
            self.epics_name = imager

        FOV_dict = {
            'IM2K4': 8.5,
            'IM3K4': 8.5,
            'IM4K4': 5.0,
            'IM5K4': 8.5,
            'IM6K4': 8.5,
            'IM1K1': 8.5,
            'IM2K1': 8.5,
            'IM1K2': 8.5,
            'IM2K2': 18.5,
            'IM3K2': 18.5,
            'IM4K2': 8.5,
            'IM5K2': 8.5,
            'IM6K2': 5.0,
            'IM7K2': 5.0,
            'IM1L1': 8.5,
            'IM2L1': 8.5,
            'IM3L1': 8.5,
            'IM4L1': 8.5,
            'IM1K3': 8.5,
            'IM2K3': 8.5,
            'IM3K3': 8.5,
            'IM3L0': 5.0
        }

        try:
            self.distance = FOV_dict[self.epics_name[0:5]] * 1e3
        except:
            self.distance = 8500.0

        try:
            self.gige = PCDSAreaDetector(self.epics_name, name='gige')
            self.reset_camera()
        except Exception:
            print('\nSomething wrong with camera server')
            self.gige = None

        # self.connect(self.gui, QtCore.SIGNAL('stop()'), self.stop)

        self.running = True

        self.im0 = data_dict['im0']

        self.im1 = np.copy(self.im0)

        # gui.img0.setImage(im1)

        # self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        # self.contrast = np.zeros((4,100))
        # self.iteration = np.tile(np.linspace(-99, 0, 100),(4,1))

        self.data_dict = data_dict

        self.counter = self.data_dict['counter']

        #### Start  #####################
        self._update()

    def stop(self):
        self.running = False

    def reset_camera(self):
        try:
            self.gige.cam.acquire.put(0, wait=True)
            self.gige.cam.acquire.put(1)
        except:
            print('no camera')

    def get_image(self):
        try:
            img = np.array(self.gige.image2.image, dtype='float')
            print(np.max(img))
            return img
        except:
            print('no image')
            return np.zeros((2048, 2048))

    def _update(self):

        if self.running:

            if self.epics_name != '':

                # self.im1 = np.ones((2048,2048))*255
                self.im1 = self.get_image()

            else:
                if self.counter <= 10:
                    self.im1 = interpolate.rotate(self.im0, .1 * self.counter, reshape=False, mode='nearest')
                    self.im1 = ndimage.filters.gaussian_filter(self.im1, 3 - self.counter * .3)
                else:
                    self.im1 = self.im0

            # self.gui.img0.setImage(np.flipud(self.im1).T,levels=(0,300))

            self.data_dict['contrast'] = np.roll(self.data_dict['contrast'], -1, axis=1)
            self.data_dict['rotation'] = np.roll(self.data_dict['rotation'], -1, axis=1)
            self.data_dict['iteration'] = np.roll(self.data_dict['iteration'], -1, axis=1)
            self.data_dict['iteration'][:, -1] = self.counter

            alignment_output = self.yag1.check_alignment(self.im1)

            self.data_dict['contrast'][:, -1] = alignment_output['contrast']

            translation = alignment_output['translation']

            # centering_x = 1024 + np.sum(translation[:,1])/4
            # centering_y = 1024 + np.sum(translation[:,0])/4
            # self.data_dict['centering'] = np.array([centering_x,centering_y])

            rotation1 = (translation[1, 1] - translation[3, 1]) / (translation[3, 0] - translation[1, 0])
            rotation2 = (translation[3, 0] - translation[2, 0]) / (translation[3, 1] - translation[2, 1])
            self.data_dict['rotation'][:, -1] = (rotation1 + rotation2) * 180 / np.pi / 2

            # scale
            scale = alignment_output['scale']
            # centering
            center = np.zeros((4, 2))
            # center[0,:] = translation[0,:]*scale[0] + np.array([1792,256])
            center[0, 0] = translation[0, 0] * scale[0] + 1792
            center[0, 1] = -translation[0, 1] * scale[0] + 256
            # center[1,:] = translation[1,:]*scale[1] + np.array([1792,1792])
            center[1, 0] = translation[1, 0] * scale[1] + 1792
            center[1, 1] = -translation[1, 1] * scale[1] + 1792
            # center[2,:] = translation[2,:]*scale[2] + np.array([256,256])
            center[2, 0] = translation[2, 0] * scale[2] + 256
            center[2, 1] = -translation[2, 1] * scale[2] + 256
            # center[3,:] = translation[3,:]*scale[3] + np.array([256,1792])
            center[3, 0] = translation[3, 0] * scale[3] + 256
            center[3, 1] = -translation[3, 1] * scale[3] + 1792
            self.data_dict['center'] = center
            self.data_dict['scale'] = scale

            pix_dist = np.zeros(4)
            pix_dist[0] = np.sqrt(np.sum(np.abs(center[0, :] - center[1, :]) ** 2))
            pix_dist[1] = np.sqrt(np.sum(np.abs(center[0, :] - center[2, :]) ** 2))
            pix_dist[2] = np.sqrt(np.sum(np.abs(center[1, :] - center[3, :]) ** 2))
            pix_dist[3] = np.sqrt(np.sum(np.abs(center[2, :] - center[3, :]) ** 2))

            self.data_dict['pixSize'] = self.distance / np.mean(pix_dist)

            # self.data_dict['rotation'][:,-1] = alignment_output['rotation']

            # data_dict = {}
            self.data_dict['im1'] = self.im1
            self.data_dict['shifts'] = alignment_output['shifts']
            self.data_dict['counter'] = self.counter

            now = time.time()
            dt = (now - self.lastupdate)
            if dt <= 0:
                dt = 0.000000000001
            fps2 = 1.0 / dt
            self.lastupdate = now
            self.fps = self.fps * 0.9 + fps2 * 0.1
            tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps)
            self.data_dict['tx'] = tx

            self.sig.emit(self.data_dict)

            if self.running:
                QtCore.QTimer.singleShot(100, self._update)
                self.counter += 1
            else:
                self.reset_camera()
