from epics import PV
import numpy as np
import scipy.ndimage.interpolation as interpolate
import scipy.ndimage as ndimage
import time
from pyqtgraph.Qt import QtCore
from pcdsdevices.areadetector.detectors import PCDSAreaDetector
from lcls_beamline_toolbox.xraybeamline2d import optics
import sys


class RunProcessing(QtCore.QObject):
    sig = QtCore.pyqtSignal(dict)

    def __init__(self, imager_prefix, data_dict):
        super(RunProcessing, self).__init__()

        # self.my_signal = QtCore.Signal()

        # self.gui = gui

        self.PPM_object = optics.PPM_Imager(imager_prefix, threshold=0.1)

        # self.cam_name = imager_prefix + 'CAM:'
        # self.epics_name = self.cam_name + 'IMAGE2:'
        #
        # # if len(sys.argv)>1:
        # #     self.cam_name = sys.argv[1]
        # #     self.epics_name = sys.argv[1] + 'IMAGE2:'
        #
        # self.image_pv = PV(self.epics_name + 'ArrayData')
        #
        # # get ROI info
        # xmin = PV(self.epics_name + 'ROI:MinX_RBV').get()
        # xmax = xmin + PV(self.epics_name + 'ROI:SizeX_RBV').get() - 1
        # ymin = PV(self.epics_name + 'ROI:MinY_RBV').get()
        # ymax = ymin + PV(self.epics_name + 'ROI:SizeY_RBV').get() - 1
        # # get binning
        # xbin = PV(self.epics_name + 'ROI:BinX_RBV').get()
        # ybin = PV(self.epics_name + 'ROI:BinY_RBV').get()
        # # get array size
        # self.xsize = PV(self.epics_name + 'ROI:ArraySizeX_RBV').get()
        # self.ysize = PV(self.epics_name + 'ROI:ArraySizeY_RBV').get()
        #
        # self.x1d = np.linspace(xmin, xmax - (xbin - 1), self.xsize)
        # self.x1d -= (xmax+1)/2
        # self.y1d = np.linspace(ymin, ymax - (ybin - 1), self.ysize)
        # self.y1d -= (ymax+1)/2
        # self.x, self.y = np.meshgrid(self.x1d, self.y1d)
        #
        # FOV_dict = {
        #     'IM2K4': 8.5,
        #     'IM3K4': 8.5,
        #     'IM4K4': 5.0,
        #     'IM5K4': 8.5,
        #     'IM6K4': 8.5,
        #     'IM1K1': 8.5,
        #     'IM2K1': 8.5,
        #     'IM1K2': 8.5,
        #     'IM2K2': 18.5,
        #     'IM3K2': 18.5,
        #     'IM4K2': 8.5,
        #     'IM5K2': 8.5,
        #     'IM6K2': 5.0,
        #     'IM7K2': 5.0,
        #     'IM1L1': 8.5,
        #     'IM2L1': 8.5,
        #     'IM3L1': 8.5,
        #     'IM4L1': 8.5,
        #     'IM1K3': 8.5,
        #     'IM2K3': 8.5,
        #     'IM3K3': 8.5,
        #     'IM3L0': 5.0
        # }
        #
        # try:
        #     self.distance = FOV_dict[self.epics_name[0:5]] * 1e3
        # except:
        #     self.distance = 8500.0

        # try:
        #     self.gige = PCDSAreaDetector(self.cam_name, name='gige')
        #     self.reset_camera()
        # except Exception:
        #     print('\nSomething wrong with camera server')
        #     self.gige = None

        # self.connect(self.gui, QtCore.SIGNAL('stop()'), self.stop)

        self.running = True

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

    def _update(self):

        if self.running:

            self.PPM_object.get_image()

            # self.im1 = np.ones((2048,2048))*255
            # self.im1, time_stamp = self.get_image()
            #
            # cx, cy = self.get_centroids(self.im1)

            # self.gui.img0.setImage(np.flipud(self.im1).T,levels=(0,300))

            # self.data_dict['contrast'] = np.roll(self.data_dict['contrast'],-1,axis=1)
            # self.data_dict['rotation'] = np.roll(self.data_dict['rotation'],-1,axis=1)
            # self.data_dict['iteration'] = np.roll(self.data_dict['iteration'],-1,axis=1)
            # self.data_dict['iteration'][:,-1] = self.counter

            self.data_dict['cx'] = np.roll(self.data_dict['cx'], -1)
            self.data_dict['cy'] = np.roll(self.data_dict['cy'], -1)
            self.data_dict['timestamps'] = np.roll(self.data_dict['timestamps'], -1)
            self.data_dict['cx'][-1] = self.PPM_object.cx
            self.data_dict['cy'][-1] = self.PPM_object.cy
            self.data_dict['timestamps'][-1] = self.PPM_object.time_stamp

            # alignment_output = self.yag1.check_alignment(self.im1)

            # self.data_dict['contrast'][:,-1] = alignment_output['contrast']

            # lineout_x = np.sum(self.PPM_object.im1, axis=0)
            # lineout_y = np.sum(self.im1, axis=1)
            lineout_x = self.PPM_object.x_lineout
            lineout_y = self.PPM_object.y_lineout

            self.data_dict['im1'] = self.PPM_object.profile
            self.data_dict['lineout_x'] = lineout_x/np.max(lineout_x)
            self.data_dict['lineout_y'] = lineout_y/np.max(lineout_y)
            self.data_dict['x'] = self.PPM_object.x
            self.data_dict['y'] = self.PPM_object.y
            # translation = alignment_output['translation']

            # centering_x = 1024 + np.sum(translation[:,1])/4
            # centering_y = 1024 + np.sum(translation[:,0])/4
            # self.data_dict['centering'] = np.array([centering_x,centering_y])

            # scale
            # centering
            center = np.zeros((4, 2))
            # center[0,:] = translation[0,:]*scale[0] + np.array([1792,256])

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
