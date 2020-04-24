from epics import PV
import numpy as np
import scipy.ndimage.interpolation as interpolate
import scipy.ndimage as ndimage
import time
from pyqtgraph.Qt import QtCore
from pcdsdevices.areadetector.detectors import PCDSAreaDetector


class RunProcessing(QtCore.QObject):
    sig = QtCore.pyqtSignal(dict)

    def __init__(self, imager_prefix, data_dict):
        super(RunProcessing, self).__init__()

        # self.my_signal = QtCore.Signal()

        # self.gui = gui

        self.cam_name = imager_prefix + 'CAM:'
        self.epics_name = self.cam_name + 'IMAGE2:'

        # if len(sys.argv)>1:
        #     self.cam_name = sys.argv[1]
        #     self.epics_name = sys.argv[1] + 'IMAGE2:'

        self.image_pv = PV(self.epics_name + 'ArrayData')

        # get ROI info
        xmin = PV(self.epics_name + 'ROI:MinX_RBV').get()
        xmax = xmin + PV(self.epics_name + 'ROI:SizeX_RBV').get() - 1
        ymin = PV(self.epics_name + 'ROI:MinY_RBV').get()
        ymax = ymin + PV(self.epics_name + 'ROI:SizeY_RBV').get() - 1
        # get binning
        xbin = PV(self.epics_name + 'ROI:BinX_RBV').get()
        ybin = PV(self.epics_name + 'ROI:BinY_RBV').get()
        # get array size
        self.xsize = PV(self.epics_name + 'ROI:ArraySizeX_RBV').get()
        self.ysize = PV(self.epics_name + 'ROI:ArraySizeY_RBV').get()

        x = np.linspace(xmin, xmax - (xbin - 1), self.xsize)
        y = np.linspace(ymin, ymax - (ybin - 1), self.ysize)
        self.x, self.y = np.meshgrid(x, y)

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
            self.gige = PCDSAreaDetector(self.cam_name, name='gige')
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
        try:
            self.gige.cam.acquire.put(0, wait=True)
        except AttributeError:
            pass

    def reset_camera(self):
        try:
            self.gige.cam.acquire.put(0, wait=True)
            self.gige.cam.acquire.put(1)
        except:
            print('no camera')

    def get_image(self):
        try:
            # image_data = self.gige.image2.get()
            image_data = self.image_pv.get_with_metadata()
            img = np.reshape(image_data['value'], (self.ysize, self.xsize)).astype(float)
            time_stamp = image_data['timestamp']
            # time_stamp = image_data.time_stamp
            # img = np.array(image_data.shaped_image,dtype='float')
            # img = np.array(self.gige.image2.image,dtype='float')
            return img, time_stamp
        except:
            print('no image')
            return np.zeros((2048, 2048))

    def threshold_image(self, img):
        # threshold image
        thresh = np.max(img) * .2
        img -= thresh
        img[img < 0] = 0

        return img

    def get_centroids(self, img):

        # get thresholded image
        thresh = self.threshold_image(img)

        cx = np.sum(thresh * self.x) / np.sum(thresh)
        cy = np.sum(thresh * self.y) / np.sum(thresh)

        return cx, cy

    def _update(self):

        if self.running:

            if self.epics_name != '':

                # self.im1 = np.ones((2048,2048))*255
                self.im1, time_stamp = self.get_image()

            else:
                if self.counter <= 10:
                    time_stamp = self.counter
                    self.im1 = interpolate.rotate(self.im0, .1 * self.counter, reshape=False, mode='nearest')
                    self.im1 = ndimage.filters.gaussian_filter(self.im1, 3 - self.counter * .3)
                else:
                    self.im1 = self.im0
                    time_stamp = self.counter

            cx, cy = self.get_centroids(self.im1)

            # self.gui.img0.setImage(np.flipud(self.im1).T,levels=(0,300))

            # self.data_dict['contrast'] = np.roll(self.data_dict['contrast'],-1,axis=1)
            # self.data_dict['rotation'] = np.roll(self.data_dict['rotation'],-1,axis=1)
            # self.data_dict['iteration'] = np.roll(self.data_dict['iteration'],-1,axis=1)
            # self.data_dict['iteration'][:,-1] = self.counter

            self.data_dict['cx'] = np.roll(self.data_dict['cx'], -1)
            self.data_dict['cy'] = np.roll(self.data_dict['cy'], -1)
            self.data_dict['timestamps'] = np.roll(self.data_dict['timestamps'], -1)
            self.data_dict['cx'][-1] = cx
            self.data_dict['cy'][-1] = cy
            self.data_dict['timestamps'][-1] = time_stamp

            # alignment_output = self.yag1.check_alignment(self.im1)

            # self.data_dict['contrast'][:,-1] = alignment_output['contrast']

            self.data_dict['im1'] = self.im1
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