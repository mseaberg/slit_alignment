#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
from epics import PV
import numpy as np
import imageio
import scipy.ndimage.interpolation as interpolate
import scipy.ndimage as ndimage
import sys
import time
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPen
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from pcdsdevices.areadetector.detectors import PCDSAreaDetector
from PyQt5.uic import loadUiType
import warnings

Ui_MainWindow, QMainWindow = loadUiType('image_register.ui')

class App(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__()
        self.setupUi(self)

        self.runButton.clicked.connect(self.change_state)

        self.minValue.returnPressed.connect(self.set_min)
        self.maxValue.returnPressed.connect(self.set_max)

        self.actionSave.triggered.connect(self.save_image)

        # connect line combo box
        self.lineComboBox.currentIndexChanged.connect(self.change_line)
        # connect imager combo box
        self.imagerComboBox.currentIndexChanged.connect(self.change_imager)

        # Full image
        self.view0 = self.canvas.addViewBox(row=0,col=0,rowspan=2,colspan=3)
        self.view0.setAspectLocked(True)
        self.view0.setRange(QtCore.QRectF(0,0, 2048, 2048))
        self.img0 = pg.ImageItem(border='w')
        self.view0.addItem(self.img0)

        #  contrast plot
        self.contrast_plot = self.canvas.addPlot(row=0,col=3,rowspan=1,colspan=2)

        labelStyle = {'color': '#FFF', 'font-size': '12pt'}

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setFamily('Arial')

        xaxis = self.contrast_plot.getAxis('bottom')
        xaxis.setLabel(text='Time (s)',**labelStyle)
        xaxis.tickFont = font
        xaxis.setPen(pg.mkPen('w',width=1))
        yaxis = self.contrast_plot.getAxis('left')
        yaxis.setLabel(text='X Centroid (pixels)',**labelStyle)
        yaxis.tickFont = font
        yaxis.setPen(pg.mkPen('w',width=1))

        self.contrast_plot.showGrid(x=True,y=True,alpha=.8)
        #self.contrast_plot.setYRange(0,1.5)
        self.hplot = {}
        names = ['Top Left','Top Right','Bottom Left','Bottom Right']
        colors = ['r','g','c','m']

        #legend = self.contrast_plot.addLegend()
        for i in range(1):

            #self.hplot[i] = self.contrast_plot.plot(np.linspace(-99,0,100),np.zeros(100),pen=(i,4),name=names[i])
            self.hplot[i] = self.contrast_plot.plot(np.linspace(-99,0,100), np.zeros(100),
                    pen=pg.mkPen(colors[i], width=5),name=names[i])

        #  rotation plot
        self.rotation_plot = self.canvas.addPlot(row=1,col=3,rowspan=1,colspan=2)

        labelStyle = {'color': '#FFF', 'font-size': '12pt'}

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setFamily('Arial')

        xaxis = self.rotation_plot.getAxis('bottom')
        xaxis.setLabel(text='Time (s)',**labelStyle)
        xaxis.tickFont = font
        xaxis.setPen(pg.mkPen('w',width=1))
        yaxis = self.rotation_plot.getAxis('left')
        yaxis.setLabel(text='Y Centroid (pixels)',**labelStyle)
        yaxis.tickFont = font
        yaxis.setPen(pg.mkPen('w',width=1))

        self.rotation_plot.showGrid(x=True,y=True,alpha=.8)
        #self.contrast_plot.setYRange(0,1.5)
        self.rplot = {}
        names = ['Top Left','Top Right','Bottom Left','Bottom Right']
        colors = ['r','g','c','m']

        legend = self.contrast_plot.addLegend()
        for i in range(1):

            #self.hplot[i] = self.contrast_plot.plot(np.linspace(-99,0,100),np.zeros(100),pen=(i,4),name=names[i])
            self.rplot[i] = self.rotation_plot.plot(np.linspace(-99,0,100), np.zeros(100),
                    pen=pg.mkPen(colors[i], width=5),name=names[i])

        legendLabelStyle = {'color': '#FFF', 'size': '10pt'}
        for item in legend.items:
            for single_item in item:
                if isinstance(single_item, pg.graphicsItems.LabelItem.LabelItem):
                    single_item.setText(single_item.text, **legendLabelStyle)

        # the image to be transformed
        #im1 = np.array(imageio.imread("test_pattern.png")[32:2650, 32:2650, 3],dtype='float')
        im1 = np.array(imageio.imread("PPM_alignment/IM1K2.png"),dtype='float')
        im1 = 255 - im1

        N, M = np.shape(im1)
        scale = 2048.0 / N

        self.im0 = interpolate.zoom(im1, scale)


        # initialize data dictionary
        self.data_dict = {}
        self.reset_data_dict()


        # list of beamlines
        self.line_list = ['L0', 'L1', 'K0', 'K1', 'K2', 'K3', 'K4']
        # dictionary of imagers
        self.imager_dict = {
            'L0': ['IM1L0', 'IM2L0', 'IM3L0', 'IM4L0'],
            'L1': ['IM1L1', 'IM2L1', 'IM3L1', 'IM4L1'],
            'K0': ['IM1K0', 'IM2K0'],
            'K1': ['IM1K1', 'IM2K1'],
            'K2': ['IM1K2', 'IM2K2', 'IM3K2', 'IM4K2', 'IM5K2', 'IM6K2', 'IM7K2'],
            'K3': ['IM1K3', 'IM2K3', 'IM3K3'],
            'K4': ['IM1K4', 'IM2K4', 'IM3K4', 'IM4K4', 'IM5K4', 'IM5K4']
        }

        # dictionary of imager PV prefixes
        self.imagerpv_dict = {
            'L0': ['IM1L0:XTES:', 'IM2L0:XTES:', 'IM3L0:PPM:', 'IM4L0:XTES:'],
            'L1': ['IM1L1:PPM:', 'IM2L1:PPM:', 'IM3L1:PPM:', 'IM4L1:PPM:'],
            'K0': ['IM1K0:XTES:', 'IM2K0:XTES:'],
            'K1': ['IM1K1:PPM:', 'IM2K1:PPM:'],
            'K2': ['IM1K2:PPM:', 'IM2K2:PPM:', 'IM3K2:PPM:', 'IM4K2:PPM:', 'IM5K2:PPM:',
                   'IM6K2:PPM:', 'IM7K2:PPM:'],
            'K3': ['IM1K3:PPM:', 'IM2K3:PPM:', 'IM3K3:PPM:'],
            'K4': ['IM1K4:PPM:', 'IM2K4:PPM:', 'IM3K4:PPM:', 'IM4K4:PPM:', 'IM5K4:PPM:',
                   'IM5K4:PPM:']
        }
        # initialize line combo box
        self.line = 'L0'
        self.lineComboBox.addItems(self.line_list)


        self.imager_list = self.imager_dict['L0']
        self.imager = self.imager_list[0]
        self.imagerpv_list = self.imagerpv_dict['L0']
        self.imagerpv = self.imagerpv_list[0]
        self.imagerComboBox.clear()
        self.imagerComboBox.addItems(self.imager_list)

        #self.data_dict['centering'] = np.zeros(2)
        self.set_min()
        self.set_max()

    def change_line(self, index):
        # update line
        self.line = self.line_list[index]
        self.imager_list = self.imager_dict[self.line]
        self.imagerpv_list = self.imagerpv_dict[self.line]
        self.imagerComboBox.clear()
        self.imagerComboBox.addItems(self.imager_list)
        self.change_imager(0)

    def change_imager(self, index):
        # update imager
        self.imager = self.imager_list[index]
        self.imagerpv = self.imagerpv_list[index]
        # reset data_dict
        self.reset_data_dict()

    def reset_data_dict(self):
        self.data_dict['im0'] = self.im0
        self.data_dict['contrast'] = np.zeros((4, 100))
        self.data_dict['rotation'] = np.zeros((4, 100))
        self.data_dict['cx'] = -np.ones(100)
        self.data_dict['cy'] = -np.ones(100)
        self.data_dict['timestamps'] = -np.ones(100)
        self.data_dict['iteration'] = np.tile(np.linspace(-99, 0, 100), (4, 1))
        self.data_dict['counter'] = 0.
        self.data_dict['center'] = np.zeros((4, 2))
        self.data_dict['scale'] = np.zeros(4)
        self.data_dict['pixSize'] = 0.0


    def change_state(self):
        if self.runButton.text() == 'Run':

            self.registration = RunProcessing(self.imagerpv, self.data_dict)
            self.thread = QtCore.QThread()
            self.thread.start()

            self.registration.moveToThread(self.thread)
            self.runButton.setText('Stop')

            self.registration.sig.connect(self.update_plots)

        elif self.runButton.text() == 'Stop':

            self.registration.stop()
            self.thread.quit()
            self.thread.wait()
            self.runButton.setText('Run')

    def set_min(self):
        self.minimum = float(self.minValue.text())

    def set_max(self):
        self.maximum = float(self.maxValue.text())

    @staticmethod
    def normalize_image(image):
        image -= np.min(image)
        image *= 255./float(np.max(image))
        image = np.array(image,dtype='uint8')
        return image

    def save_image(self):
        formats = 'Portable Network Graphic (*.png)'
        filename = QtGui.QFileDialog.getSaveFileName(self, 
                'Save Image','untitled.png',formats)
        #name = str(name).strip()
        if not filename[0] == '':
            im = App.normalize_image(self.data_dict['im1'])
            filename = App.get_filename(filename)
            #imageio.imwrite(filename,self.data_dict['im1'])
            imageio.imwrite(filename,im)
        print(filename)
        #im = Image.fromarray(self.data_dict['im1'])
        #im.save(name)
     

    @staticmethod
    def get_filename(name):
        path = name[0]
        extension = name[1].split('*')[1][0:4]
        if path[-4:] != extension:
            path = path + extension
        return path


    def closeEvent(self, event):
        if self.runButton.text() == 'Stop':
            self.registration.stop()
            self.thread.quit()
            self.thread.wait()

    def update_plots(self,data_dict):

        if self.checkBox.isChecked():
            self.minimum = np.min(data_dict['im1'])
            self.maximum = np.max(data_dict['im1'])
            self.minValue.setText('%d' % self.minimum)
            self.maxValue.setText('%d' % self.maximum)
            


        self.data_dict = data_dict
        self.img0.setImage(np.flipud(data_dict['im1']).T,
                levels=(self.minimum, self.maximum))

        N, M = np.shape(data_dict['im1'])

        self.view0.setRange(QtCore.QRectF(0,0, M, N))

        now = datetime.now()
        now_stamp = datetime.timestamp(now)

        timestamp = data_dict['timestamps'] - now_stamp
        cx = data_dict['cx']
        cy = data_dict['cy']

        mask = cx>0
        cx = cx[mask]
        cy = cy[mask]
        timestamp = timestamp[mask]
        
        cx_range = np.max(cx)-np.min(cx)
        cy_range = np.max(cy)-np.min(cy)

        self.hplot[0].setData(timestamp, cx)
        self.contrast_plot.setXRange(-10, 0)
        #self.contrast_plot.setYRange(np.mean(cx)-5*cx_range, np.mean(cx)+5*cx_range)
        self.rplot[0].setData(timestamp, cy)
        self.rotation_plot.setXRange(-10, 0)
        #self.rotation_plot.setYRange(np.mean(cy)-5*cy_range, np.mean(cy)+5*cy_range)

        #self.circ0.setRect(full_center[1]-25,full_center[0]-25,50,50)

        self.label.setText(data_dict['tx'])


class RunProcessing(QtCore.QObject):

    sig = QtCore.pyqtSignal(dict)

    def __init__(self, imager_prefix, data_dict):
        super(RunProcessing, self).__init__()

        #self.my_signal = QtCore.Signal()

        #self.gui = gui

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
        xsize = PV(self.epics_name + 'ROI:ArraySizeX_RBV').get()
        ysize = PV(self.epics_name + 'ROI:ArraySizeY_RBV').get()

        x = np.linspace(xmin, xmax-(xbin-1), xsize)
        y = np.linspace(ymin, ymax-(ybin-1), ysize)
        self.x, self.y = np.meshgrid(x,y)

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
            self.distance = FOV_dict[self.epics_name[0:5]]*1e3
        except:
            self.distance = 8500.0

        try:
            self.gige = PCDSAreaDetector(self.cam_name, name='gige')
            self.reset_camera()
        except Exception:
            print('\nSomething wrong with camera server') 
            self.gige = None

        
        #self.connect(self.gui, QtCore.SIGNAL('stop()'), self.stop)

        self.running = True

        self.im0 = data_dict['im0']

        self.im1 = np.copy(self.im0)

        #gui.img0.setImage(im1)

        #self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        #self.contrast = np.zeros((4,100))
        #self.iteration = np.tile(np.linspace(-99, 0, 100),(4,1))

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
            #image_data = self.gige.image2.get()
            image_data = self.image_pv.get_with_metadata()
            img = np.reshape(image_data['value'],(512,512)).astype(float)
            time_stamp = image_data['timestamp']
            #time_stamp = image_data.time_stamp
            #img = np.array(image_data.shaped_image,dtype='float')
            #img = np.array(self.gige.image2.image,dtype='float')
            return img, time_stamp
        except:
            print('no image')
            return np.zeros((2048,2048))

    def threshold_image(self, img):
        # threshold image
        thresh = np.max(img)*.2
        img -= thresh
        img[img<0] = 0

        return img

    def get_centroids(self, img):

        # get thresholded image
        thresh = self.threshold_image(img)
        
        cx = np.sum(thresh*self.x)/np.sum(thresh)
        cy = np.sum(thresh*self.y)/np.sum(thresh)

        return cx, cy


    def _update(self):

        if self.running:

            if self.epics_name != '':

                #self.im1 = np.ones((2048,2048))*255
                self.im1, time_stamp = self.get_image()

            else:
                if self.counter <= 10:
                    time_stamp = self.counter
                    self.im1 = interpolate.rotate(self.im0,.1*self.counter,reshape=False,mode='nearest')
                    self.im1 = ndimage.filters.gaussian_filter(self.im1,3-self.counter*.3)
                else:
                    self.im1 = self.im0
                    time_stamp = self.counter

            cx, cy = self.get_centroids(self.im1)

            #self.gui.img0.setImage(np.flipud(self.im1).T,levels=(0,300))

            #self.data_dict['contrast'] = np.roll(self.data_dict['contrast'],-1,axis=1)
            #self.data_dict['rotation'] = np.roll(self.data_dict['rotation'],-1,axis=1)
            #self.data_dict['iteration'] = np.roll(self.data_dict['iteration'],-1,axis=1)
            #self.data_dict['iteration'][:,-1] = self.counter

            self.data_dict['cx'] = np.roll(self.data_dict['cx'],-1)
            self.data_dict['cy'] = np.roll(self.data_dict['cy'],-1)
            self.data_dict['timestamps'] = np.roll(self.data_dict['timestamps'],-1)
            self.data_dict['cx'][-1] = cx
            self.data_dict['cy'][-1] = cy
            self.data_dict['timestamps'][-1] = time_stamp
            

            #alignment_output = self.yag1.check_alignment(self.im1)

            #self.data_dict['contrast'][:,-1] = alignment_output['contrast']
           
            self.data_dict['im1'] = self.im1
            #translation = alignment_output['translation']

            #centering_x = 1024 + np.sum(translation[:,1])/4
            #centering_y = 1024 + np.sum(translation[:,0])/4
            #self.data_dict['centering'] = np.array([centering_x,centering_y])

            # scale
            # centering
            center = np.zeros((4,2))
            #center[0,:] = translation[0,:]*scale[0] + np.array([1792,256])

            now = time.time()
            dt = (now-self.lastupdate)
            if dt <= 0:
                dt = 0.000000000001
            fps2 = 1.0 / dt
            self.lastupdate = now
            self.fps = self.fps * 0.9 + fps2 * 0.1
            tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
            self.data_dict['tx'] = tx

            self.sig.emit(self.data_dict)

            if self.running:
                QtCore.QTimer.singleShot(100, self._update)
                self.counter += 1
            else:
                self.reset_camera()


if __name__ == '__main__':

    warnings.filterwarnings('ignore', '.*output shape of zoom.*')
    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())
