#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
from epics import PV
import numpy as np
import imageio
import scipy.ndimage.interpolation as interpolate
import sys
import time
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from PyQt5.uic import loadUiType
from PyQt5.QtCore import Qt
import warnings
from processing_module import RunProcessing
from Image_registration_epics import App
import PPM_widgets

Ui_MainWindow, QMainWindow = loadUiType('PPM_screen.ui')

class PPM_Interface(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(PPM_Interface, self).__init__()
        self.setupUi(self)

        self.runButton.clicked.connect(self.change_state)

        self.minValue.returnPressed.connect(self.set_min)
        self.maxValue.returnPressed.connect(self.set_max)

        self.actionSave.triggered.connect(self.save_image)
        self.actionAlignment_Screen.triggered.connect(self.run_alignment_screen)

        # connect line combo box
        self.lineComboBox.currentIndexChanged.connect(self.change_line)
        # connect imager combo box
        self.imagerComboBox.currentIndexChanged.connect(self.change_imager)

        # initialize tab to basic tab
        self.tabWidget.setCurrentIndex(0)

        # font styles
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.font = QtGui.QFont()
        self.font.setPointSize(10)
        self.font.setFamily('Arial')

        # Full image
        self.raw_image = PPM_widgets.LineoutImage(self.imageGroupBox)
      
        # crosshairs
        self.crosshairObject = PPM_widgets.CrosshairWidget(self.crosshairGroupBox, self.raw_image)

        self.red_crosshair_widget, self.blue_crosshair_widget = self.crosshairObject.get_crosshairs()
        

        # connect to mouse click on image
        self.raw_image.rect.scene().sigMouseClicked.connect(self.mouseClicked)

        self.redCrosshair = self.crosshairObject.redButton
        self.blueCrosshair = self.crosshairObject.blueButton
        self.red_x = self.crosshairObject.red_x
        self.red_y = self.crosshairObject.red_y
        self.blue_x = self.crosshairObject.blue_x
        self.blue_y = self.crosshairObject.blue_y

        # connect crosshair selection
        self.redCrosshair.toggled.connect(self.red_crosshair_toggled)
        self.blueCrosshair.toggled.connect(self.blue_crosshair_toggled)
        self.red_x.returnPressed.connect(self.red_crosshair_widget.update_position)
        self.red_y.returnPressed.connect(self.red_crosshair_widget.update_position)
        self.blue_x.returnPressed.connect(self.blue_crosshair_widget.update_position)
        self.blue_y.returnPressed.connect(self.blue_crosshair_widget.update_position)


        self.horizontalPlot = self.raw_image.horizontalPlot
        self.horizontalLineout = self.raw_image.horizontalLineout
        self.horizontalFit = self.raw_image.horizontalFit
        self.verticalPlot = self.raw_image.verticalPlot
        self.verticalLineout = self.raw_image.verticalLineout
        self.verticalFit = self.raw_image.verticalFit

        #  centroid plot
        self.centroid_plot = self.plotCanvas.addPlot(row=0,col=0,rowspan=1,colspan=2)

        xaxis = self.centroid_plot.getAxis('bottom')
        xaxis.setLabel(text='Time (s)',**self.labelStyle)
        xaxis.tickFont = self.font
        xaxis.setPen(pg.mkPen('w',width=1))
        yaxis = self.centroid_plot.getAxis('left')
        yaxis.setLabel(text=u'Beam Centroid (\u03BCm)',**self.labelStyle)
        yaxis.tickFont = self.font
        yaxis.setPen(pg.mkPen('w',width=1))

        self.centroid_plot.showGrid(x=True,y=True,alpha=.8)
        self.centroid_lines = {}
        names = ['X','Y','X smoothed','Y smoothed']
        colors = ['r','c','m','g']

        legend = self.centroid_plot.addLegend()
        for i in range(4):

            self.centroid_lines[i] = self.centroid_plot.plot(np.linspace(-99,0,100), np.zeros(100),
                    pen=pg.mkPen(colors[i], width=5),name=names[i])

        self.setup_legend(legend)

        # plot FWHM widths
        self.width_plot = self.plotCanvas.addPlot(row=1,col=0,rowspan=1,colspan=2)

        labelStyle = {'color': '#FFF', 'font-size': '12pt'}

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setFamily('Arial')

        xaxis = self.width_plot.getAxis('bottom')
        xaxis.setLabel(text='Time (s)',**labelStyle)
        xaxis.tickFont = font
        xaxis.setPen(pg.mkPen('w',width=1))
        yaxis = self.width_plot.getAxis('left')
        yaxis.setLabel(text=u'Beam FWHM (\u03BCm)',**labelStyle)
        yaxis.tickFont = font
        yaxis.setPen(pg.mkPen('w',width=1))

        self.width_plot.showGrid(x=True,y=True,alpha=.8)
        #self.contrast_plot.setYRange(0,1.5)
        self.width_lines = {}
        names = ['X', 'Y', 'X smoothed', 'Y smoothed']
        colors = ['r', 'c','m','g']

        legend = self.width_plot.addLegend()
        for i in range(4):

            self.width_lines[i] = self.width_plot.plot(np.linspace(-99,0,100), np.zeros(100),
                    pen=pg.mkPen(colors[i], width=5),name=names[i])

        self.setup_legend(legend)

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
            'K4': ['IM1K4', 'IM2K4', 'IM3K4', 'IM4K4', 'IM5K4', 'IM6K4']
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
                   'IM6K4:PPM:']
        }

        # list of imagers with a wavefront sensor
        self.WFS_list = ['IM2K0', 'IM2L0', 'IM5K4', 'IM6K4', 'IM6K2', 'IM3K3', 'IM4L1']

        # dictionary of wavefront sensor corresponding to imager
        self.WFS_dict = {
            'IM2K0': 'PF1K0',
            'IM2L0': 'PF1L0',
            'IM5K4': 'PF1K4',
            'IM6K4': 'PF2K4',
            'IM6K2': 'PF1K2',
            'IM3K3': 'PF1K3',
            'IM4L1': 'PF1L1'
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

        # disable wavefront checkbox by default since IM1L0 doesn't have a WFS
        self.wavefrontCheckBox.setEnabled(False)

        self.set_min()
        self.set_max()

        # initialize registration object
        self.registration = None

        # initialize crosshair selection (None selected)
        self.current_crosshair = None

    def red_crosshair_toggled(self, evt):
        if evt:
            if self.blueCrosshair.isChecked():
                self.blueCrosshair.toggle()
            self.current_crosshair = self.red_crosshair_widget
        else:
            self.current_crosshair = None

    def blue_crosshair_toggled(self, evt):
        if evt:
            if self.redCrosshair.isChecked():
                self.redCrosshair.toggle()
            self.current_crosshair = self.blue_crosshair_widget
        else:
            self.current_crosshair = None

    def mouseClicked(self, evt):
        # translate scene coordinates to viewbox coordinates
        coords = self.raw_image.view.mapSceneToView(evt.scenePos())

        if self.current_crosshair is not None:
            self.current_crosshair.xLineEdit.setText('%.1f' % coords.x())
            self.current_crosshair.yLineEdit.setText('%.1f' % coords.y())
            self.current_crosshair.update_position()

    def setup_legend(self, legend):

        legendLabelStyle = {'color': '#FFF', 'size': '10pt'}
        for item in legend.items:
           for single_item in item:
               if isinstance(single_item, pg.graphicsItems.LabelItem.LabelItem):
                   single_item.setText(single_item.text, **legendLabelStyle)

    def run_alignment_screen(self):

        cam_name = self.imagerpv + 'CAM:'

        alignment_app = App(parent=self, imager=cam_name)
        alignment_app.show()

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
        # check if this imager has a wavefront sensor
        if self.imager in self.WFS_list:
            self.wavefrontCheckBox.setEnabled(True)
        else:
            self.wavefrontCheckBox.setEnabled(False)
        self.imagerpv = self.imagerpv_list[index]
        # reset data_dict
        self.reset_data_dict()

    def reset_data_dict(self):
        self.data_dict['im0'] = np.zeros((1024,1024))
        self.data_dict['contrast'] = np.zeros((4, 100))
        self.data_dict['rotation'] = np.zeros((4, 100))
        self.data_dict['cx'] = -np.ones(100)
        self.data_dict['cy'] = -np.ones(100)
        self.data_dict['wx'] = -np.ones(100)
        self.data_dict['wy'] = -np.ones(100)
        self.data_dict['cx_smooth'] = -np.ones(100)
        self.data_dict['cy_smooth'] = -np.ones(100)
        self.data_dict['wx_smooth'] = -np.ones(100)
        self.data_dict['wy_smooth'] = -np.ones(100)
        self.data_dict['timestamps'] = -np.ones(100)
        self.data_dict['iteration'] = np.tile(np.linspace(-99, 0, 100), (4, 1))
        self.data_dict['counter'] = 0.
        self.data_dict['center'] = np.zeros((4, 2))
        self.data_dict['scale'] = np.zeros(4)
        self.data_dict['pixSize'] = 0.0
        self.data_dict['lineout_x'] = np.zeros(100)
        self.data_dict['lineout_y'] = np.zeros(100)
        self.data_dict['fit_x'] = np.zeros(100)
        self.data_dict['fit_y'] = np.zeros(100)
        self.data_dict['x'] = np.linspace(-1024, 1023, 100)
        self.data_dict['y'] = np.linspace(-1024, 1023, 100)

        # wavefront sensor data
        self.data_dict['z_x'] = -np.ones(100)
        self.data_dict['z_y'] = -np.ones(100)
        self.data_dict['x_res'] = np.zeros(100)
        self.data_dict['y_res'] = np.zeros(100)
        self.data_dict['x_prime'] = np.linspace(-1024, 1023, 100)
        self.data_dict['y_prime'] = np.linspace(-1024, 1023, 100)

    def change_state(self):
        if self.runButton.text() == 'Run':

            if self.wavefrontCheckBox.isChecked():
                wfs_name = self.WFS_dict[self.imager]
                self.registration = RunProcessing(self.imagerpv, self.data_dict, wfs_name=wfs_name)
            else:
                self.registration = RunProcessing(self.imagerpv, self.data_dict)

            width, height = self.registration.get_FOV()

            self.raw_image.update_viewbox(width, height)

            # update crosshair sizes
            self.red_crosshair_widget.update_width()
            self.blue_crosshair_widget.update_width()

            self.thread = QtCore.QThread()
            self.thread.start()

            self.registration.moveToThread(self.thread)
            self.runButton.setText('Stop')
            # disable wavefront sensor checkbox until stop is pressed
            self.wavefrontCheckBox.setEnabled(False)

            self.registration.sig.connect(self.update_plots)

            self.lineComboBox.setEnabled(False)
            self.imagerComboBox.setEnabled(False)

        elif self.runButton.text() == 'Stop':

            self.registration.stop()
            self.thread.quit()
            self.thread.wait()
            self.runButton.setText('Run')
            # enable wavefront sensor checkbox if imager is has a wavefront sensor
            if self.imager in self.WFS_list:
                self.wavefrontCheckBox.setEnabled(True)
            self.lineComboBox.setEnabled(True)
            self.imagerComboBox.setEnabled(True)

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
            
        x = data_dict['x']
        y = data_dict['y']
        x_width = np.max(x) - np.min(x)
        y_width = np.max(y) - np.min(y)

        self.data_dict = data_dict
        self.raw_image.img.setImage(np.flipud(data_dict['im1']).T,
                levels=(self.minimum, self.maximum))
        self.raw_image.img.setRect(QtCore.QRectF(np.min(x),np.min(y),x_width, y_width))

        N, M = np.shape(data_dict['im1'])

        # self.view0.setRange(QtCore.QRectF(0,0, M, N))

        now = datetime.now()
        now_stamp = datetime.timestamp(now)

        timestamp = data_dict['timestamps'] - now_stamp
        cx = data_dict['cx']
        cy = data_dict['cy']
        cxs = data_dict['cx_smooth']
        cys = data_dict['cy_smooth']
        wx = data_dict['wx']
        wy = data_dict['wy']
        wxs = data_dict['wx_smooth']
        wys = data_dict['wy_smooth']

        mask = data_dict['timestamps']>0
        cx = cx[mask]
        cy = cy[mask]
        cxs = cxs[mask]
        cys = cys[mask]
        wx = wx[mask]
        wy = wy[mask]
        wxs = wxs[mask]
        wys = wys[mask]
        timestamp = timestamp[mask]
        
        cx_range = np.max(cx)-np.min(cx)
        cy_range = np.max(cy)-np.min(cy)

        # lineouts
        self.centroid_lines[0].setData(timestamp, cx)
        self.centroid_lines[1].setData(timestamp, cy)
        self.centroid_lines[2].setData(timestamp, cxs)
        self.centroid_lines[3].setData(timestamp, cys)
        self.centroid_plot.setXRange(-10, 0)
        #self.contrast_plot.setYRange(np.mean(cx)-5*cx_range, np.mean(cx)+5*cx_range)
        self.width_lines[0].setData(timestamp, wx)
        self.width_lines[1].setData(timestamp, wy)
        self.width_lines[2].setData(timestamp, wxs)
        self.width_lines[3].setData(timestamp, wys)
        self.width_plot.setXRange(-10, 0)
        #self.rotation_plot.setYRange(np.mean(cy)-5*cy_range, np.mean(cy)+5*cy_range)

        self.horizontalLineout.setData(data_dict['x'], data_dict['lineout_x'])
        self.horizontalFit.setData(data_dict['x'], data_dict['fit_x'])
        self.verticalLineout.setData(data_dict['lineout_y'], data_dict['y'])
        self.verticalFit.setData(data_dict['fit_y'], data_dict['y'])

        #self.circ0.setRect(full_center[1]-25,full_center[0]-25,50,50)

        self.label.setText(data_dict['tx'])
