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

        self.actionSave.triggered.connect(self.save_image)
        self.actionAlignment_Screen.triggered.connect(self.run_alignment_screen)

        self.plotRangeLineEdit.returnPressed.connect(self.set_time_range)

        # connect line combo box
        self.lineComboBox.currentIndexChanged.connect(self.change_line)
        # connect imager combo box
        self.imagerComboBox.currentIndexChanged.connect(self.change_imager)

        # initialize tab to basic tab
        self.tabWidget.setCurrentIndex(0)

        # connect orientations
        #self.action0.toggled.connect(self

        # font styles
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.font = QtGui.QFont()
        self.font.setPointSize(10)
        self.font.setFamily('Arial')

        # connect levels to image
        self.imageWidget.connect_levels(self.levelsWidget)
        # connect crosshairs to image
        self.imageWidget.connect_crosshairs(self.crosshairsWidget)

        self.imagerStats.connect_image(self.imageWidget)

        # wavefront retrieval
        self.wavefrontWidget.change_lineout_label('Phase (rad)')

        # connect levels to wavefront image
        self.wavefrontWidget.connect_levels(self.wavefrontLevelsWidget)
        # connect wavefront image to crosshairs
        self.wavefrontCrosshairsWidget.connect_image(self.wavefrontWidget)

        # add centroid plot
        self.centroid_plot = PPM_widgets.StripChart(self.centroidCanvas, u'Beam Centroid (\u03BCm)')
      
        labels = ['X', 'Y', 'X smoothed', 'Y smoothed']
        keys = ['x', 'y', 'x_smooth', 'y_smooth']

        self.centroid_plot.addSeries(keys, labels)

        # add FWHM plot
        self.width_plot = PPM_widgets.StripChart(self.fwhmCanvas, u'Beam FWHM (\u03BCm)')
        self.width_plot.addSeries(keys, labels)

        # add focus distance plot
        self.focus_plot = PPM_widgets.StripChart(self.focusCanvas, 'Focus position (mm)')
        self.focus_plot.addSeries(keys, labels)

        # add rms error plot
        self.rms_plot = PPM_widgets.StripChart(self.rmsErrorCanvas, 'RMS wavefront error (rad)')
        self.rms_plot.addSeries(keys, labels)

        # initialize data dictionary
        self.data_dict = {}
        self.reset_data_dict()

        # list of beamlines
        self.line_list = ['L0', 'L1', 'K0', 'K1', 'K2', 'K3', 'K4']
        # dictionary of imagers
        self.imager_dict = {
            'L0': ['IM1L0', 'IM2L0', 'IM3L0', 'IM4L0', 'HX2_shared', 'xcs_yag1', 'mec_yag0', 'xcs_yag2', 'xcs_yag3', 'xcs_yag3m', 'cxi_dg1_yag', 
                'mfx_dg1_yag', 'mec_yag1', 'xpp_gige_13'],
            'L1': ['IM1L1', 'IM2L1', 'IM3L1', 'IM4L1'],
            'K0': ['IM1K0', 'IM2K0'],
            'K1': ['IM1K1', 'IM2K1'],
            'K2': ['IM1K2', 'IM2K2', 'IM3K2', 'IM4K2', 'IM5K2', 'IM6K2', 'IM7K2'],
            'K3': ['IM1K3', 'IM2K3', 'IM3K3'],
            'K4': ['IM1K4', 'IM2K4', 'IM3K4', 'IM4K4', 'IM5K4', 'IM6K4']
        }

        # dictionary of imager PV prefixes
        self.imagerpv_dict = {
            'L0': ['IM1L0:XTES:', 'IM2L0:XTES:', 'IM3L0:PPM:', 'IM4L0:XTES:', 'XPP:GIGE:01:', 'HXX:UM6:CVV:01:', 'HXX:HXM:CVV:01:',
                'HFX:DG2:CVV:01:', 'XCS:DG3:CVV:02:', 'HFX:DG3:CVV:01:', 'CXI:DG1:P6740:', 'MFX:DG1:P6740:', 'MEC:HXM:CVV:01:', 'XPP:GIGE:13:'],
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
        # make sure this initializes properly
        self.change_imager(0)
        self.imagerpv_list = self.imagerpv_dict['L0']
        self.imagerpv = self.imagerpv_list[0]
        self.imagerComboBox.clear()
        self.imagerComboBox.addItems(self.imager_list)

        # disable wavefront checkbox by default since IM1L0 doesn't have a WFS
        self.wavefrontCheckBox.setEnabled(False)

        # initialize registration object
        self.processing = None

    def set_time_range(self):
        try:
            time_range = float(self.plotRangeLineEdit.text())
        except ValueError:
            time_range = 10.0
            self.plotRangeLineEdit.setText('10.0')

        self.centroid_plot.set_time_range(time_range)
        self.width_plot.set_time_range(time_range)

    def setup_legend(self, legend):

        legendLabelStyle = {'color': '#FFF', 'size': '10pt'}
        for item in legend.items:
           for single_item in item:
               if isinstance(single_item, pg.graphicsItems.LabelItem.LabelItem):
                   single_item.setText(single_item.text, **legendLabelStyle)

    def run_alignment_screen(self):

        cam_name = self.imagerpv

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
        self.imageGroupBox.setTitle(self.imager)
        # check if this imager has a wavefront sensor
        if self.imager in self.WFS_list:
            self.wavefrontCheckBox.setEnabled(True)
        else:
            self.wavefrontCheckBox.setEnabled(False)
        self.imagerpv = self.imagerpv_list[index]
        self.imagerControls.change_imager(self.imagerpv)
        # reset data_dict
        self.reset_data_dict()

    def reset_data_dict(self):
        N = 1024
        self.data_dict['im0'] = np.zeros((1024,1024))
        self.data_dict['cx'] = -np.ones(N)
        self.data_dict['cy'] = -np.ones(N)
        self.data_dict['wx'] = -np.ones(N)
        self.data_dict['wy'] = -np.ones(N)
        self.data_dict['cx_smooth'] = -np.ones(N)
        self.data_dict['cy_smooth'] = -np.ones(N)
        self.data_dict['wx_smooth'] = -np.ones(N)
        self.data_dict['wy_smooth'] = -np.ones(N)
        self.data_dict['timestamps'] = -np.ones(N)
        self.data_dict['counter'] = 0.
        self.data_dict['pixSize'] = 0.0
        self.data_dict['lineout_x'] = np.zeros(100)
        self.data_dict['lineout_y'] = np.zeros(100)
        self.data_dict['fit_x'] = np.zeros(100)
        self.data_dict['fit_y'] = np.zeros(100)
        self.data_dict['x'] = np.linspace(-1024, 1023, 100)
        self.data_dict['y'] = np.linspace(-1024, 1023, 100)

        # wavefront sensor data
        self.data_dict['z_x'] = -np.ones(N)
        self.data_dict['z_y'] = -np.ones(N)
        self.data_dict['rms_x'] = -np.ones(N)
        self.data_dict['rms_y'] = -np.ones(N)
        self.data_dict['x_res'] = np.zeros(100)
        self.data_dict['y_res'] = np.zeros(100)
        self.data_dict['x_prime'] = np.linspace(-1024, 1023, 100)
        self.data_dict['y_prime'] = np.linspace(-1024, 1023, 100)

    def change_state(self):
        if self.runButton.text() == 'Run':

            if self.wavefrontCheckBox.isChecked():
                wfs_name = self.WFS_dict[self.imager]
                self.processing = RunProcessing(self.imagerpv, self.data_dict, self.averageWidget, wfs_name=wfs_name, threshold=self.imagerStats.get_threshold())
            else:
                self.processing = RunProcessing(self.imagerpv, self.data_dict, self.averageWidget, threshold=self.imagerStats.get_threshold())

            width, height = self.processing.get_FOV()

            self.imageWidget.update_viewbox(width, height)

            # update crosshair sizes
            self.crosshairsWidget.update_crosshair_width()
            self.imagerStats.update_width()

            self.thread = QtCore.QThread()
            self.thread.start()

            self.processing.moveToThread(self.thread)
            self.runButton.setText('Stop')
            # disable wavefront sensor checkbox until stop is pressed
            self.wavefrontCheckBox.setEnabled(False)

            self.processing.sig.connect(self.update_plots)

            self.lineComboBox.setEnabled(False)
            self.imagerComboBox.setEnabled(False)

        elif self.runButton.text() == 'Stop':

            self.processing.stop()
            self.thread.quit()
            self.thread.wait()
            self.runButton.setText('Run')
            # enable wavefront sensor checkbox if imager is has a wavefront sensor
            if self.imager in self.WFS_list:
                self.wavefrontCheckBox.setEnabled(True)
            self.lineComboBox.setEnabled(True)
            self.imagerComboBox.setEnabled(True)

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
            self.processing.stop()
            self.thread.quit()
            self.thread.wait()

    def update_plots(self,data_dict):

        x = data_dict['x']
        y = data_dict['y']
        image_data = data_dict['im1']
        xlineout = data_dict['lineout_x']
        ylineout = data_dict['lineout_y']
        fit_x = data_dict['fit_x']
        fit_y = data_dict['fit_y']

        # wfs widget plots
        dummy_image = data_dict['imDummy']
        x_prime = data_dict['x_prime']
        y_prime = data_dict['y_prime']
        x_res = data_dict['x_res']
        y_res = data_dict['y_res']
        x_res_fit = np.zeros_like(x_res)
        y_res_fit = np.zeros_like(y_res)
        
        self.imageWidget.update_plots(image_data, x, y, xlineout, ylineout, fit_x, fit_y)

        self.wavefrontWidget.update_plots(dummy_image, x_prime, y_prime, x_res, y_res, x_res_fit, y_res_fit)

        self.data_dict = data_dict

        self.centroid_plot.update_plots(data_dict['timestamps'], x=data_dict['cx'], y=data_dict['cy'], x_smooth=data_dict['cx_smooth'], y_smooth=data_dict['cy_smooth'])

        self.width_plot.update_plots(data_dict['timestamps'], x=data_dict['wx'], y=data_dict['wy'], x_smooth=data_dict['wx_smooth'], y_smooth=data_dict['wy_smooth'])

        self.focus_plot.update_plots(data_dict['timestamps'], x=data_dict['z_x'], y=data_dict['z_y'])
        self.rms_plot.update_plots(data_dict['timestamps'], x=data_dict['rms_x'], y=data_dict['rms_y'])

        self.label.setText(data_dict['tx'])

        stats_dict = {}
        stats_dict['cx'] = data_dict['cx']
        stats_dict['cy'] = data_dict['cy']
        stats_dict['wx'] = data_dict['wx']
        stats_dict['wy'] = data_dict['wy']

        self.imagerStats.update_stats(stats_dict)
