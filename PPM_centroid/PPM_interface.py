#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
from epics import PV
import numpy as np
import imageio
import scipy.ndimage.interpolation as interpolate
import sys
import time
import json
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

        # button to start calculations
        self.runButton.clicked.connect(self.change_state)
        # method to save an image. Maybe replace and/or supplement this with image "recording" in the future
        self.actionSave.triggered.connect(self.save_image)
        # open alignment screen for calculating center and pixel size
        self.actionAlignment_Screen.triggered.connect(self.run_alignment_screen)

        # adjustment for amount of time to show on plots (this should be cleaned up later)
        self.plotRangeLineEdit.returnPressed.connect(self.set_time_range)
        self.wfsPlotRangeLineEdit.returnPressed.connect(self.set_time_range)

        # connect line combo box
        self.lineComboBox.currentIndexChanged.connect(self.change_line)
        # connect imager combo box
        self.imagerComboBox.currentIndexChanged.connect(self.change_imager)

        # list of QAction objects for controlling the image orientation
        self.orientation_actions = [self.action0, self.action90, self.action180, self.action270, 
                self.action0_flip, self.action90_flip, self.action180_flip, self.action270_flip]

        # dictionary of QAction objects. Probably this could replace the above list eventually, but it works so won't
        # break it for now...
        self.orientation_dict = {
                'action0': self.action0,
                'action90': self.action90,
                'action180': self.action180,
                'action270': self.action270,
                'action0_flip': self.action0_flip,
                'action90_flip': self.action90_flip,
                'action180_flip': self.action180_flip,
                'action270_flip': self.action270_flip
                }

        # connect orientation actions
        for action in self.orientation_actions:
            action.triggered.connect(self.change_orientation)

        # connect method to save the current orientation
        self.actionSave_orientation.triggered.connect(self.save_orientation)

        # set orientation
        self.orientation = 'action0'

        # initialize tab to basic tab
        self.tabWidget.setCurrentIndex(0)

        # connect levels to image
        self.imageWidget.connect_levels(self.levelsWidget)
        # connect crosshairs to image
        self.imageWidget.connect_crosshairs(self.crosshairsWidget)

        # connect stats to image. This is for displaying the circle on the image centered on the
        # beam with diameter of 2*FWHM
        self.imagerStats.connect_image(self.imageWidget)

        # wavefront retrieval
        self.wavefrontWidget.change_lineout_label('Phase (rad)')

        # connect levels to wavefront image
        self.wavefrontWidget.connect_levels(self.wavefrontLevelsWidget)
        # connect wavefront image to crosshairs
        self.wavefrontWidget.connect_crosshairs(self.wavefrontCrosshairsWidget)

        # add centroid plot
        self.centroid_plot = PPM_widgets.StripChart(self.centroidCanvas, u'Beam Centroid (\u03BCm)')

        # labels and keys for plots
        labels = ['X', 'Y', 'X smoothed', 'Y smoothed']
        keys = ['x', 'y', 'x_smooth', 'y_smooth']

        # plot for centroids
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

        # make a list of all the plots
        self.all_plots = [self.centroid_plot, self.width_plot, self.focus_plot, self.rms_plot]

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
            'K4': ['IM1K4:XTES:', 'IM2K4:PPM:', 'IM3K4:PPM:', 'IM4K4:PPM:', 'IM5K4:PPM:',
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

        # initialize imager list and imager
        self.imager_list = self.imager_dict['L0']
        self.imager = self.imager_list[0]

        # set wavefront sensor attribute
        self.wfs_name = None

        # make sure this initializes properly
        self.imagerpv_list = self.imagerpv_dict['L0']
        self.imagerpv = self.imagerpv_list[0]
        self.imagerComboBox.clear()
        self.imagerComboBox.addItems(self.imager_list)

        # more initialization...
        self.change_line(0)

        # disable wavefront checkbox by default since IM1L0 doesn't have a WFS
        self.wavefrontCheckBox.setEnabled(False)

        # initialize registration object
        self.processing = None

    def uncheck_all(self):
        """
        Method to uncheck all orientation options.
        """
        for action in self.orientation_actions:
            action.setChecked(False)

    def change_orientation(self):
        """
        Method that is called when an orientation menu item is selected. This causes a change to the orientation
        of the displayed image.
        """
        menu_item = self.sender()

        self.uncheck_all()
        menu_item.setChecked(True)

        self.orientation = menu_item.objectName()

        # check if running, if so send orientation information
        if self.processing is not None:
            self.processing.set_orientation(self.orientation)

    def load_orientation(self):
        """
        Method to load the previously saved orientation. Defaults to no rotation if there hasn't been anything saved.
        """
        try:
            # read the imagers.db file
            with open('/reg/neh/home/seaberg/Commissioning_Tools/PPM_centroid/imagers.db') as json_file:
                data = json.load(json_file)
            # set orientation from the file
            self.orientation = data[self.imager]['orientation']
            print('using orientation %s' % self.orientation)
        except json.decoder.JSONDecodeError:
            # catch the exception that the file doesn't exist
            self.orientation = 'action0'
        except KeyError:
            # catch the exception that the orientation hasn't been saved for this imager
            print('orientation not set, using 0.')
            self.orientation = 'action0'

        # set appropriate checkbox and uncheck any other boxes
        self.uncheck_all()
        self.orientation_dict[self.orientation].setChecked(True)

    def save_orientation(self):
        """
        Method to save the current image orientation.
        """
        # get current file contents
        try:
            with open('imagers.db') as json_file:
                data = json.load(json_file)

        except json.decoder.JSONDecodeError:
            # give up if there's no file for now...
            pass

        # check if there is already information about this imager
        if self.imager in data:
            # if so, add orientation information
            data[self.imager]['orientation'] = self.orientation
        else:
            # if not, initialize information about this imager
            data[self.imager] = {}
            data[self.imager]['orientation'] = self.orientation

        # write to the file under the corresponding imager field
        with open('imagers.db', 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def set_time_range(self, time_range=10.0):
        """
        Method to set the time range of the centroid, etc plots.
        :param time_range: float
            time for x-axis in seconds
        """
        # check if this is called as a callback
        if self.sender():
            rangeLineEdit = self.sender()
            try:
                time_range = float(rangeLineEdit.text())
            except ValueError:
                time_range = 10.0
                rangeLineEdit.setText('10.0')

        # set time range for all stripchart-type plots
        for plot in self.all_plots:
            plot.set_time_range(time_range)

    def setup_legend(self, legend):
        """
        Method to set up a legend for a plot. This should probably belong in the PPM_widgets module.
        Parameters
        ----------
        legend: string
            pyqtgraph Legend object
        """

        # set style: white text, 10pt
        legendLabelStyle = {'color': '#FFF', 'size': '10pt'}
        # the following was just taken from the web...
        for item in legend.items:
           for single_item in item:
               if isinstance(single_item, pg.graphicsItems.LabelItem.LabelItem):
                   single_item.setText(single_item.text, **legendLabelStyle)

    def run_alignment_screen(self):
        """
        Method to open the alignment screen
        """

        cam_name = self.imagerpv

        alignment_app = App(parent=self, imager=cam_name)
        alignment_app.show()

    def change_line(self, index):
        """
        Method to change which beamline from which to select an imager.

        Parameters
        ----------
        index: int
            index corresponding to which beamline as defined in self.line_list
        """
        # update line
        self.line = self.line_list[index]
        self.imager_list = self.imager_dict[self.line]
        self.imagerpv_list = self.imagerpv_dict[self.line]
        self.imagerComboBox.clear()
        self.imagerComboBox.addItems(self.imager_list)
        self.change_imager(0)

    def change_imager(self, index):
        """
        Method to change settings based on selection of a new imager

        Parameters
        ----------
        index: int
            index corresponding to which imager as defined in self.imager_list
        """
        # update imager
        self.imager = self.imager_list[index]
        self.imageGroupBox.setTitle(self.imager)
        self.wavefrontGroupBox.setTitle(self.imager)
        # check if this imager has a wavefront sensor
        if self.imager in self.WFS_list:
            self.wavefrontCheckBox.setEnabled(True)
            # update wfs_name
            self.wfs_name = self.WFS_dict[self.imager]
        else:
            self.wavefrontCheckBox.setChecked(False)
            self.wavefrontCheckBox.setEnabled(False)
            # no wavefront sensor associated with this imager
            self.wfs_name = None

        self.imagerpv = self.imagerpv_list[index]
        self.imagerControls.change_imager(self.imagerpv)
        self.wfsControls.change_wfs(self.wfs_name)
        # reset data_dict
        self.reset_data_dict()
        self.load_orientation()

    def reset_data_dict(self):
        """
        Method to reset the data. This is called when a new imager is selected for instance.
        This should really be cleaned up so that it's easier to add new items to the data dictionary. Basically
        just need to categorize different types of data based on array size in terms of how to initialize them. This
        could be defined in a file.
        """
        N = 1024
        self.data_dict['im0'] = np.zeros((1024,1024))
        self.data_dict['cx'] = -np.ones(N)
        self.data_dict['cy'] = -np.ones(N)
        self.data_dict['wx'] = -np.ones(N)
        self.data_dict['wy'] = -np.ones(N)
        self.data_dict['intensity'] = -np.ones(N)
        self.data_dict['cx_smooth'] = -np.ones(N)
        self.data_dict['cy_smooth'] = -np.ones(N)
        self.data_dict['wx_smooth'] = -np.ones(N)
        self.data_dict['wy_smooth'] = -np.ones(N)
        self.data_dict['timestamps'] = -np.ones(N)
        self.data_dict['counter'] = 0.
        self.data_dict['pixSize'] = 0.0
        self.data_dict['lineout_x'] = np.zeros(100)
        self.data_dict['lineout_y'] = np.zeros(100)
        self.data_dict['projection_x'] = np.zeros(100)
        self.data_dict['projection_y'] = np.zeros(100)
        self.data_dict['fit_x'] = np.zeros(100)
        self.data_dict['fit_y'] = np.zeros(100)
        self.data_dict['x'] = np.linspace(-1024, 1023, 100)
        self.data_dict['y'] = np.linspace(-1024, 1023, 100)

        # wavefront sensor data
        self.data_dict['z_x'] = -np.ones(N)
        self.data_dict['z_y'] = -np.ones(N)
        self.data_dict['z_x_smooth'] = -np.ones(N)
        self.data_dict['z_y_smooth'] = -np.ones(N)
        self.data_dict['rms_x'] = -np.ones(N)
        self.data_dict['rms_y'] = -np.ones(N)
        self.data_dict['rms_x_smooth'] = -np.ones(N)
        self.data_dict['rms_y_smooth'] = -np.ones(N)
        self.data_dict['x_res'] = np.zeros(100)
        self.data_dict['y_res'] = np.zeros(100)
        self.data_dict['x_prime'] = np.linspace(-1024, 1023, 100)
        self.data_dict['y_prime'] = np.linspace(-1024, 1023, 100)

    def change_state(self):
        """
        Method to start the calculation running, or stop it.
        """

        # check if "Run" was selected
        if self.runButton.text() == 'Run':

            # check if we are going to calculate the wavefront. Set wfs_name to None if not.
            if self.wavefrontCheckBox.isChecked():
                wfs_name = self.wfs_name
            else:
                wfs_name = None

            # get Talbot fraction. This will eventually be automated based on the photon energy and WFS state.
            try:
                fraction = float(self.wfsControls.fractionLineEdit.text())
            except ValueError:
                fraction = 1

            # initialize processing object. This really needs a dictionary as input...
            self.processing = RunProcessing(self.imagerpv, self.data_dict, self.averageWidget, wfs_name=wfs_name,
                                            threshold=self.imagerStats.get_threshold(), focusFOV=self.displayWidget.FOV, fraction=fraction, focus_z=self.displayWidget.focus_z)

            # find out what the FOV of the screen is
            width, height = self.processing.get_FOV()
            # set the orientation for processing
            self.processing.set_orientation(self.orientation)

            # update viewboxes based on FOV
            self.imageWidget.update_viewbox(width, height)
            if self.displayWidget.display_choice == 'Focus':
                self.wavefrontWidget.update_viewbox(self.displayWidget.FOV, self.displayWidget.FOV)
            else:
                # this would eventually change for the FFT option
                self.wavefrontWidget.update_viewbox(width, height)

            # update crosshair sizes
            self.crosshairsWidget.update_crosshair_width()
            self.wavefrontCrosshairsWidget.update_crosshair_width()

            # update width for circle displayed on beam
            self.imagerStats.update_width()

            # initialize a new thread
            self.thread = QtCore.QThread()
            self.thread.start()

            # move the processing object to the new thread
            self.processing.moveToThread(self.thread)
            # change the button state
            self.runButton.setText('Stop')
            # disable wavefront sensor checkbox until stop is pressed
            self.wavefrontCheckBox.setEnabled(False)

            # connect processing object to plotting function
            self.processing.sig.connect(self.update_plots)

            # disable imager selection until Stop is pressed
            self.lineComboBox.setEnabled(False)
            self.imagerComboBox.setEnabled(False)

        # check if "Stop" was selected
        elif self.runButton.text() == 'Stop':

            # stop processing and quit the thread
            self.processing.stop()
            self.thread.quit()
            self.thread.wait()

            # update the button to be ready to "Run"
            self.runButton.setText('Run')
            # re-enable wavefront sensor checkbox if imager is has a wavefront sensor
            if self.imager in self.WFS_list:
                self.wavefrontCheckBox.setEnabled(True)

            # re-enable imager selection
            self.lineComboBox.setEnabled(True)
            self.imagerComboBox.setEnabled(True)

    @staticmethod
    def normalize_image(image):
        """
        This probably belongs somewhere else... The idea is to normalize an image to 8-bit dynamic range
        for saving to a png file

        Parameters
        ----------
        image: ndarray (N,M)
            input image
        Returns
        -------
        ndarray (N,M)
            output image
        """
        image -= np.min(image)
        image *= 255./float(np.max(image))
        image = np.array(image,dtype='uint8')
        return image

    def save_image(self):
        """
        Method to save a png image based on the latest image grabbed
        """
        formats = 'Portable Network Graphic (*.png)'
        filename = QtGui.QFileDialog.getSaveFileName(self, 
                'Save Image','untitled.png',formats)
        # make sure a file name was chosen
        if not filename[0] == '':
            # normalize the image and write to file
            im = App.normalize_image(self.data_dict['im1'])
            filename = App.get_filename(filename)
            imageio.imwrite(filename,im)
        print(filename)

    @staticmethod
    def get_filename(name):
        """
        Method to get the filename from QFileDialog
        Parameters
        ----------
        name: list of strings
            first entry is the full path including file name, second entry is the extension

        Returns
        -------
        string
            full path including file name and extension
        """
        path = name[0]
        extension = name[1].split('*')[1][0:4]
        if path[-4:] != extension:
            path = path + extension
        return path

    def closeEvent(self, event):
        """
        Method to control what happens if the window is closed.

        Parameters
        ----------
        event: signal
        """
        # check if anything is running, otherwise do nothing else
        if self.runButton.text() == 'Stop':
            self.processing.stop()
            self.thread.quit()
            self.thread.wait()

    def update_plots(self, data_dict):
        """
        Method to update all the plots. Would be nice to find a way to make this less explicit. One idea would be
        to pass the dictionary keys into the plots when they are first initialized. Seems like passing the dictionary
        around to all the plot functions probably only passes by reference or something.

        Parameters
        ----------
        data_dict: dict
            This is where all the data to display is stored
        """

        x = data_dict['x']
        y = data_dict['y']
        image_data = data_dict['im1']
        xlineout = data_dict['lineout_x']
        ylineout = data_dict['lineout_y']
        xprojection = data_dict['projection_x']
        yprojection = data_dict['projection_y']
        fit_x = data_dict['fit_x']
        fit_y = data_dict['fit_y']

        # wfs widget plots
        x_prime = data_dict['x_prime']
        y_prime = data_dict['y_prime']
        x_res = data_dict['x_res']
        y_res = data_dict['y_res']
        x_res_fit = np.zeros_like(x_res)
        y_res_fit = np.zeros_like(y_res)

        # update main image and lineouts
        self.imageWidget.update_plots(image_data, x, y, xprojection, yprojection, fit_x, fit_y, 
                xlineout_data=xlineout, ylineout_data=ylineout)

        # update wavefront tab
        if self.wavefrontCheckBox.isChecked():

            # get focus coordinates
            xf = data_dict['xf']

            # check what is supposed to be displayed. Would be nice to also clean this up
            if self.displayWidget.display_choice == 'Focus':
                xline = data_dict['focus_horizontal']
                yline = data_dict['focus_vertical']
                self.wavefrontWidget.update_plots(data_dict['focus'], xf, xf, xline, yline, xline, yline)
            elif self.displayWidget.display_choice == 'Fourier transform':
                self.wavefrontWidget.update_plots(data_dict['F0'], x, y, xprojection, yprojection, fit_x, fit_y)
            elif self.displayWidget.display_choice == 'Phase':
                self.wavefrontWidget.update_plots(data_dict['wave'], x_prime, y_prime, x_res, y_res, x_res, y_res)

            # update wavefront tab stripchart plots
            self.focus_plot.update_plots(data_dict['timestamps'], x=data_dict['z_x'], y=data_dict['z_y'], x_smooth=data_dict['z_x_smooth'], y_smooth=data_dict['z_y_smooth'])
            self.rms_plot.update_plots(data_dict['timestamps'], x=data_dict['rms_x'], y=data_dict['rms_y'], x_smooth=data_dict['rms_x_smooth'], y_smooth=data_dict['rms_y_smooth'])

        # update data. Should check if this is ever used anywhere...
        self.data_dict = data_dict

        # update centroid plots
        self.centroid_plot.update_plots(data_dict['timestamps'], x=data_dict['cx'], y=data_dict['cy'], x_smooth=data_dict['cx_smooth'], y_smooth=data_dict['cy_smooth'])
        self.width_plot.update_plots(data_dict['timestamps'], x=data_dict['wx'], y=data_dict['wy'], x_smooth=data_dict['wx_smooth'], y_smooth=data_dict['wy_smooth'])

        self.label.setText(data_dict['tx'])

        # dict for stats. This probably isn't needed, should just pass in full data_dict
        stats_dict = {}
        stats_dict['cx'] = data_dict['cx']
        stats_dict['cy'] = data_dict['cy']
        stats_dict['wx'] = data_dict['wx']
        stats_dict['wy'] = data_dict['wy']
        stats_dict['intensity'] = data_dict['intensity']

        # dict for wfs. This probably isn't needed, should just pass in full data_dict
        wfs_dict = {}
        wfs_dict['z_x'] = data_dict['z_x']
        wfs_dict['z_y'] = data_dict['z_y']
        wfs_dict['rms_x'] = data_dict['rms_x']
        wfs_dict['rms_y'] = data_dict['rms_y']

        # update stats values
        self.imagerStats.update_stats(stats_dict)
        self.wfsStats.update_stats(wfs_dict)
