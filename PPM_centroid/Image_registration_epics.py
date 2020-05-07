#!/usr/bin/env python
# coding: utf-8

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
from processing_module import RunRegistration
from analysis_tools import YagAlign

Ui_MainWindow, QMainWindow = loadUiType('image_register.ui')


class App(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__()
        self.setupUi(self)


        self.runButton.clicked.connect(self.change_state)

        self.minValue.returnPressed.connect(self.set_min)
        self.maxValue.returnPressed.connect(self.set_max)

        self.actionSave.triggered.connect(self.save_image)

        # Full image
        self.view0 = self.canvas.addViewBox(row=0,col=0,rowspan=3,colspan=3)
        self.view0.setAspectLocked(True)
        self.view0.setRange(QtCore.QRectF(0,0, 2048, 2048))
        self.img0 = pg.ImageItem(border='w')
        self.view0.addItem(self.img0)
        rect1 = QtWidgets.QGraphicsRectItem(0,0,160,160)
        rect1.setPen(QPen(Qt.cyan, 8, Qt.SolidLine))
        rect2 = QtWidgets.QGraphicsRectItem(1888,0,160,160)
        rect2.setPen(QPen(Qt.darkMagenta, 8, Qt.SolidLine))
        rect3 = QtWidgets.QGraphicsRectItem(0,1888,160,160)
        rect3.setPen(QPen(Qt.red, 8, Qt.SolidLine))
        rect4 = QtWidgets.QGraphicsRectItem(1888,1888,160,160)
        rect4.setPen(QPen(Qt.green, 8, Qt.SolidLine))


        #circ1 = QtWidgets.QGraphicsEllipseItem(1024-25,1024-25,50,50)
        #circ1.setPen(QPen(Qt.green, 8, Qt.SolidLine))
        crossx = QtWidgets.QGraphicsLineItem(1024-25,1024,1024+25,1024)
        crossy = QtWidgets.QGraphicsLineItem(1024,1024-25,1024,1024+25)
        crossx.setPen(QPen(Qt.green, 8, Qt.SolidLine))
        crossy.setPen(QPen(Qt.green, 8, Qt.SolidLine))

        
        #self.circ0 = QtWidgets.QGraphicsEllipseItem(1024-25,1024-25,50,50)
        #self.circ0.setPen(QPen(Qt.red, 8, Qt.SolidLine))
        self.crossx = QtWidgets.QGraphicsLineItem(1024-25,1024,1024+25,1024)
        self.crossy = QtWidgets.QGraphicsLineItem(1024,1024-25,1024,1024+25)
        self.crossx.setPen(QPen(Qt.red, 8, Qt.SolidLine))
        self.crossy.setPen(QPen(Qt.red, 8, Qt.SolidLine))

        
        self.circ1 = QtWidgets.QGraphicsRectItem(256-25,1792-25,50,50)
        self.circ1.setPen(QPen(Qt.red, 8, Qt.SolidLine))
        self.circ2 = QtWidgets.QGraphicsRectItem(1792-25,1792-25,50,50)
        self.circ2.setPen(QPen(Qt.green, 8, Qt.SolidLine))
        self.circ3 = QtWidgets.QGraphicsRectItem(256-25,256-25,50,50)
        self.circ3.setPen(QPen(Qt.cyan, 8, Qt.SolidLine))
        self.circ4 = QtWidgets.QGraphicsRectItem(1792-25,256-25,50,50)
        self.circ4.setPen(QPen(Qt.darkMagenta, 8, Qt.SolidLine))
        self.view0.addItem(rect1)
        self.view0.addItem(rect2)
        self.view0.addItem(rect3)
        self.view0.addItem(rect4)
        #self.view0.addItem(circ1)
        self.view0.addItem(crossx)
        self.view0.addItem(crossy)
        self.view0.addItem(self.crossx)
        self.view0.addItem(self.crossy)
        #self.view0.addItem(self.circ0)
        self.view0.addItem(self.circ1)
        self.view0.addItem(self.circ2)
        self.view0.addItem(self.circ3)
        self.view0.addItem(self.circ4)

        self.pix_size_text = pg.TextItem('Pixel size: %.2f microns' % 0.0,
                color=(200,200,200), border='c', fill='b',anchor=(0,1))
        self.pix_size_text.setFont(QtGui.QFont("", 10, QtGui.QFont.Bold))
        self.pix_size_text.setPos(300,.5)
        self.view0.addItem(self.pix_size_text)


        # Top left image
        self.view1 = self.canvas.addViewBox(row=0,col=3,rowspan=1,colspan=1)
        self.view1.setAspectLocked(True)
        self.view1.setRange(QtCore.QRectF(0,0, 90, 90))
        self.img1 = pg.ImageItem(border='w',title='Top Left')
        self.view1.addItem(self.img1)
        recttl = QtWidgets.QGraphicsRectItem(0, 0, 90, 90)
        recttl.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        self.view1.addItem(recttl)


        # Top right image
        self.view2 = self.canvas.addViewBox(row=0,col=4,rowspan=1,colspan=1)
        self.view2.setAspectLocked(True)
        self.view2.setRange(QtCore.QRectF(0,0, 90, 90))
        self.img2 = pg.ImageItem(border='w',title='Top Right')
        self.view2.addItem(self.img2)
        recttr = QtWidgets.QGraphicsRectItem(0, 0, 90, 90)
        recttr.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        self.view2.addItem(recttr)

        # Bottom left image
        self.view3 = self.canvas.addViewBox(row=1,col=3,rowspan=1,colspan=1)
        self.view3.setAspectLocked(True)
        self.view3.setRange(QtCore.QRectF(0,0, 90, 90))
        self.img3 = pg.ImageItem(border='w',title='Bottom Left')
        self.view3.addItem(self.img3)
        rectbl = QtWidgets.QGraphicsRectItem(0, 0, 90, 90)
        rectbl.setPen(QPen(Qt.cyan, 2, Qt.SolidLine))
        self.view3.addItem(rectbl)

        # Bottom right image
        self.view4 = self.canvas.addViewBox(row=1,col=4,rowspan=1,colspan=1)
        self.view4.setAspectLocked(True)
        self.view4.setRange(QtCore.QRectF(0,0, 90, 90))
        self.img4 = pg.ImageItem(border='w',title='Bottom Right')
        self.view4.addItem(self.img4)
        rectbr = QtWidgets.QGraphicsRectItem(0, 0, 90, 90)
        rectbr.setPen(QPen(Qt.darkMagenta, 2, Qt.SolidLine))
        self.view4.addItem(rectbr)

        #  contrast plot
        self.contrast_plot = self.canvas.addPlot(row=2,col=3,rowspan=1,colspan=1)
        



        labelStyle = {'color': '#FFF', 'font-size': '12pt'}

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setFamily('Arial')


        xaxis = self.contrast_plot.getAxis('bottom')
        xaxis.setLabel(text='Image number',**labelStyle)
        xaxis.tickFont = font
        xaxis.setPen(pg.mkPen('w',width=1))
        yaxis = self.contrast_plot.getAxis('left')
        yaxis.setLabel(text='Contrast',**labelStyle)
        yaxis.tickFont = font
        yaxis.setPen(pg.mkPen('w',width=1))

        self.contrast_plot.showGrid(x=True,y=True,alpha=.8)
        #self.contrast_plot.setYRange(0,1.5)
        self.hplot = {}
        names = ['Top Left','Top Right','Bottom Left','Bottom Right']
        colors = ['r','g','c','m']
        
        
        legend = self.contrast_plot.addLegend()
        for i in range(4):

            #self.hplot[i] = self.contrast_plot.plot(np.linspace(-99,0,100),np.zeros(100),pen=(i,4),name=names[i])
            self.hplot[i] = self.contrast_plot.plot(np.linspace(-99,0,100), np.zeros(100),
                    pen=pg.mkPen(colors[i], width=5),name=names[i])

        legendLabelStyle = {'color': '#FFF', 'size': '10pt'}
        for item in legend.items:
            for single_item in item:
                if isinstance(single_item, pg.graphicsItems.LabelItem.LabelItem):
                    single_item.setText(single_item.text, **legendLabelStyle)

        #  rotation plot
        self.rotation_plot = self.canvas.addPlot(row=2,col=4,rowspan=1,colspan=1)
        



        labelStyle = {'color': '#FFF', 'font-size': '12pt'}

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setFamily('Arial')


        xaxis = self.rotation_plot.getAxis('bottom')
        xaxis.setLabel(text='Image number',**labelStyle)
        xaxis.tickFont = font
        xaxis.setPen(pg.mkPen('w',width=1))
        yaxis = self.rotation_plot.getAxis('left')
        yaxis.setLabel(text='Rotation (degrees)',**labelStyle)
        yaxis.tickFont = font
        yaxis.setPen(pg.mkPen('w',width=1))

        self.rotation_plot.showGrid(x=True,y=True,alpha=.8)
        #self.contrast_plot.setYRange(0,1.5)
        self.rplot = {}
        names = ['Top Left','Top Right','Bottom Left','Bottom Right']
        colors = ['r','g','c','m']
        
        
        legend = self.contrast_plot.addLegend()
        for i in range(4):

            #self.hplot[i] = self.contrast_plot.plot(np.linspace(-99,0,100),np.zeros(100),pen=(i,4),name=names[i])
            self.rplot[i] = self.rotation_plot.plot(np.linspace(-99,0,100), np.zeros(100),
                    pen=pg.mkPen(colors[i], width=5),name=names[i])

        legendLabelStyle = {'color': '#FFF', 'size': '10pt'}
        for item in legend.items:
            for single_item in item:
                if isinstance(single_item, pg.graphicsItems.LabelItem.LabelItem):
                    single_item.setText(single_item.text, **legendLabelStyle)




        self.yag1 = YagAlign()

        # the image to be transformed
        #im1 = np.array(imageio.imread("test_pattern.png")[32:2650, 32:2650, 3],dtype='float')
        im1 = np.array(imageio.imread("PPM_alignment/IM1K2.png"),dtype='float')
        im1 = 255 - im1

        N, M = np.shape(im1)
        scale = 2048.0 / N

        im0 = interpolate.zoom(im1, scale)

        self.data_dict = {}
        self.data_dict['im0'] = im0
        self.data_dict['contrast'] = np.zeros((4, 100))
        self.data_dict['rotation'] = np.zeros((4,100))
        self.data_dict['iteration'] = np.tile(np.linspace(-99, 0, 100), (4, 1))
        self.data_dict['counter'] = 0.
        self.data_dict['center'] = np.zeros((4,2))
        self.data_dict['scale'] = np.zeros(4)
        self.data_dict['pixSize'] = 0.0
        #self.data_dict['centering'] = np.zeros(2)
        self.set_min()
        self.set_max()

    def change_state(self):
        if self.runButton.text() == 'Run':

            self.registration = RunRegistration(self.yag1, self.data_dict)
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
            
        pixSize = data_dict['pixSize']

        full_center = np.mean(data_dict['center'],axis=0)

        self.pix_size_text.setText('Pixel size: %.2f microns' 
                % data_dict['pixSize'])
        self.label_pixSize.setText('Pixel size: %.2f \u03BCm'
                % data_dict['pixSize'])
        centerText = ('YAG center (x,y): %.2f \u03BCm, %.2f \u03BCm'
                % ((full_center[1]-1024)*pixSize, (full_center[0]-1024)*pixSize))
        self.label_center.setText(centerText)
        self.data_dict = data_dict
        self.img0.setImage(np.flipud(data_dict['im1']).T,
                levels=(self.minimum, self.maximum))


        center = data_dict['center']
        scale = data_dict['scale']
        #self.circ0.setRect(full_center[1]-25,full_center[0]-25,50,50)

        self.crossx.setLine(full_center[1]-25,full_center[0],
                full_center[1]+25,full_center[0])
        self.crossy.setLine(full_center[1],full_center[0]-25,
                full_center[1],full_center[0]+25)


        self.circ1.setRect(center[0,1]-scale[0]*45,center[0,0]-scale[0]*45,
                90*scale[0],90*scale[0])
        self.circ2.setRect(center[1,1]-scale[1]*45,center[1,0]-scale[1]*45,
                90*scale[1],90*scale[1])
        self.circ3.setRect(center[2,1]-scale[2]*45,center[2,0]-scale[2]*45,
                90*scale[2],90*scale[2])
        self.circ4.setRect(center[3,1]-scale[3]*45,center[3,0]-scale[3]*45,
                90*scale[3],90*scale[3])

        self.img1.setImage(np.flipud(data_dict['shifts'][0][210:300, 210:300]).T, 
                levels=(self.minimum, self.maximum))
        self.img2.setImage(np.flipud(data_dict['shifts'][1][210:300, 210:300]).T,
                levels=(self.minimum, self.maximum))
        self.img3.setImage(np.flipud(data_dict['shifts'][2][210:300, 210:300]).T,
                levels=(self.minimum, self.maximum))
        self.img4.setImage(np.flipud(data_dict['shifts'][3][210:300, 210:300]).T,
                levels=(self.minimum, self.maximum))

        iteration = data_dict['iteration']
        contrast = data_dict['contrast']
        rotation = data_dict['rotation']

        for i in range(4):
            self.hplot[i].setData(iteration[i, :], contrast[i, :])

            self.rplot[i].setData(iteration[i, :], rotation[i, :])

        #self.contrast_plot.setYRange(0, 1.5)
        self.contrast_plot.setXRange(np.min(iteration), np.max(iteration))
        self.rotation_plot.setXRange(np.min(iteration), np.max(iteration))
        self.rotation_plot.setYRange(-4,4)
        self.label.setText(data_dict['tx'])


if __name__ == '__main__':

    warnings.filterwarnings('ignore', '.*output shape of zoom.*')
    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())
