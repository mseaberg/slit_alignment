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

class YagAlign:
    
    def __init__(self):
        
        # the TEMPLATE
        im0 = imageio.imread("pattern.png")[:,:,3]
        Nt, Mt = np.shape(im0)
        im0 = np.pad(im0,((int((512-Nt)/2+1),int((512-Nt)/2)),(int((512-Mt)/2+1),int((512-Mt)/2))),mode='constant')
        self.Nt,self.Mt = np.shape(im0)
       
        self.im0 = np.array(im0,dtype=float)
        #self.im0 = 255.0-im0
        
        x = np.linspace(-self.Mt/2,self.Mt/2-1,self.Mt)
        fx_max = 0.5
        fx = x*fx_max/np.max(x)
        
        y = np.linspace(-self.Nt/2,self.Nt/2-1,self.Nt)
        fy_max = 0.5
        fy = y*fy_max/np.max(y)
        
        self.fx, self.fy = np.meshgrid(fx,fy)
        
        self.fr = np.sqrt(self.fx**2+self.fy**2)
        self.ftheta = np.arctan2(self.fy,self.fx)
        
        self.logbase = 1.1
        
    def check_alignment(self,data):
        
        # assume that the data coming in is already pretty well-centered
        ims = {}
        # crop out one corner
        ims[0] = data[0:200,0:200]
        # get the average border value
        max0 = self.get_borderval(ims[0])
        # normalize image
        ims[0] = (ims[0]-np.min(ims[0]))/(max0-np.min(ims[0]))*255
        # pad image up to a 512x512 array
        ims[0] = np.pad(ims[0],((0,512-200),(0,512-200)),'constant',
                constant_values=self.get_borderval(ims[0]))
        ims[1] = data[0:200,1848:]
        max1 = self.get_borderval(ims[1])
        ims[1] = (ims[1]-np.min(ims[1]))/(max1-np.min(ims[1]))*255
        ims[1] = np.pad(ims[1],((0,512-200),(512-200,0)),'constant',
                constant_values=self.get_borderval(ims[1]))
        ims[2] = data[1848:,0:200]
        max2 = self.get_borderval(ims[2])
        ims[2] = (ims[2]-np.min(ims[2]))/(max2-np.min(ims[2]))*255
        ims[2] = np.pad(ims[2],((512-200,0),(0,512-200)),'constant',
                constant_values=self.get_borderval(ims[2]))
        ims[3] = data[1848:,1848:]
        max3 = self.get_borderval(ims[3])
        ims[3] = (ims[3]-np.min(ims[3]))/(max3-np.min(ims[3]))*255
        ims[3] = np.pad(ims[3],((512-200,0),(512-200,0)),'constant',
                constant_values=self.get_borderval(ims[3]))

        shifts = {}
        transforms = {}
        for i in range(4):
            shifts[i], transforms[i] = self.get_transform(ims[i])

        contrast = np.zeros(4)
        for i in range(4):
            contrast[i] = self.get_contrast(shifts[i])['avg']
    
        translation = np.zeros((4,2))
        rotation = np.zeros(4)
        scale = np.zeros(4)
        for i in range(4):
            scale[i] = transforms[i]['scale']
            rotation[i] = transforms[i]['theta']
            translation[i,:] = transforms[i]['translation']

        output = {}
        output['shifts'] = shifts
        output['scale'] = scale
        output['contrast'] = contrast
        output['rotation'] = rotation
        output['translation'] = translation
        
        return output
        
    @staticmethod
    def get_contrast(img):
        img[img<0] = 0
        img = img-np.min(img)
        line1 = np.mean(img[220:250,220:250],axis=1)
        line2 = np.mean(img[220:250,259:290],axis=0)
        line3 = np.mean(img[260:290,220:250],axis=0)
        line4 = np.mean(img[260:290,260:290],axis=1)

        norm = np.mean(img[310:370,220:295])

        #r1 = np.std(line1/norm)*np.sqrt(2)*2
        #r2 = np.std(line2/norm)*np.sqrt(2)*2
        #r3 = np.std(line3/norm)*np.sqrt(2)*2
        #r4 = np.std(line4/norm)*np.sqrt(2)*2
        #line1 = line1/norm
        #line2 = line2/norm
        #line3 = line3/norm
        #line4 = line4/norm
        r1 = np.std(line1/np.max(line1))*np.sqrt(2)*2
        r2 = np.std(line2/np.max(line2))*np.sqrt(2)*2
        r3 = np.std(line3/np.max(line3))*np.sqrt(2)*2
        r4 = np.std(line4/np.max(line4))*np.sqrt(2)*2

        #r1 = (np.max(line1)-np.min(line1))/(np.max(line1)+np.min(line1))
        #r2 = (np.max(line2)-np.min(line2))/(np.max(line2)+np.min(line2))
        #r3 = (np.max(line3)-np.min(line3))/(np.max(line3)+np.min(line3))
        #r4 = (np.max(line4)-np.min(line4))/(np.max(line4)+np.min(line4))
        r_avg = (r1+r2+r3+r4)/4.0
        
        contrast = {}
        contrast['1'] = r1
        contrast['2'] = r2
        contrast['3'] = r3
        contrast['4'] = r4
        contrast['avg'] = r_avg
        
        return contrast
    
    def get_transform(self,img):
       


        # log-polar coordinate system
        r1 = np.linspace(0,np.log(self.Nt/8)/np.log(self.logbase),128)
        r1p = np.linspace(0,np.log(self.Nt/8)/np.log(self.logbase),128)
        theta1 = np.linspace(-np.pi/36,np.pi/36,11)
        r1,theta1 = np.meshgrid(r1,theta1)
        
        r2 = np.exp(r1*np.log(self.logbase))
        
        # coordinates to map to
        y = r2*np.sin(theta1) + self.Nt/2
        x = r2*np.cos(theta1) + self.Mt/2
        
        # FFT of each image
        F1 = self.FT(self.im0)
        F2 = self.FT(img)
        
        # map to log polar coordinates
        F1abs_out = np.empty_like(y)
        ndimage.map_coordinates(np.log(np.abs(F1)+1),[y,x],output=F1abs_out,order=3)
        
        F2abs_out = np.empty_like(y)
        ndimage.map_coordinates(np.log(np.abs(F2)+1),[y,x],output=F2abs_out,order=3)
        
        # Fourier transforms of log-polar FFTs
        M1 = np.fft.fft2(F1abs_out)
        M2 = np.fft.fft2(F2abs_out)
        
        # cross-correlation of M1 and M2
        cps = self.x_corr(M1,M2)
        cps_real = np.abs(np.fft.ifft2(cps))
        
        # set first column of x-corr to zero (by design we don't expect that zoom = 1)
        cps_real[:,0] = 0
        # restrict zoom to be within a certain range
        cps_real[:,32:] = 0

        # find correlation peak
        peak = np.unravel_index(np.argmax(cps_real),cps_real.shape)
        # determine rotation from peak location
        theta_offset = theta1[peak[0],0]*180/np.pi + 5
        # determine zoom from peak location
        scale = self.logbase**r1p[peak[1]]

        # get theta nearest to zero
        if theta_offset>45:
            theta_offset = 90-theta_offset

        # get background value of image
        bgval = self.get_borderval(img)

        # change scale and rotate based on above results
        zoom_out = interpolate.zoom(interpolate.rotate(img,-theta_offset,reshape=False,
                                                       mode='constant',cval=bgval),1./scale)
        # get new image size
        Nz,Mz = np.shape(zoom_out)
        
        # embed image in a size matching the template
        zoom_embed = np.zeros_like(self.im0) + bgval

        zoom_embed = self.embed_to(zoom_embed,zoom_out)
        
        ## figure out translation
        # we already have F1
        F2p = self.FT(zoom_embed)

        # cross-correlation for determining translation
        cps = self.x_corr(F1,F2p)
        cps_real = np.fft.fftshift(np.abs(np.fft.ifft2(np.fft.fftshift(cps))))

        # find peak (relative to center)
        peak = np.unravel_index(np.argmax(cps_real),cps_real.shape)
        peak = np.array(peak) - 256

        # line up with template based on translation
        shifted = np.roll(zoom_embed, peak, axis=(0,1))
        
        transform = {}
        transform['scale'] = scale
        transform['theta'] = theta_offset
        transform['translation'] = peak
        
        return shifted, transform

    @staticmethod 
    def FT(img):
        N,M = np.shape(img)
        F = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img-np.mean(img))))
        # remove discontinuity at zero
        F[int(N/2),int(M/2)] = F[int(N/2),int(M/2)-1]
        return F
    
    @staticmethod
    def x_corr(F1,F2):
        cps = F1*np.conj(F2)/(np.abs(F1)*np.abs(F2)+np.finfo(float).eps)
        
        return cps

    @staticmethod
    def get_borderval(img, radius=None):
        """
        Given an image and a radius, examine the average value of the image
        at most radius pixels from the edge
        """
        if radius is None:
            mindim = min(img.shape)
            radius = max(1, mindim // 20)
        mask = np.zeros_like(img, dtype=np.bool)
        mask[:, :radius] = True
        mask[:, -radius:] = True
        mask[:radius, :] = True
        mask[-radius:, :] = True

        mean = np.median(img[mask])
        return mean

    @classmethod
    def embed_to(cls,where, what):
        """
        Given a source and destination arrays, put the source into
        the destination so it is centered and perform all necessary operations
        (cropping or aligning)

        Args:
            where: The destination array (also modified inplace)
            what: The source array

        Returns:
            The destination array
        """
        slices_from, slices_to = cls._get_emslices(where.shape, what.shape)

        where[slices_to[0], slices_to[1]] = what[slices_from[0], slices_from[1]]
        return where
    
    @staticmethod
    def _get_emslices(shape1, shape2):
        """
        Common code used by :func:`embed_to` and :func:`undo_embed`
        """
        slices_from = []
        slices_to = []
        for dim1, dim2 in zip(shape1, shape2):
            diff = dim2 - dim1
            # In fact: if diff == 0:
            slice_from = slice(None)
            slice_to = slice(None)

            # dim2 is bigger => we will skip some of their pixels
            if diff > 0:
                # diff // 2 + rem == diff
                rem = diff - (diff // 2)
                #rem = diff % 2
                slice_from = slice(diff // 2, dim2 - rem)
            if diff < 0:
                diff *= -1
                rem = diff - (diff // 2)
                #rem = diff % 2
                slice_to = slice(diff // 2, dim1 - rem)
            slices_from.append(slice_from)
            slices_to.append(slice_to)
        return slices_from, slices_to

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

        #legendLabelStyle = {'color': '#FFF', 'size': '10pt'}
        #for item in legend.items:
        #    for single_item in item:
        #        if isinstance(single_item, pg.graphicsItems.LabelItem.LabelItem):
        #            single_item.setText(single_item.text, **legendLabelStyle)

        #  rotation plot
        self.rotation_plot = self.canvas.addPlot(row=2,col=4,rowspan=1,colspan=1)
        



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
        self.data_dict['cx'] = np.empty(100)
        self.data_dict['cy'] = np.empty(100)
        self.data_dict['timestamps'] = np.empty(100)
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
        
        self.hplot[0].setData(timestamp, cx)
        self.contrast_plot.setXRange(-10, 0)
        self.rplot[0].setData(timestamp, cy)
        self.rotation_plot.setXRange(-10, 0)

        #self.circ0.setRect(full_center[1]-25,full_center[0]-25,50,50)

        self.label.setText(data_dict['tx'])


class RunRegistration(QtCore.QObject):

    sig = QtCore.pyqtSignal(dict)

    def __init__(self, yag, data_dict):
        super(RunRegistration, self).__init__()

        #self.my_signal = QtCore.Signal()

        #self.gui = gui

        self.yag1 = yag
        self.epics_name = ''
        if len(sys.argv)>1:
            self.cam_name = sys.argv[1]
            self.epics_name = sys.argv[1] + 'IMAGE2:'

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

            self.data_dict['contrast'] = np.roll(self.data_dict['contrast'],-1,axis=1)
            self.data_dict['rotation'] = np.roll(self.data_dict['rotation'],-1,axis=1)
            self.data_dict['iteration'] = np.roll(self.data_dict['iteration'],-1,axis=1)
            self.data_dict['iteration'][:,-1] = self.counter

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
