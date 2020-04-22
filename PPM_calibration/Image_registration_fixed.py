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
        ims[0] = data[0:200,0:200]
        ims[0] = np.pad(ims[0],((512-200,0),(512-200,0)),'constant',constant_values=np.min(ims[0]))
        #im1[:,125:] = 255
        #ims[1] = data[0:512,1536:]
        ims[1] = data[0:200,1848:]
        ims[1] = np.pad(ims[1],((0,512-200),(512-200,0)),'constant',constant_values=np.min(ims[1]))
        ims[2] = data[1848:,0:200]
        ims[2] = np.pad(ims[2],((512-200,0),(0,512-200)),'constant',constant_values=np.min(ims[2]))
        ims[3] = data[1848:,1848:]
        ims[3] = np.pad(ims[3],((512-200,0),(512-200,0)),'constant',constant_values=np.min(ims[3]))
        #ims[2] = data[1536:,0:512]
        #ims[3] = data[1536:,1536:]

        shifts = {}
        shifts[0] = data[0:200,0:200]
        shifts[1] = data[0:200,1848:]
        shifts[2] = data[1848:,0:200]
        shifts[3] = data[1848:,1848:]
        #transforms = {}
        #for i in range(4):
        #    shifts[i], transforms[i] = self.get_transform(ims[i])
            




        #contrast = np.zeros(4)
        #for i in range(4):
        #    contrast[i] = self.get_contrast(shifts[i])['avg']
    
        #translation = np.zeros((4,2))
        #rotation = np.zeros(4)
        #for i in range(4):
        #    rotation[i] = transforms[i]['theta']
        #    translation[i,:] = transforms[i]['translation']

        output = {}
        output['shifts'] = shifts
        #output['contrast'] = contrast
        #output['rotation'] = rotation
        #output['translation'] = translation
        
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
        theta1 = np.linspace(0,np.pi/2,90)
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
        theta_offset = theta1[peak[0],0]*180/np.pi
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
        #self.circ2 = QtWidgets.QGraphicsEllipseItem(1024-25,1024-25,50,50)
        #self.circ2.setPen(QPen(Qt.red, 8, Qt.SolidLine))
        self.view0.addItem(rect1)
        self.view0.addItem(rect2)
        self.view0.addItem(rect3)
        self.view0.addItem(rect4)
        #self.view0.addItem(circ1)
        #self.view0.addItem(self.circ2)

        # Top left image
        self.view1 = self.canvas.addViewBox(row=0,col=3,rowspan=1,colspan=1)
        self.view1.setAspectLocked(True)
        self.view1.setRange(QtCore.QRectF(0,0, 200, 200))
        self.img1 = pg.ImageItem(border='w',title='Top Left')
        self.view1.addItem(self.img1)
        recttl = QtWidgets.QGraphicsRectItem(0, 0, 200, 200)
        recttl.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        self.view1.addItem(recttl)


        # Top right image
        self.view2 = self.canvas.addViewBox(row=0,col=4,rowspan=1,colspan=1)
        self.view2.setAspectLocked(True)
        self.view2.setRange(QtCore.QRectF(0,0, 200, 200))
        self.img2 = pg.ImageItem(border='w',title='Top Right')
        self.view2.addItem(self.img2)
        recttr = QtWidgets.QGraphicsRectItem(0, 0, 200, 200)
        recttr.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        self.view2.addItem(recttr)

        # Bottom left image
        self.view3 = self.canvas.addViewBox(row=1,col=3,rowspan=1,colspan=1)
        self.view3.setAspectLocked(True)
        self.view3.setRange(QtCore.QRectF(0,0, 200, 200))
        self.img3 = pg.ImageItem(border='w',title='Bottom Left')
        self.view3.addItem(self.img3)
        rectbl = QtWidgets.QGraphicsRectItem(0, 0, 200, 200)
        rectbl.setPen(QPen(Qt.cyan, 2, Qt.SolidLine))
        self.view3.addItem(rectbl)

        # Bottom right image
        self.view4 = self.canvas.addViewBox(row=1,col=4,rowspan=1,colspan=1)
        self.view4.setAspectLocked(True)
        self.view4.setRange(QtCore.QRectF(0,0, 200, 200))
        self.img4 = pg.ImageItem(border='w',title='Bottom Right')
        self.view4.addItem(self.img4)
        rectbr = QtWidgets.QGraphicsRectItem(0, 0, 200, 200)
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
        im1 = np.array(imageio.imread("PPM_alignment/IM1K1.png"),dtype='float')
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
            self.minValue.setText(str(self.minimum))
            self.maxValue.setText(str(self.maximum))
             

        self.data_dict = data_dict
        self.img0.setImage(np.flipud(data_dict['im1']).T,
                levels=(self.minimum, self.maximum))
        #self.circ2.setRect(data_dict['centering'][0]-25,data_dict['centering'][1]-25,50,50)
        self.img1.setImage(np.flipud(data_dict['shifts'][0]).T, 
                levels=(self.minimum, self.maximum))
        self.img2.setImage(np.flipud(data_dict['shifts'][1]).T,
                levels=(self.minimum, self.maximum))
        self.img3.setImage(np.flipud(data_dict['shifts'][2]).T,
                levels=(self.minimum, self.maximum))
        self.img4.setImage(np.flipud(data_dict['shifts'][3]).T,
                levels=(self.minimum, self.maximum))

        iteration = data_dict['iteration']
        #contrast = data_dict['contrast']
        #rotation = data_dict['rotation']

        #for i in range(4):
        #    self.hplot[i].setData(iteration[i, :], contrast[i, :])

        #    self.rplot[i].setData(iteration[i, :], rotation[i, :])

        #self.contrast_plot.setYRange(0, 1.5)
        #self.contrast_plot.setXRange(np.min(iteration), np.max(iteration))
        #self.rotation_plot.setXRange(np.min(iteration), np.max(iteration))
        #self.rotation_plot.setYRange(-4,4)
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
            self.epics_name = sys.argv[1]


        try:
            self.gige = PCDSAreaDetector(self.epics_name, name='gige')
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

    def reset_camera(self):
        try:
            self.gige.cam.acquire.put(0, wait=True)
            self.gige.cam.acquire.put(1)
        except:
            print('no camera')

    def get_image(self):
        try:
            return np.array(self.gige.image1.image,dtype='float')
        except:
            return np.zeros((2048,2048))

    def _update(self):

        if self.running:

            if self.epics_name != '':

                #self.im1 = np.ones((2048,2048))*255
                self.im1 = self.get_image()

            else:
                if self.counter <= 10:
                    self.im1 = interpolate.rotate(self.im0,.1*self.counter,reshape=False,mode='nearest')
                    self.im1 = ndimage.filters.gaussian_filter(self.im1,3-self.counter*.3)
                else:
                    self.im1 = self.im0

            #self.gui.img0.setImage(np.flipud(self.im1).T,levels=(0,300))

            #self.data_dict['contrast'] = np.roll(self.data_dict['contrast'],-1,axis=1)
            #self.data_dict['rotation'] = np.roll(self.data_dict['rotation'],-1,axis=1)
            self.data_dict['iteration'] = np.roll(self.data_dict['iteration'],-1,axis=1)
            self.data_dict['iteration'][:,-1] = self.counter

            alignment_output = self.yag1.check_alignment(self.im1)

            #self.data_dict['contrast'][:,-1] = alignment_output['contrast']
            
            #translation = alignment_output['translation']

            #centering_x = 1024 + np.sum(translation[:,1])/4
            #centering_y = 1024 + np.sum(translation[:,0])/4
            #self.data_dict['centering'] = np.array([centering_x,centering_y])

            #rotation1 = (translation[1,1]-translation[3,1])/(translation[3,0]-translation[1,0])
            #rotation2 = (translation[3,0]-translation[2,0])/(translation[3,1]-translation[2,1])
            #self.data_dict['rotation'][:,-1] = (rotation1+rotation2)*180/np.pi/2

            #self.data_dict['rotation'][:,-1] = alignment_output['rotation']

            #data_dict = {}
            self.data_dict['im1'] = self.im1
            self.data_dict['shifts'] = alignment_output['shifts']
            self.data_dict['counter'] = self.counter

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
