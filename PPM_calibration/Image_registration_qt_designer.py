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
from PyQt5.uic import loadUiType

Ui_MainWindow, QMainWindow = loadUiType('image_register.ui')

class YagAlign:
    
    def __init__(self):
        
        # the TEMPLATE
        im0 = imageio.imread("pattern.png")[:,:,3]
        Nt, Mt = np.shape(im0)
        im0 = np.pad(im0,((int((512-Nt)/2+1),int((512-Nt)/2)),(int((512-Mt)/2+1),int((512-Mt)/2))),mode='constant')
        self.Nt,self.Mt = np.shape(im0)
        
        self.im0 = 255.0-im0
        
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
        im1 = data[0:200,0:200]
        im1 = np.pad(im1,((512-200,0),(512-200,0)),'constant',constant_values=255)
        #im1[:,125:] = 255
        im1 = data[0:512,0:512]
        im2 = data[0:512,1536:]
        im3 = data[1536:,0:512]
        im4 = data[1536:,1536:]
        
        shift1, transform1 = self.get_transform(im1)
        shift2, transform2 = self.get_transform(im2)
        shift3, transform3 = self.get_transform(im3)
        shift4, transform4 = self.get_transform(im4)
        contrast1 = self.get_contrast(shift1)['avg']
        contrast2 = self.get_contrast(shift2)['avg']
        contrast3 = self.get_contrast(shift3)['avg']
        contrast4 = self.get_contrast(shift4)['avg']
        
        output = {}
        output['shift1'] = shift1
        output['shift2'] = shift2
        output['shift3'] = shift3
        output['shift4'] = shift4
        output['contrast1'] = contrast1
        output['contrast2'] = contrast2
        output['contrast3'] = contrast3
        output['contrast4'] = contrast4

        
        return output
        
    @staticmethod
    def get_contrast(img):
        line1 = np.mean(img[220:250,220:250],axis=1)
        line2 = np.mean(img[220:250,259:290],axis=0)
        line3 = np.mean(img[260:290,220:250],axis=0)
        line4 = np.mean(img[260:290,260:290],axis=1)

        #norm = np.mean(img[310:370,220:295])
        background = np.mean(img[310:370,220:295])
        r1 = np.std(line1/norm)*np.sqrt(2)*2
        r2 = np.std(line2/norm)*np.sqrt(2)*2
        r3 = np.std(line3/norm)*np.sqrt(2)*2
        r4 = np.std(line4/norm)*np.sqrt(2)*2
        
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
        
#         plt.figure()
#         plt.imshow(cps_real)
        
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
        
#         shifted = shifted[210:302,210:302]
        
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

        #### Create Gui Elements ###########

        #self.mainbox = QtGui.QWidget()
        #self.setCentralWidget(self.mainbox)
        #self.mainbox.setLayout(QtGui.QVBoxLayout())
        #self.mainbox.setFixedSize(1200,800)


        self.runButton.clicked.connect(self.change_state)

        #self.canvas = pg.GraphicsLayoutWidget()
        #self.mainbox.layout().addWidget(self.canvas)
        #self.qtgraphWidget.layout().addWidget

        #self.label = QtGui.QLabel()
        #self.centralwidget.layout().addWidget(self.label)

        self.view0 = self.canvas.addViewBox(row=0,col=0,rowspan=2,colspan=2)
        self.view0.setAspectLocked(True)
        self.view0.setRange(QtCore.QRectF(0,0, 2048, 2048))

        self.view1 = self.canvas.addViewBox(row=0,col=2,rowspan=1,colspan=1)
        self.view1.setAspectLocked(True)
        self.view1.setRange(QtCore.QRectF(0,0, 90, 90))


        self.view2 = self.canvas.addViewBox(row=0,col=3,rowspan=1,colspan=1)
        self.view2.setAspectLocked(True)
        self.view2.setRange(QtCore.QRectF(0,0, 90, 90))

        self.view3 = self.canvas.addViewBox(row=1,col=2,rowspan=1,colspan=1)
        self.view3.setAspectLocked(True)
        self.view3.setRange(QtCore.QRectF(0,0, 90, 90))

        self.view4 = self.canvas.addViewBox(row=1,col=3,rowspan=1,colspan=1)
        self.view4.setAspectLocked(True)
        self.view4.setRange(QtCore.QRectF(0,0, 90, 90))

        #  image plot
        self.img0 = pg.ImageItem(border='w')
        self.view0.addItem(self.img0)
        rect1 = QtWidgets.QGraphicsRectItem(0,0,125,125)
        rect1.setPen(QPen(Qt.cyan, 8, Qt.SolidLine))
        rect2 = QtWidgets.QGraphicsRectItem(1923,0,125,125)
        rect2.setPen(QPen(Qt.darkMagenta, 8, Qt.SolidLine))
        rect3 = QtWidgets.QGraphicsRectItem(0,1923,125,125)
        rect3.setPen(QPen(Qt.red, 8, Qt.SolidLine))
        rect4 = QtWidgets.QGraphicsRectItem(1923,1923,125,125)
        rect4.setPen(QPen(Qt.green, 8, Qt.SolidLine))
        self.view0.addItem(rect1)
        self.view0.addItem(rect2)
        self.view0.addItem(rect3)
        self.view0.addItem(rect4)
        #self.view0.addItem(QtWidgets.QGraphicsRectItem(0,0,125,125))
        #self.view0.addItem(QtWidgets.QGraphicsRectItem(1923,0,125,125))
        #self.view0.addItem(QtWidgets.QGraphicsRectItem(0,1923,125,125))
        #self.view0.addItem(QtWidgets.QGraphicsRectItem(1923,1923,125,125))

        self.img1 = pg.ImageItem(border='w',title='Top Left')
        self.view1.addItem(self.img1)
        recttl = QtWidgets.QGraphicsRectItem(0, 0, 90, 90)
        recttl.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        self.view1.addItem(recttl)

        self.img2 = pg.ImageItem(border='w',title='Top Right')
        self.view2.addItem(self.img2)
        recttr = QtWidgets.QGraphicsRectItem(0, 0, 90, 90)
        recttr.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        self.view2.addItem(recttr)

        self.img3 = pg.ImageItem(border='w',title='Bottom Left')
        self.view3.addItem(self.img3)
        rectbl = QtWidgets.QGraphicsRectItem(0, 0, 90, 90)
        rectbl.setPen(QPen(Qt.cyan, 2, Qt.SolidLine))
        self.view3.addItem(rectbl)

        self.img4 = pg.ImageItem(border='w',title='Bottom Right')
        self.view4.addItem(self.img4)
        rectbr = QtWidgets.QGraphicsRectItem(0, 0, 90, 90)
        rectbr.setPen(QPen(Qt.darkMagenta, 2, Qt.SolidLine))
        self.view4.addItem(rectbr)

        #self.canvas.nextRow()
        #  line plot
        self.contrast_plot = self.canvas.addPlot(row=2,col=0,rowspan=1,colspan=4)

        self.contrast_plot.addLegend()
        self.contrast_plot.showGrid(x=True,y=True,alpha=.8)
        self.contrast_plot.setYRange(0,1.5)

        self.hplot = {}

        self.yag1 = YagAlign()

        # the image to be transformed
        im1 = imageio.imread("test_pattern.png")[32:2650, 32:2650, 3]
        im1 = 255 - im1

        #im1 = np.flipud(im1)

        N, M = np.shape(im1)
        scale = 2048.0 / N

        im0 = interpolate.zoom(im1, scale)

        names = ['Top Left','Top Right','Bottom Left','Bottom Right']

        for i in range(4):

            self.hplot[i] = self.contrast_plot.plot(np.linspace(-99,0,100),np.zeros(100),pen=(i,4),name=names[i])

        self.data_dict = {}
        self.data_dict['im0'] = im0
        self.data_dict['contrast'] = np.zeros((4, 100))
        self.data_dict['iteration'] = np.tile(np.linspace(-99, 0, 100), (4, 1))
        self.data_dict['counter'] = 0.



    def change_state(self):
        if self.runButton.text() == 'Run':

            #self.thread._update()
            self.registration = RunRegistration(self.yag1, self.data_dict)
            self.thread = QtCore.QThread()
            self.thread.start()

            self.registration.moveToThread(self.thread)
            self.runButton.setText('Stop')

            self.registration.sig.connect(self.update_plots)
            #self.connect(self.registration_thread.sig, QtCore.SIGNAL('update_plots(PyQt_PyObject)', dict), self.update_plots)

        elif self.runButton.text() == 'Stop':

            self.registration.stop()
            self.thread.quit()
            self.thread.wait()
            #self.registration_thread.quit()
            #self.emit(QtCore.SIGNAL('stop()'))
            self.runButton.setText('Run')


    def update_plots(self,data_dict):
        self.data_dict = data_dict
        self.img0.setImage(np.flipud(data_dict['im1']).T, levels=(0, 300))
        self.img1.setImage(np.flipud(data_dict['shift1'][210:300, 210:300]).T, levels=(0, 270))
        self.img2.setImage(np.flipud(data_dict['shift2'][210:300, 210:300]).T, levels=(0, 270))
        self.img3.setImage(np.flipud(data_dict['shift3'][210:300, 210:300]).T, levels=(0, 270))
        self.img4.setImage(np.flipud(data_dict['shift4'][210:300, 210:300]).T, levels=(0, 270))

        iteration = data_dict['iteration']
        contrast = data_dict['contrast']

        for i in range(4):
            self.hplot[i].setData(iteration[i, :], contrast[i, :])

        self.contrast_plot.setYRange(0, 1.5)
        self.contrast_plot.setXRange(np.min(iteration), np.max(iteration))

        self.label.setText(data_dict['tx'])

class RunRegistration(QtCore.QObject):

    sig = QtCore.pyqtSignal(dict)

    def __init__(self, yag, data_dict):
        super(RunRegistration, self).__init__()

        #self.my_signal = QtCore.Signal()

        #self.gui = gui

        self.yag1 = yag

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

    def _update(self):

        if self.running:

            if self.counter <= 10:
                self.im1 = interpolate.rotate(self.im0,.1*self.counter,reshape=False,mode='nearest')
                self.im1 = ndimage.filters.gaussian_filter(self.im1,3-self.counter*.3)
            else:
                self.im1 = self.im0

            #self.gui.img0.setImage(np.flipud(self.im1).T,levels=(0,300))

            self.data_dict['contrast'] = np.roll(self.data_dict['contrast'],-1,axis=1)
            self.data_dict['iteration'] = np.roll(self.data_dict['iteration'],-1,axis=1)

            self.data_dict['iteration'][:,-1] = self.counter

            alignment_output = self.yag1.check_alignment(self.im1)
            # alignment_output = {}
            # alignment_output['contrast1'] = 0
            # alignment_output['contrast2'] = 0
            # alignment_output['contrast3'] = 0
            # alignment_output['contrast4'] = 0
            # alignment_output['shift1'] = np.zeros((500,500))
            # alignment_output['shift2'] = np.zeros((500, 500))
            # alignment_output['shift3'] = np.zeros((500, 500))
            # alignment_output['shift4'] = np.zeros((500, 500))

            self.data_dict['contrast'][0,-1] = alignment_output['contrast1']
            self.data_dict['contrast'][1,-1] = alignment_output['contrast2']
            self.data_dict['contrast'][2,-1] = alignment_output['contrast3']
            self.data_dict['contrast'][3,-1] = alignment_output['contrast4']

            #data_dict = {}
            self.data_dict['im1'] = self.im1
            self.data_dict['shift1'] = alignment_output['shift1']
            self.data_dict['shift2'] = alignment_output['shift2']
            self.data_dict['shift3'] = alignment_output['shift3']
            self.data_dict['shift4'] = alignment_output['shift4']
            self.data_dict['counter'] = self.counter




            #self.hplot[0].setData(self.iteration[0,:],self.contrast[0,:])

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

            #self.gui.label.setText(tx)
            if self.running:
                QtCore.QTimer.singleShot(100, self._update)
                self.counter += 1


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())
