from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from PyQt5.uic import loadUiType
import numpy as np
from datetime import datetime
from matplotlib import cm
from PyQt5.QtGui import QPen


Ui_LineoutImage, QLineoutImage = loadUiType('LineoutImage.ui')
Ui_Crosshair, QCrosshair = loadUiType('Crosshair.ui')
Ui_LevelsWidget, QLevelsWidget = loadUiType('LevelsWidget.ui')


class LineoutImage(QLineoutImage, Ui_LineoutImage):
    """
    Class to represent a widget containing an image with horizontal and vertical lineouts. Linked to LineoutImage.ui.
    """
    def __init__(self, parent=None):
        """
        Initialize the widget.
        :param parent:
        """
        super(LineoutImage, self).__init__()
        self.setupUi(self)

        # initialize levels widget attribute
        self.levels = None
        # initialize crosshair widget attribute
        self.crosshairsWidget = None

        # set default image levels
        self.minimum = 0
        self.maximum = 4096

        # add viewbox for image
        self.view = self.image_canvas.addViewBox()

        # add colormap to context menu
        colormapMenu = self.view.menu.addMenu("Colormap")
        gnuplot = colormapMenu.addAction("gnuplot")
        grayscale = colormapMenu.addAction("grayscale")
        viridis = colormapMenu.addAction("viridis")
        bwr = colormapMenu.addAction("BlueWhiteRed")
        hsv = colormapMenu.addAction("hsv")

        # connect callbacks
        gnuplot.triggered.connect(self.set_gnuplot)
        grayscale.triggered.connect(self.set_grayscale)
        viridis.triggered.connect(self.set_viridis)
        bwr.triggered.connect(self.set_bwr)
        hsv.triggered.connect(self.set_hsv)

        # define colormaps
        self.colormaps = self.define_colormaps()

        # setup viewbox and get corresponding QRect
        self.rect = self.setup_viewbox(1024)
       
        # lock aspect ratio
        self.view.setAspectLocked(True)
        # add an image
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        # set default colormap
        self.img.setLookupTable(self.colormaps['gnuplot'])

        # font styles
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.font = QtGui.QFont()
        self.font.setPointSize(10)
        self.font.setFamily('Arial')

        # initialize lineouts
        (self.horizontalPlot,
         self.horizontalLineout,
         self.horizontalFit) = self.initialize_lineout(self.xlineout_canvas, 'horizontal')
        (self.verticalPlot,
         self.verticalLineout,
         self.verticalFit) = self.initialize_lineout(self.ylineout_canvas, 'vertical')

    def get_lut(self, name):
        cmap = cm.get_cmap(name)
        cmap._init()
        lut = (cmap._lut * 255).view(np.ndarray)

        return lut

    def define_colormaps(self):

        cmap_dict = {}

        cmap_dict['grayscale'] = self.get_lut("gray")
        cmap_dict['gnuplot'] = self.get_lut("gnuplot")
        cmap_dict['viridis'] = self.get_lut("viridis")
        cmap_dict['bwr'] = self.get_lut("bwr")
        cmap_dict['hsv'] = self.get_lut("hsv")

        return cmap_dict
        
    def set_viridis(self, evt):
        self.img.setLookupTable(self.colormaps['viridis'])

    def set_hsv(self, evt):
        self.img.setLookupTable(self.colormaps['hsv'])

    def set_bwr(self, evt):
        self.img.setLookupTable(self.colormaps['bwr'])
    
    def set_grayscale(self, evt):

        self.img.setLookupTable(self.colormaps['grayscale'])

    def set_gnuplot(self, evt):
        
        self.img.setLookupTable(self.colormaps['gnuplot'])

    def connect_crosshairs(self, crosshairs):
        """
        Method to connect a CrosshairWidget.
        :param crosshairs: CrosshairWidget
        :return:
        """
        # set attribute
        self.crosshairsWidget = crosshairs

        # setup crosshairs
        self.crosshairsWidget.connect_image(self)

        self.rect.scene().sigMouseClicked.connect(self.mouseClicked)


    def mouseClicked(self, evt):
        """
        Method to define new crosshair location based on mouseclick.
        :param evt: mouse click event
            Contains scene position
        :return:
        """

        # translate scene coordinates to viewbox coordinates
        coords = self.view.mapSceneToView(evt.scenePos())

        # update crosshair
        self.crosshairsWidget.update_crosshair_coords(coords)

    def connect_levels(self, levels):
        """
        Method to connect a Levels widget for scaling the image.
        :param levels: LevelsWidget
        :return:
        """
        # set attribute
        self.levels = levels

        # set levels based on current entries
        self.set_min()
        self.set_max()
        # connect line edit return to set_min, set_max methods
        self.levels.minLineEdit.returnPressed.connect(self.set_min)
        self.levels.maxLineEdit.returnPressed.connect(self.set_max)

    def get_canvases(self):
        """
        Method to give access to GraphicsLayoutWidgets
        :return:
        """
        return self.image_canvas, self.xlineout_canvas, self.ylineout_canvas

    def update_plots(self, image_data, x, y, xlineout_data, ylineout_data, fit_x, fit_y):
        """
        Method to update image, lineout plots
        :param image_data: (N, M) ndarray
            array corresponding to image data to display
        :param x: (M) ndarray
            1D array defining x axis coordinates
        :param y: (N) ndarray
            1D array defining y axis coordinates
        :param xlineout_data: (M) ndarray
            1D array containing horizontal image lineout
        :param ylineout_data: (N) ndarray
            1D array containing vertical image lineout
        :param fit_x: (M) ndarray
            1D array containing gaussian fit to horizontal lineout
        :param fit_y: (N) ndarray
            1D array containing gaussian fit to vertical lineout
        :return:
        """
        # check if there is an associated levels widget
        if self.levels is not None:
            # check if we're autoscaling
            if self.levels.checkBox.isChecked():
                self.minimum = np.min(image_data)
                self.maximum = np.max(image_data)
                # set text on levels widget
                self.levels.setText(self.minimum, self.maximum)
        else:
            # autoscale if there is no levels widget
            self.minimum = np.min(image_data)
            self.maximum = np.max(image_data)

        # figure out image extent based on coordinates
        x_width = np.max(x) - np.min(x)
        y_width = np.max(y) - np.min(y)

        # set image data
        self.img.setImage(np.flipud(image_data).T,
                levels=(self.minimum, self.maximum))

        # set rect size based on coordinates
        self.img.setRect(QtCore.QRectF(np.min(x),np.min(y),x_width, y_width))

        # set lineout data
        self.horizontalLineout.setData(x, xlineout_data)
        self.horizontalFit.setData(x, fit_x)
        self.verticalLineout.setData(ylineout_data, y)
        self.verticalFit.setData(fit_y, y)

    def set_min(self):
        """
        Method called when return is pressed on levels.minLineEdit.
        :return:
        """
        # update the minimum to the new value
        self.minimum = float(self.levels.minLineEdit.text())

    def set_max(self):
        """
        Method called when return is pressed on levels.maxLineEdit.
        :return:
        """
        # update the maximum to the new value
        self.maximum = float(self.levels.maxLineEdit.text())

    def setup_viewbox(self, width):
        """
        Helper function to set up viewbox with title
        :param width: image width in pixels (int)
        """
        # lock aspect ratio
        self.view.setAspectLocked(True)
        # update viewbox range
        self.view.setRange(QtCore.QRectF(-width/2., -width/2., width, width))
        # draw a white rectangle that is the same size as the image to show the image boundary
        rect1 = QtGui.QGraphicsRectItem(-width/2., -width/2., width, width)
        rect1.setPen(QtGui.QPen(QtCore.Qt.white, width/50., QtCore.Qt.SolidLine))
        # add the rectangle to the viewbox
        self.view.addItem(rect1)
        # return the rectangle
        return rect1
        
    def update_viewbox(self, width, height):
        """
        Helper function to adjust viewbox settings
        :param width: new width in pixels (int)
        :param height: new height in pixels (int)
        :return:
        """
        # set range to new size
        self.view.setRange(QtCore.QRectF(-width/2, -height/2, width, height))
        # update the bounding rectangle
        self.rect.setPen(QtGui.QPen(QtCore.Qt.white, width/50., QtCore.Qt.SolidLine))
        self.rect.setRect(-width/2, -height/2, width, height)

    def change_lineout_label(self, ylabel):
        """
        Method to change the "y-axis" label on the lineouts
        :param ylabel: str
            New label for lineout y-axis
        :return:
        """
        # update lineout labels
        PlotUtil.label_plot(self.horizontalPlot, u'x (\u03BCm)', ylabel)
        PlotUtil.label_plot(self.verticalPlot, ylabel, u'y (\u03BCm)')

    def initialize_lineout(self, canvas, direction):
        """
        Method to set up lineout plots.
        :param canvas: pg.GraphicsLayoutWidget
            Layout widget used for adding pyqtgraph widgets
        :param direction: str
            'horizontal' or 'vertical': direction of the lineout
        """
        # legend names
        names = ['Lineout', 'Fit']
        # line colors
        colors = ['r', 'c']

        # add plot to canvas
        if direction == 'horizontal':
            # horizontal lineout
            lineoutPlot = canvas.addPlot()
            # initialize legend and adjust position
            legend = lineoutPlot.addLegend(offset=(10,0))
            # initialize lineout plot
            lineoutData = lineoutPlot.plot(np.linspace(-1024, 1023, 100), np.zeros(100),
                                           pen=pg.mkPen(colors[0], width=2),name=names[0])
            # initialize fit plot
            lineoutFit = lineoutPlot.plot(np.linspace(-1024, 1023, 100), np.zeros(100),
                                           pen=pg.mkPen(colors[1], width=2),name=names[1])

            # add legend
            PlotUtil.setup_legend(legend)

            # set range to be normalized
            lineoutPlot.setYRange(0, 1)

            # plot labels
            PlotUtil.label_plot(lineoutPlot, u'x (\u03BCm)', 'Intensity')
            # link axis to image
            lineoutPlot.setXLink(self.view)

        elif direction == 'vertical':
            # vertical lineout
            lineoutPlot = canvas.addPlot()
            # initialize lineout plot
            lineoutData = lineoutPlot.plot(np.zeros(100), np.linspace(-1024, 1023, 100),
                                           pen=pg.mkPen(colors[0], width=2),name=names[0])
            # initialize fit plot
            lineoutFit = lineoutPlot.plot(np.zeros(100), np.linspace(-1024, 1023, 100),
                                           pen=pg.mkPen(colors[1], width=2),name=names[1])

            # set range to be normalized
            lineoutPlot.setXRange(0, 1)
            # plot labels
            PlotUtil.label_plot(lineoutPlot, 'Intensity', u'y (\u03BCm)')
            # link axis to image
            lineoutPlot.setYLink(self.view)
        else:
            # just to catch anything weird
            lineoutPlot = None
            lineoutData = None
            lineoutFit = None

        # return the plot widget and line plots
        return lineoutPlot, lineoutData, lineoutFit


class PlotUtil:
    """
    Utility class for PPM widgets. Contains only static methods.
    """

    labelStyle = {'color': '#FFF', 'font-size': '10pt'}
    font = QtGui.QFont()
    font.setPointSize(10)
    font.setFamily('Arial')

    @staticmethod
    def setup_legend(legend):
        """
        Method for setting legend style
        :param legend: pg.LegendItem
            legend that needs formatting
        :return:
        """

        # set style
        legendLabelStyle = {'color': '#FFF', 'size': '10pt'}
        # loop through legend items
        for item in legend.items:
           for single_item in item:
               # set style
               if isinstance(single_item, pg.graphicsItems.LabelItem.LabelItem):
                   single_item.setText(single_item.text, **legendLabelStyle)

    @staticmethod
    def label_plot(plot, xlabel, ylabel):
        """
        Helper function to set plot labels
        :param plot: pyqtgraph plot item
        :param xlabel: str
            x-axis label
        :param ylabel: str
            y-axis label
        """
        # label x-axis
        xaxis = plot.getAxis('bottom')
        PlotUtil.set_axislabel(xaxis, xlabel, 'w', 1)

        # label y-axis
        yaxis = plot.getAxis('left')
        PlotUtil.set_axislabel(yaxis, ylabel, 'w', 1)

    @staticmethod
    def set_axislabel(axis, text, color, width):
        """
        Convenience method for axis labeling
        :param axis: pyqtgraph axis item
            axis being labeled
        :param text: str
            label text
        :param color: str
            character corresponding to QtPen color
        :param width: int
            width of pen
        :return:
        """
        # set label
        axis.setLabel(text=text, **PlotUtil.labelStyle)
        # set font
        axis.tickFont = PlotUtil.font
        # set pen color and size
        axis.setPen(pg.mkPen(color, width=width))


class ImageBase:

    def __init__(self, canvas):
        self.view = canvas.addViewBox()
        self.view.setAspectLocked(True)


class ImageZoom:
    """
    Class for displaying a zoomed-in image
    """

    def __init__(self, canvas, color):

        self.view = canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0,0, 90, 90))
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)
        rect = QtWidgets.QGraphicsRectItem(0, 0, 90, 90)
        #rect.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        rect.setPen(pg.mkPen(color, width=2))
        self.view.addItem(rect)

        self.levels = None
        self.minimum = 0
        self.maximum = 4096

    def update_image(self, image_data):

        self.img.setImage(np.flipud(image_data).T, levels=(self.minimum, self.maximum))
   
    def connect_levels(self, levels):
        """
        Method to connect a Levels widget for scaling the image.
        :param levels: LevelsWidget
        :return:
        """
        # set attribute
        self.levels = levels

        # set levels based on current entries
        self.set_min()
        self.set_max()
        # connect line edit return to set_min, set_max methods
        self.levels.minLineEdit.returnPressed.connect(self.set_min)
        self.levels.maxLineEdit.returnPressed.connect(self.set_max)
   
    def set_min(self):
        """
        Method called when return is pressed on levels.minLineEdit.
        :return:
        """
        # update the minimum to the new value
        self.minimum = float(self.levels.minLineEdit.text())

    def set_max(self):
        """
        Method called when return is pressed on levels.maxLineEdit.
        :return:
        """
        # update the maximum to the new value
        self.maximum = float(self.levels.maxLineEdit.text())


class ImageRegister:
    """
    Class for displaying image registration screen
    """

    def __init__(self, canvas):

        self.canvas = canvas

        # Full image
        self.view = self.canvas.addViewBox()

        width = 1024

        self.levels = None

        self.rect = self.setup_viewbox(1024) 

        self.view.setAspectLocked(True)
        
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)

        self.rect1 = QtWidgets.QGraphicsRectItem(0,0,160,160)
        self.rect1.setPen(QPen(Qt.cyan, 8, Qt.SolidLine))
        self.rect2 = QtWidgets.QGraphicsRectItem(1888,0,160,160)
        self.rect2.setPen(QPen(Qt.darkMagenta, 8, Qt.SolidLine))
        self.rect3 = QtWidgets.QGraphicsRectItem(0,1888,160,160)
        self.rect3.setPen(QPen(Qt.red, 8, Qt.SolidLine))
        self.rect4 = QtWidgets.QGraphicsRectItem(1888,1888,160,160)
        self.rect4.setPen(QPen(Qt.green, 8, Qt.SolidLine))


        #circ1 = QtWidgets.QGraphicsEllipseItem(1024-25,1024-25,50,50)
        #circ1.setPen(QPen(Qt.green, 8, Qt.SolidLine))
        self.crossx0 = QtWidgets.QGraphicsLineItem(1024-25,1024,1024+25,1024)
        self.crossy0 = QtWidgets.QGraphicsLineItem(1024,1024-25,1024,1024+25)
        self.crossx0.setPen(QPen(Qt.green, 8, Qt.SolidLine))
        self.crossy0.setPen(QPen(Qt.green, 8, Qt.SolidLine))

        
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
        self.view.addItem(self.rect1)
        self.view.addItem(self.rect2)
        self.view.addItem(self.rect3)
        self.view.addItem(self.rect4)
        #self.view0.addItem(circ1)
        self.view.addItem(self.crossx0)
        self.view.addItem(self.crossy0)
        self.view.addItem(self.crossx)
        self.view.addItem(self.crossy)
        #self.view0.addItem(self.circ0)
        self.view.addItem(self.circ1)
        self.view.addItem(self.circ2)
        self.view.addItem(self.circ3)
        self.view.addItem(self.circ4)



        self.pix_size_text = pg.TextItem('Pixel size: %.2f microns' % 0.0,
                color=(200,200,200), border='c', fill='b',anchor=(0,1))
        self.pix_size_text.setFont(QtGui.QFont("", 10, QtGui.QFont.Bold))
        self.pix_size_text.setPos(-width/2+width/7,-width/2+width/2048)
        self.view.addItem(self.pix_size_text)

        self.update_viewbox(1024,1024)

        self.minimum = 0
        self.maximum = 4096

    def connect_levels(self, levels):
        """
        Method to connect a Levels widget for scaling the image.
        :param levels: LevelsWidget
        :return:
        """
        # set attribute
        self.levels = levels

        # set levels based on current entries
        self.set_min()
        self.set_max()
        # connect line edit return to set_min, set_max methods
        self.levels.minLineEdit.returnPressed.connect(self.set_min)
        self.levels.maxLineEdit.returnPressed.connect(self.set_max)
   
    def set_min(self):
        """
        Method called when return is pressed on levels.minLineEdit.
        :return:
        """
        # update the minimum to the new value
        self.minimum = float(self.levels.minLineEdit.text())

    def set_max(self):
        """
        Method called when return is pressed on levels.maxLineEdit.
        :return:
        """
        # update the maximum to the new value
        self.maximum = float(self.levels.maxLineEdit.text())


    def update_image(self, image_data, pixSize, center=None, scale=None):
        if self.levels is not None:
            # check if we're autoscaling
            if self.levels.checkBox.isChecked():
                self.minimum = np.min(image_data)
                self.maximum = np.max(image_data)
                # set text on levels widget
                self.levels.setText(self.minimum, self.maximum)
        else:
            # autoscale if there is no levels widget
            self.minimum = np.min(image_data)
            self.maximum = np.max(image_data)
    
        width = self.rect.rect().width()
        height = self.rect.rect().height()
        self.img.setImage(np.flipud(image_data).T,
                levels=(self.minimum, self.maximum))

        self.img.setRect(QtCore.QRectF(-width/2, -height/2, width, height))
       
       

        if center is not None:

            center = center - width/2
            rwidth = width/45
            self.circ1.setRect(center[0,1]-scale[0]*rwidth,center[0,0]-scale[0]*rwidth,
                2*rwidth*scale[0],2*rwidth*scale[0])
            self.circ2.setRect(center[1,1]-scale[1]*rwidth,center[1,0]-scale[1]*rwidth,
                2*rwidth*scale[1],2*rwidth*scale[1])
            self.circ3.setRect(center[2,1]-scale[2]*rwidth,center[2,0]-scale[2]*rwidth,
                2*rwidth*scale[2],2*rwidth*scale[2])
            self.circ4.setRect(center[3,1]-scale[3]*rwidth,center[3,0]-scale[3]*rwidth,
                2*rwidth*scale[3],2*rwidth*scale[3])


            full_center = np.mean(center,axis=0)

            self.crossx.setLine(full_center[1]-width/80,full_center[0],
                full_center[1]+width/80,full_center[0])
            self.crossy.setLine(full_center[1],full_center[0]-width/80,
                full_center[1],full_center[0]+width/80)


        self.pix_size_text.setText('Pixel size: %.2f microns' 
                % pixSize)




    def setup_viewbox(self, width):
        """
        Helper function to set up viewbox with title
        :param width: image width in pixels (int)
        """
        # lock aspect ratio
        self.view.setAspectLocked(True)
        # update viewbox range
        self.view.setRange(QtCore.QRectF(-width/2., -width/2., width, width))
        # draw a white rectangle that is the same size as the image to show the image boundary
        rect1 = QtGui.QGraphicsRectItem(-width/2., -width/2., width, width)
        rect1.setPen(QtGui.QPen(QtCore.Qt.white, width/50., QtCore.Qt.SolidLine))
        # add the rectangle to the viewbox
        self.view.addItem(rect1)
        # return the rectangle
        return rect1
        
    def update_viewbox(self, width, height):
        """
        Helper function to adjust viewbox settings
        :param width: new width in pixels (int)
        :param height: new height in pixels (int)
        :return:
        """
        # lock aspect ratio
        self.view.setAspectLocked(True)
        
        # set range to new size
        self.view.setRange(QtCore.QRectF(-width/2, -height/2, width, height))
        # update the bounding rectangle
        self.rect.setPen(QtGui.QPen(QtCore.Qt.white, width/256., QtCore.Qt.SolidLine))
        self.rect.setRect(-width/2, -height/2, width, height)
        
        self.rect1.setRect(-width/2,-width/2,width/12, width/12)
        self.rect1.setPen(QPen(Qt.cyan, width/256, Qt.SolidLine))
        self.rect2.setRect(width/2-width/12,-width/2,width/12,width/12)
        self.rect2.setPen(QPen(Qt.darkMagenta, width/256, Qt.SolidLine))
        self.rect3.setRect(-width/2,width/2-width/12,width/12,width/12)
        self.rect3.setPen(QPen(Qt.red, width/256, Qt.SolidLine))
        self.rect4.setRect(width/2-width/12,width/2-width/12,width/12,width/12)
        self.rect4.setPen(QPen(Qt.green, width/256, Qt.SolidLine))


        
        self.crossx0.setLine(-width/80,0,width/80,0)
        self.crossy0.setLine(0,-width/80,0,width/80)
        self.crossx0.setPen(QPen(Qt.green, width/256, Qt.SolidLine))
        self.crossy0.setPen(QPen(Qt.green, width/256, Qt.SolidLine))

        
        self.crossx.setLine(-width/80,0,width/80,0)
        self.crossy.setLine(0,-width/80,0,width/80)
        self.crossx.setPen(QPen(Qt.red, width/256, Qt.SolidLine))
        self.crossy.setPen(QPen(Qt.red, width/256, Qt.SolidLine))

        
        self.circ1.setRect(-width/2,width/2-width/40,width/40,width/40)
        self.circ1.setPen(QPen(Qt.red, width/256, Qt.SolidLine))
        self.circ2.setRect(width/2-width/40,width/2-width/40,width/40,width/40)
        self.circ2.setPen(QPen(Qt.green, width/256, Qt.SolidLine))
        self.circ3.setRect(-width/2,-width/2,width/40,width/40)
        self.circ3.setPen(QPen(Qt.cyan, width/256, Qt.SolidLine))
        self.circ4.setRect(width/2-width/40,-width/2,width/40,width/40)
        self.circ4.setPen(QPen(Qt.darkMagenta, width/256, Qt.SolidLine))
        self.pix_size_text.setPos(-width/2+width/7,-width/2+width/2048)

class StripChart:
    """
    Class for displaying time series data
    """

    def __init__(self, canvas, ylabel):
        """
        Initialize a StripChart
        :param canvas: pg.GraphicsLayoutWidget
            Canvas where the plot will be added. Should not contain any other widgets.
        :param ylabel: str
            label for y-axis
        """
        # font styles
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.font = QtGui.QFont()
        self.font.setPointSize(10)
        self.font.setFamily('Arial')

        # set canvas as attribute
        self.canvas = canvas
        # generate plot
        self.plotWidget = self.canvas.addPlot()

        # label plot axes
        PlotUtil.label_plot(self.plotWidget, 'Time (s)', ylabel)

        # add grid
        self.plotWidget.showGrid(x=True,y=True,alpha=.8)
        # initialize dictionary of lines to plot
        self.lines = {}

        # set color order
        self.color_order = ['r', 'c', 'm', 'g', 'b', 'y']

        # default time range (seconds)
        self.time_range = 10

    def addSeries(self, series_keys, series_labels):
        """
        Method to define plot series. May or may not be ok to call this method more than once...
        :param series_keys: list of strings
            keys for lines dictionary
        :param series_labels: list of strings
            legend labels for lines
        :return:
        """

        # add a LegendItem
        legend = self.plotWidget.addLegend()

        # loop through keys and make PlotItems
        for num, key in enumerate(series_keys):
            # make a PlotItem
            self.lines[key] = self.plotWidget.plot(np.linspace(-99,0,100), np.zeros(100),
                    pen=pg.mkPen(self.color_order[num], width=5),name=series_labels[num])

        # set up the legend
        PlotUtil.setup_legend(legend)

    def set_time_range(self, time_range):
        """
        Method to change the time range for the plot. Could connect this to a callback.
        :param time_range: float
            Time range for the plot in seconds
        :return:
        """
        # set time range attribute
        self.time_range = time_range

    def update_plots(self, time_stamps, **data):
        """
        Method to update the plots with incoming data
        :param time_stamps: (N) ndarray
            image PV timestamps
        :param data: (N) ndarrays
            1D arrays with keywords corresponding to those in lines dict
        :return:
        """

        # filter out any data that doesn't exist yet
        mask = time_stamps > 0

        # get current time
        now = datetime.now()
        now_stamp = datetime.timestamp(now)
        # subtract current time from timestamps
        time_stamps = time_stamps - now_stamp
        # mask out invalid data
        time_stamps = time_stamps[mask]

        # loop through datasets
        for key, value in data.items():
            # filter data with mask
            filtered_value = value[mask]
            try:
                # set plot data
                self.lines[key].setData(time_stamps, filtered_value)
            except KeyError:
                # catch exceptions related to the wrong key
                print('Data had the wrong name')

        # reset plot range
        self.plotWidget.setXRange(-self.time_range, 0)


class CrosshairWidget(QCrosshair, Ui_Crosshair):
    """
    Class to define a crosshair widget. The widget consists of two crosshair buttons and 4 LineEdits corresponding
    to crosshair positions.

    Attributes
    ----------
    redButton: QPushButton
        Button for selecting red crosshair
    blueButton: QPushButton
        Button for selecting blue crosshair
    red_x: QLineEdit
        x position of red crosshair
    red_y: QLineEdit
        y position of red crosshair
    blue_x: QLineEdit
        x position of blue crosshair
    blue_y: QLineEdit
        y position of blue crosshair
    lineout_image: LineoutImage
        image widget to connect to
    red_crosshair: Crosshair
        red crosshair object that is displayed on lineout_image
    blue_crosshair: Crosshair
        blue crosshair object that is displayed on lineout_image
    current_crosshair: Crosshair
        Variable to keep track of the active crosshair. Can have values of None, red_crosshair, or blue_crosshair.
    """

    def __init__(self, parent=None):
        """
        Initialize the widget
        :param parent:
        """
        super(CrosshairWidget, self).__init__()
        self.setupUi(self)

        # initialize attributes
        self.red_crosshair = None
        self.blue_crosshair = None
        self.current_crosshair = None

    def connect_image(self, image_widget):
        """
        Method to connect the crosshair widget to an image
        :param image_widget: LineoutImage
            The image to connect to
        :return:
        """
        # create Crosshair objects
        self.red_crosshair = Crosshair('red', self.red_x, self.red_y, image_widget)
        self.blue_crosshair = Crosshair('blue', self.blue_x, self.blue_y, image_widget)

        # connect callbacks
        # crosshair selection
        self.redButton.toggled.connect(self.red_crosshair_toggled)
        self.blueButton.toggled.connect(self.blue_crosshair_toggled)
        # red crosshair position
        self.red_x.returnPressed.connect(self.update_red_crosshair)
        self.red_y.returnPressed.connect(self.update_red_crosshair)
        # blue crosshair position
        self.blue_x.returnPressed.connect(self.update_blue_crosshair)
        self.blue_y.returnPressed.connect(self.update_blue_crosshair)
        #self.red_x.returnPressed.connect(self.update_crosshair(self.red_crosshair))
        #self.red_y.returnPressed.connect(self.update_crosshair(self.red_crosshair))
        #self.blue_x.returnPressed.connect(self.update_crosshair(self.blue_crosshair))
        #self.blue_y.returnPressed.connect(self.update_crosshair(self.blue_crosshair))

    def calculate_distance(self):

        try:
            red_x = float(self.red_x.text())
        except ValueError:
            red_x = 0
        try:
            red_y = float(self.red_y.text())
        except ValueError:
            red_y = 0
        try:
            blue_x = float(self.blue_x.text())
        except ValueError:
            blue_x = 0
        try:
            blue_y = float(self.blue_y.text())
        except ValueError:
            blue_y = 0

        distance = np.sqrt((red_x-blue_x)**2 + (red_y-blue_y)**2)*1e-3

        #self.distanceLineEdit.setText('%.2f' % distance)
        self.distanceLabel.setText('Distance between crosshairs: %.2f mm' % distance)
        #pass

    def update_red_crosshair(self):
        self.calculate_distance()
        self.red_crosshair.update_position()

    def update_blue_crosshair(self):
        self.calculate_distance()
        self.blue_crosshair.update_position()

    def red_crosshair_toggled(self, evt):
        """
        Method that is called when redButton is pressed
        :param evt: bool from signal
            True if redButton is checked, False if not
        :return:
        """

        if evt:
            # if red button is now checked
            # check if blue button is checked
            if self.blueButton.isChecked():
                # if so, uncheck it
                self.blueButton.toggle()
            # update current crosshair
            self.current_crosshair = self.red_crosshair
        else:
            # if red button is now unchecked, set current crosshair to None
            self.current_crosshair = None

    def blue_crosshair_toggled(self, evt):
        """
        Method that is called when blueButton is pressed
        :param evt: bool from signal
            True if blueButton is checked, False if not
        :return:
        """
        if evt:
            # if blue button is now checked
            # check if red button is checked
            if self.redButton.isChecked():
                # if so, uncheck it
                self.redButton.toggle()
            # update current crosshair
            self.current_crosshair = self.blue_crosshair
        else:
            # if red button is now unchecked, set current crosshair to None
            self.current_crosshair = None

    def update_crosshair_coords(self, coords):
        """
        Method to define new crosshair location based on mouseclick.
        :param evt: mouse click event
            Contains scene position
        :return:
        """
        # update current crosshair coordinates based on mouse click location
        if self.current_crosshair is not None:
            # update text to display crosshair location (in whatever units the viewbox coordinates are in)
            self.current_crosshair.xLineEdit.setText('%.1f' % coords.x())
            self.current_crosshair.yLineEdit.setText('%.1f' % coords.y())
            # draw crosshair at new location
            self.current_crosshair.update_position()

            # re-calculate crosshair distance
            self.calculate_distance()

    def update_crosshair_width(self):
        """
        Method to update the width of both crosshairs. Called when image changes shape
        :return:
        """
        # call update width for both crosshairs
        self.red_crosshair.update_width()
        self.blue_crosshair.update_width()


class LevelsWidget(QLevelsWidget, Ui_LevelsWidget):
    """
    Class to define a widget for adjusting image levels
    """

    def __init__(self, parent=None):
        """
        Initialize the widget.
        :param parent:
        """
        super(LevelsWidget, self).__init__()
        self.setupUi(self)

    def setText(self, minimum, maximum):
        """
        Method to update the text when autoscaling.
        :param minimum: int
            value to set in minLineEdit
        :param maximum: int
            value to set in maxLineEdit
        :return:
        """
        # set the text
        self.minLineEdit.setText('%d' % minimum)
        self.maxLineEdit.setText('%d' % maximum)


class Crosshair:
    """
    Class to represent a crosshair on an image. Draws crosshair on an image widget.
    """

    def __init__(self, color, xLineEdit, yLineEdit, lineout_image):
        """
        Method to create a crosshair
        :param color: str
            Must be a color defined in Qt package
        :param xLineEdit: QLineEdit
            LineEdit corresponding to horizontal position
        :param yLineEdit: QLineEdit
            LineEdit corresponding to vertical position
        :param rect: QGraphicsRectItem
            bounding rectangle of the viewbox
        :param lineout_image: LineoutImage
            widget where the crosshair goes
        """

        # try to set the color based on string input
        try:
            self.color = getattr(Qt, color)
        except AttributeError:
            # default to red
            self.color = Qt.red

        # set some attributes
        self.xLineEdit = xLineEdit
        self.yLineEdit = yLineEdit
        self.lineout_image = lineout_image

        # define lines that define the crosshair
        self.crossh = QtWidgets.QGraphicsLineItem(1024 - 25, 1024, 1024 + 25, 1024)
        self.crossv = QtWidgets.QGraphicsLineItem(1024, 1024 - 25, 1024, 1024 + 25)
        self.crossh.setPen(QtGui.QPen(self.color, 8, Qt.SolidLine))
        self.crossv.setPen(QtGui.QPen(self.color, 8, Qt.SolidLine))

        # put it in the viewbox
        self.lineout_image.view.addItem(self.crossh)
        self.lineout_image.view.addItem(self.crossv)

    def update_width(self):
        """
        Method to update crosshair size if viewbox changed
        :return:
        """
        # get width of the bounding rect
        rect_width = self.lineout_image.rect.boundingRect().width()
        # set line thickness to 1% of the viewbox width
        thickness = rect_width * .01
        # update lines
        self.crossh.setPen(QtGui.QPen(self.color, thickness, Qt.SolidLine))
        self.crossv.setPen(QtGui.QPen(self.color, thickness, Qt.SolidLine))

        try:
            # try to get the position of the crosshair
            xPos = float(self.xLineEdit.text())
            yPos = float(self.yLineEdit.text())
        except ValueError:
            # if it didn't work put it in the corner
            xPos = -rect_width/2
            yPos = -rect_width/2
        # set width of crosshair to 4% of the viewbox width
        self.crossh.setLine(xPos - rect_width * .02, yPos,
                             xPos + rect_width * .02, yPos)
        self.crossv.setLine(xPos, yPos - rect_width * .02,
                             xPos, yPos + rect_width * .02)

    def update_position(self):
        """
        Method to move the crosshair
        :return:
        """
        rect_width = self.lineout_image.rect.boundingRect().width()
        try:
            xPos = float(self.xLineEdit.text())
            yPos = float(self.yLineEdit.text())
        except ValueError:
            # if it didn't work put it in the corner
            xPos = -rect_width / 2
            yPos = -rect_width / 2
        # move crosshair
        self.crossh.setLine(xPos - rect_width*.02, yPos, xPos + rect_width*.02, yPos)
        self.crossv.setLine(xPos, yPos - rect_width*.02, xPos, yPos + rect_width*.02)
