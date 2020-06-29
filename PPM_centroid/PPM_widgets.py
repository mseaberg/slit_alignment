from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from PyQt5.uic import loadUiType
import numpy as np
from datetime import datetime


Ui_LineoutImage, QLineoutImage = loadUiType('LineoutImage.ui')
Ui_Crosshair, QCrosshair = loadUiType('Crosshair.ui')
Ui_LevelsWidget, QLevelsWidget = loadUiType('LevelsWidget.ui')


class LineoutImage(QLineoutImage, Ui_LineoutImage):

    def __init__(self, parent=None):
      
        super(LineoutImage, self).__init__()
        self.setupUi(self)

        # define layout
        #layout = QtWidgets.QGridLayout()
        #layout.addWidget(self)
        #layout.addWidget(self.image_canvas,0,0,4,4)
        #layout.addWidget(self.xlineout_canvas,4,0,2,4)
        #layout.addWidget(self.ylineout_canvas,0,4,4,2)
        #groupbox.setLayout(layout)

        # connect levels
        self.levels = None

        self.minimum = 0
        self.maximum = 4096

        # add viewbox for image
        self.view = self.image_canvas.addViewBox()
        # setup viewbox and get corresponding QRect
        self.rect = self.setup_viewbox(1024)
        
        # lock aspect ratio
        self.view.setAspectLocked(True)
        # add an image
        self.img = pg.ImageItem(border='w')
        self.view.addItem(self.img)
        
        # font styles
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.font = QtGui.QFont()
        self.font.setPointSize(10)
        self.font.setFamily('Arial')


        # initialize lineouts
        self.horizontalPlot, self.horizontalLineout, self.horizontalFit = self.initialize_lineout(self.xlineout_canvas, 'horizontal')
        self.verticalPlot, self.verticalLineout, self.verticalFit = self.initialize_lineout(self.ylineout_canvas, 'vertical')

    def connect_levels(self, levels):
        # connect levels
        self.levels = levels

        
        self.set_min()
        self.set_max()
        self.levels.minLineEdit.returnPressed.connect(self.set_min)
        self.levels.maxLineEdit.returnPressed.connect(self.set_max)



    def get_canvases(self):
        return self.image_canvas, self.xlineout_canvas, self.ylineout_canvas

    def update_plots(self, image_data, x, y, xlineout_data, ylineout_data, fit_x, fit_y):

        if self.levels is not None:
            if self.levels.checkBox.isChecked():
                self.minimum = np.min(image_data)
                self.maximum = np.max(image_data)
                self.levels.setText(self.minimum, self.maximum)
        else:
            self.minimum = np.min(image_data)
            self.maximum = np.max(image_data)
        
        x_width = np.max(x) - np.min(x)
        y_width = np.max(y) - np.min(y)

        self.img.setImage(np.flipud(image_data).T,
                levels=(self.minimum, self.maximum))

        self.img.setRect(QtCore.QRectF(np.min(x),np.min(y),x_width, y_width))

        self.horizontalLineout.setData(x, xlineout_data)
        self.horizontalFit.setData(x, fit_x)
        self.verticalLineout.setData(ylineout_data, y)
        self.verticalFit.setData(fit_y, y)



    def set_min(self):
        self.minimum = float(self.levels.minLineEdit.text())

    def set_max(self):
        self.maximum = float(self.levels.maxLineEdit.text())

    def setup_viewbox(self, width):
        """
        Helper function to set up viewbox with title
        :param viewbox: pyqtgraph viewbox
        :param width: image width in pixels (int)
        """
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(-width/2., -width/2., width, width))
        rect1 = QtGui.QGraphicsRectItem(-width/2., -width/2., width, width)
        rect1.setPen(QtGui.QPen(QtCore.Qt.white, width/50., QtCore.Qt.SolidLine))
        self.view.addItem(rect1)
        return rect1
        
    def update_viewbox(self, width, height):
        """
        Helper function to adjust viewbox settings
        :param width: new width in pixels (int)
        :param height: new height in pixels (int)
        :return:
        """
        self.view.setRange(QtCore.QRectF(-width/2, -height/2, width, height))
        self.rect.setPen(QtGui.QPen(QtCore.Qt.white, width/50., QtCore.Qt.SolidLine))
        self.rect.setRect(-width/2, -height/2, width, height)

    def initialize_lineout(self, canvas, direction):
        """
        Method to set up lineout plots.
        """
        names = ['Lineout', 'Fit']
        colors = ['r', 'c']

        if direction == 'horizontal':
            lineoutPlot = canvas.addPlot()
            legend = lineoutPlot.addLegend(offset=(10,0))
            lineoutData = lineoutPlot.plot(np.linspace(-1024, 1023, 100), np.zeros(100),
                                           pen=pg.mkPen(colors[0], width=2),name=names[0])
            lineoutFit = lineoutPlot.plot(np.linspace(-1024, 1023, 100), np.zeros(100),
                                           pen=pg.mkPen(colors[1], width=2),name=names[1])
            lineoutPlot.setYRange(0, 1)
            PlotUtil.setup_legend(legend)
            self.label_plot(lineoutPlot, u'x (\u03BCm)', 'Intensity')
            lineoutPlot.setXLink(self.view)
        elif direction == 'vertical':
            lineoutPlot = canvas.addPlot()
            lineoutData = lineoutPlot.plot(np.zeros(100), np.linspace(-1024, 1023, 100),
                                           pen=pg.mkPen(colors[0], width=2),name=names[0])
            lineoutFit = lineoutPlot.plot(np.zeros(100), np.linspace(-1024, 1023, 100),
                                           pen=pg.mkPen(colors[1], width=2),name=names[1])
            lineoutPlot.setXRange(0, 1)
            self.label_plot(lineoutPlot, 'Intensity', u'y (\u03BCm)')
            lineoutPlot.setYLink(self.view)
        else:
            lineoutPlot = None
            lineoutData = None
            lineoutFit = None
            pass
        return lineoutPlot, lineoutData, lineoutFit

    def label_plot(self, plot, xlabel, ylabel):
        """
        Helper function to set plot labels
        :param plot: pyqtgraph plot item
        :param xlabel: x-axis label (str)
        :param ylabel: y-axis label (str)
        """
        xaxis = plot.getAxis('bottom')
        xaxis.setLabel(text=xlabel, **self.labelStyle)
        xaxis.tickFont = self.font
        xaxis.setPen(pg.mkPen('w', width=1))
        yaxis = plot.getAxis('left')
        yaxis.setLabel(text=ylabel, **self.labelStyle)
        yaxis.tickFont = self.font
        yaxis.setPen(pg.mkPen('w', width=1))

    def add_crosshair(self, color, xLineEdit, yLineEdit):
        
        crosshairObject = Crosshair(color, xLineEdit, yLineEdit, self)

        return crosshairObject


class PlotUtil:

    @staticmethod
    def setup_legend(legend):

        legendLabelStyle = {'color': '#FFF', 'size': '10pt'}
        for item in legend.items:
           for single_item in item:
               if isinstance(single_item, pg.graphicsItems.LabelItem.LabelItem):
                   single_item.setText(single_item.text, **legendLabelStyle)




class StripChart:

    def __init__(self, canvas, ylabel):
       
        # font styles
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.font = QtGui.QFont()
        self.font.setPointSize(10)
        self.font.setFamily('Arial')
        
        # generate plot
        self.canvas = canvas
        self.plotWidget = self.canvas.addPlot()

        # set x axis label (this class is specifically for time series)
        xaxis = self.plotWidget.getAxis('bottom')
        xaxis.setLabel(text='Time (s)',**self.labelStyle)
        xaxis.tickFont = self.font
        xaxis.setPen(pg.mkPen('w',width=1))
        yaxis = self.plotWidget.getAxis('left')
        yaxis.setLabel(text=ylabel,**self.labelStyle)
        yaxis.tickFont = self.font
        yaxis.setPen(pg.mkPen('w',width=1))

        self.plotWidget.showGrid(x=True,y=True,alpha=.8)
        # initialize dictionary of lines to plot
        self.lines = {}

        self.color_order = ['r', 'c', 'm', 'g']

        # default time range (seconds)
        self.time_range = 10

    def addSeries(self, series_keys, series_labels):
        
        legend = self.plotWidget.addLegend()

        for num, key in enumerate(series_keys):

            self.lines[key] = self.plotWidget.plot(np.linspace(-99,0,100), np.zeros(100),
                    pen=pg.mkPen(self.color_order[num], width=5),name=series_labels[num])

        PlotUtil.setup_legend(legend)

    def set_time_range(self, time_range):
        self.time_range = time_range

    def update_plots(self, time_stamps, **data):

        # filter out any data that doesn't exist yet
        mask = time_stamps > 0

        # process time stamps
        now = datetime.now()
        now_stamp = datetime.timestamp(now)

        time_stamps = time_stamps - now_stamp
        time_stamps = time_stamps[mask]

        for key, value in data.items():
            # filter data with mask
            filtered_value = value[mask]
            try:
                self.lines[key].setData(time_stamps, filtered_value)
            except KeyError:
                print('Data had the wrong name')

        # reset plot range
        self.plotWidget.setXRange(-self.time_range, 0)




class CrosshairWidget(QCrosshair, Ui_Crosshair):

    def __init__(self, parent=None):
        super(CrosshairWidget, self).__init__()
        self.setupUi(self)

        self.lineout_image = None

        self.red_crosshair = None
        self.blue_crosshair = None

        self.current_crosshair = None

    def connect_image(self, image_widget):
        self.lineout_image = image_widget
        # connect to mouse click on image
        self.lineout_image.rect.scene().sigMouseClicked.connect(self.mouseClicked)
        self.red_crosshair = Crosshair('red', self.red_x, self.red_y, self.lineout_image)
        self.blue_crosshair = Crosshair('blue', self.blue_x, self.blue_y, self.lineout_image)

        self.redButton.toggled.connect(self.red_crosshair_toggled)
        self.blueButton.toggled.connect(self.blue_crosshair_toggled)
        self.red_x.returnPressed.connect(self.red_crosshair.update_position)
        self.red_y.returnPressed.connect(self.red_crosshair.update_position)
        self.blue_x.returnPressed.connect(self.blue_crosshair.update_position)
        self.blue_y.returnPressed.connect(self.blue_crosshair.update_position)

    def red_crosshair_toggled(self, evt):
        if evt:
            if self.blueButton.isChecked():
                self.blueButton.toggle()
            self.current_crosshair = self.red_crosshair
        else:
            self.current_crosshair = None

    def blue_crosshair_toggled(self, evt):
        if evt:
            if self.redButton.isChecked():
                self.redButton.toggle()
            self.current_crosshair = self.blue_crosshair
        else:
            self.current_crosshair = None

    def mouseClicked(self, evt):
        # translate scene coordinates to viewbox coordinates
        coords = self.lineout_image.view.mapSceneToView(evt.scenePos())

        if self.current_crosshair is not None:
            self.current_crosshair.xLineEdit.setText('%.1f' % coords.x())
            self.current_crosshair.yLineEdit.setText('%.1f' % coords.y())
            self.current_crosshair.update_position()

    def update_crosshair_width(self):
        self.red_crosshair.update_width()
        self.blue_crosshair.update_width()

    def get_crosshairs(self):
        return self.red_crosshair, self.blue_crosshair


class LevelsWidget(QLevelsWidget, Ui_LevelsWidget):

    def __init__(self, parent=None):
        super(LevelsWidget, self).__init__()
        self.setupUi(self)

    def setText(self, minimum, maximum):
        self.minLineEdit.setText('%d' % minimum)
        self.maxLineEdit.setText('%d' % maximum)


class Crosshair:

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
        rect_width = self.lineout_image.rect.boundingRect().width()
        thickness = rect_width * .01
        self.crossh.setPen(QtGui.QPen(self.color, thickness, Qt.SolidLine))
        self.crossv.setPen(QtGui.QPen(self.color, thickness, Qt.SolidLine))

        try:
            xPos = float(self.xLineEdit.text())
            yPos = float(self.yLineEdit.text())
        except ValueError:
            xPos = -rect_width/2
            yPos = -rect_width/2
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
        xPos = float(self.xLineEdit.text())
        yPos = float(self.yLineEdit.text())
        self.crossh.setLine(xPos - rect_width*.02, yPos, xPos + rect_width*.02, yPos)
        self.crossv.setLine(xPos, yPos - rect_width*.02, xPos, yPos + rect_width*.02)

