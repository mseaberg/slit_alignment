from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from PyQt5.uic import loadUiType


Ui_LineoutImage, QLineoutImage = loadUiType('LineoutImage.ui')


class LineoutImage(QLineoutImage, Ui_LineoutImage):

    def __init__(self, groupbox):
      
        super(LineoutImage, self).__init__()
        self.setupUi(self)

        # make canvases
        #self.image_canvas = pg.GraphicsLayoutWidget()
        #sp = self.image_canvas.sizePolicy()
        #sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        #sp.setHorizontalStretch(0)
        #self.image_canvas.setSizePolicy(sp)
        #self.xlineout_canvas = pg.GraphicsLayoutWidget()
        #self.ylineout_canvas = pg.GraphicsLayoutWidget()

        #self.image_canvas.setMinimumSize(QtCore.QSize(200,200))
        #self.image_canvas.setMaximumSize(QtCore.QSize(1024,1024))
        #self.xlineout_canvas.setMaximumSize(QtCore.QSize(16777215, 120))
        #self.ylineout_canvas.setMaximumSize(QtCore.QSize(120, 16777215))

        # define layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self)
        #layout.addWidget(self.image_canvas,0,0,4,4)
        #layout.addWidget(self.xlineout_canvas,4,0,2,4)
        #layout.addWidget(self.ylineout_canvas,0,4,4,2)
        groupbox.setLayout(layout)

        #return image_canvas, xlineout_canvas, ylineout_canvas
    
    def get_canvases(self):
        return self.image_canvas, self.xlineout_canvas, self.ylineout_canvas


class Crosshair:

    def __init__(self, color, xLineEdit, yLineEdit, rect, viewbox):
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
        :param viewbox: pg.ViewBox
            Viewbox for displaying the crosshair
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
        self.rect = rect

        # define lines that define the crosshair
        self.crossh = QtWidgets.QGraphicsLineItem(1024 - 25, 1024, 1024 + 25, 1024)
        self.crossv = QtWidgets.QGraphicsLineItem(1024, 1024 - 25, 1024, 1024 + 25)
        self.crossh.setPen(QtGui.QPen(self.color, 8, Qt.SolidLine))
        self.crossv.setPen(QtGui.QPen(self.color, 8, Qt.SolidLine))

        # put it in the viewbox
        viewbox.addItem(self.crossh)
        viewbox.addItem(self.crossv)

    def update_width(self):
        """
        Method to update crosshair size if viewbox changed
        :return:
        """
        rect_width = self.rect.boundingRect().width()
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
        rect_width = self.rect.boundingRect().width()
        xPos = float(self.xLineEdit.text())
        yPos = float(self.yLineEdit.text())
        self.crossh.setLine(xPos - rect_width*.02, yPos, xPos + rect_width*.02, yPos)
        self.crossv.setLine(xPos, yPos - rect_width*.02, xPos, yPos + rect_width*.02)

