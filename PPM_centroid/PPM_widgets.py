from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


class Crosshair:

    def __init__(self, color, button, xLineEdit, yLineEdit, rect):

        try:
            self.color = getattr(Qt, color)
        except AttributeError:
            # default to red
            self.color = Qt.red
        self.button = button
        self.xLineEdit = xLineEdit
        self.yLineEdit = yLineEdit
        self.rect = rect

        self.crossh = QtWidgets.QGraphicsLineItem(1024 - 25, 1024, 1024 + 25, 1024)
        self.crossv = QtWidgets.QGraphicsLineItem(1024, 1024 - 25, 1024, 1024 + 25)
        self.crossh.setPen(QtGui.QPen(self.color, 8, Qt.SolidLine))
        self.crossv.setPen(QtGui.QPen(self.color, 8, Qt.SolidLine))

    def embed(self, viewbox):
        viewbox.addItem(self.crossh)
        viewbox.addItem(self.crossv)

    def update_width(self):
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
        rect_width = self.rect.boundingRect().width()
        xPos = float(self.xLineEdit.text())
        yPos = float(self.yLineEdit.text())
        self.crossh.setLine(xPos - rect_width*.02, yPos, xPos + rect_width*.02, yPos)
        self.crossv.setLine(xPos, yPos - rect_width*.02, xPos, yPos + rect_width*.02)

