from PPM_interface import PPM_Interface
import warnings
from pyqtgraph.Qt import QtGui
import sys

if __name__ == '__main__':

    warnings.filterwarnings('ignore', '.*output shape of zoom.*')
    app = QtGui.QApplication(sys.argv)
    thisapp = PPM_Interface()
    thisapp.show()
    sys.exit(app.exec_())
