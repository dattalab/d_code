#!/usr/bin/env python
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('Qt4Agg')
mpl.rcParams['backend.qt4']='PySide'

import numpy as np
import pprint

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from PySide import QtCore, QtGui


class CellPicker(QtGui.QWidget):
    def __init__(self):
        super(CellPicker, self).__init__()
        self.initMPL()
        self.shiftDown = False

    def initUI(self):
        cb = QtGui.QCheckBox('Show title', self)
        cb.move(20, 20)
        cb.toggle()
        cb.stateChanged.connect(self.changeTitle)
        
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('QtGui.QCheckBox')
        self.show()

    def changeTitle(self, state):
        if state == QtCore.Qt.Checked:
            self.setWindowTitle('Checkbox')
        else:
            self.setWindowTitle('')

    def initMPL(self):
        # generate the plot
        self.fig = plt.Figure(figsize=(200,200), dpi=72, facecolor=(1,1,1), edgecolor=(0,0,0))
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(np.random.random((100,100)), picker=True)
        # generate the canvas to display the plot
        self.canvas = FigureCanvas(self.fig)

        self.canvas.resize(200,200)

        self.win = QtGui.QMainWindow()
        self.win.setCentralWidget(self.canvas)

        self.canvas.mpl_connect('pick_event', self.onpick)
        self.win.keyPressEvent = self.keyPressEvent
        self.win.keyReleaseEvent = self.keyReleaseEvent

        self.win.setWindowTitle('fuck')
        self.win.show()

    def keyPressEvent(self, e):
        if e.key() == 16777248:
            self.shiftDown = True
        else:
            print e.key()

    def keyReleaseEvent(self, e):
        if e.key() == 16777248:
            self.shiftDown = False
        else:
            print e.key()
        
    def onpick(self,event):
        if isinstance(event.artist, mpl.image.AxesImage):
            # print 'this is an image'
            self.im = event.artist

            print 'dir(event): %s' % dir(event)
            try:
                pprint.pprint (vars(event.mouseevent))
            except AttributeError:
                pass

            # we can use event.mouseevent.key == 'shift' or 'control' to alter behavior
            if self.shiftDown:
                print 'modified by shift'

            # X AND Y ARE NOT ON THE SAME DIM AS DATA!!!!
            self.y,self.x = int(np.floor(event.mouseevent.xdata)),int(np.floor(event.mouseevent.ydata))
            x=self.x
            y=self.y
        
            print 'x: %s, y: %s' % (x,y)

def runPyside():
    try:
        app = QtGui.QApplication(sys.argv)
        print 'launching new app...'
    except:
        app = QtGui.QApplication.instance()
    ex = CellPicker()

    app.connect(app, QtCore.SIGNAL("lastWindowClosed()"), app, QtCore.SLOT("quit()"))

    app.exec_()
    return ex.win.windowTitle()

if __name__ == '__main__':
    runPyside()
