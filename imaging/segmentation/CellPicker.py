import matplotlib as mpl

from PySide import QtCore, QtGui
import sys
mpl.rcParams['backend.qt4']='PySide'
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np

import pymorph
import mahotas

import matplotlib.nxutils as nx

from traces import smooth

from sklearn import decomposition

from scipy.stats import skew

class Communicate(QtCore.QObject):
    keyPressed = QtCore.Signal(tuple)
    mouseSingleClicked = QtCore.Signal(tuple)
    mouseSingleShiftClicked = QtCore.Signal(tuple)
    mouseDoubleClicked = QtCore.Signal(tuple)

class MatplotlibWidget(FigureCanvas):
    def __init__(self, cellImage, parent=None):
        super(MatplotlibWidget, self).__init__(Figure())

        self.c = Communicate()
        self.image = cellImage
        
        self.setParent(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.axes.imshow(self.image, cmap=mpl.cm.gray)
        
        self.setGeometry(QtCore.QRect(150, 10, 768, 768))
        self.width = self.geometry().width()
        self.height = self.geometry().height()

        self.setFocus()

    @QtCore.Slot()
    def updateImage(self, image=None):
        self.mask = image
        self.axes.clear()
        self.axes.imshow(self.image, cmap=mpl.cm.gray)
        self.axes.imshow(self.mask, cmap=mpl.cm.jet)
        self.draw()    
    
    #signal
    def keyPressEvent(self, event):
        # self.c.keyPressed.emit(event.text())
        self.c.keyPressed.emit(event.key())
        event.accept()
        
    #signal
    def mousePressEvent(self, event):
        self.setFocus()
        if self.height == self.width:
            x = int(float(event.pos().y()) / self.height * self.image.shape[0])
            y = int(float(event.pos().x()) / self.width * self.image.shape[1])
        elif self.height < self.width:
            y = int(float(event.pos().x()) / self.width * self.image.shape[1])
            
            x = 0
        elif self.height > self.width:
            x = int(float(event.pos().y()) / self.height * self.image.shape[0])

            y = 0
        
        #switch here for shift-click (emit different signal)
        if QtCore.Qt.Modifier.SHIFT and event.modifiers():
            self.c.mouseSingleShiftClicked.emit((x, y))
        else: 
            self.c.mouseSingleClicked.emit((x, y))

        event.accept()
      
    def mouseDoubleClickEvent(self, event):
        self.setFocus()
        event.accept()
    
class CellPickerGUI(object):
    def setupUi(self, MainWindow, cellImage, mask, series):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(921, 802)
        self.MainWindow = MainWindow
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.image_widget = MatplotlibWidget(cellImage, parent=self.centralwidget)
        # note that the widget size is hardcoded in the class (not the best, but at least it's all
        # in the constructor
        
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.image_widget.sizePolicy().hasHeightForWidth())
        self.image_widget.setSizePolicy(sizePolicy)
        self.image_widget.setMinimumSize(QtCore.QSize(512, 512))
        self.image_widget.setObjectName("image_widget")
        self.image_widget.setFocus()

        self.splitter = QtGui.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(10, 480, 111, 41))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.label_2 = QtGui.QLabel(self.splitter)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.dilation_disk = QtGui.QSpinBox(self.splitter)
        self.dilation_disk.setProperty("value", 1)
        self.dilation_disk.setObjectName("dilation_disk")
        self.splitter_2 = QtGui.QSplitter(self.centralwidget)
        self.splitter_2.setGeometry(QtCore.QRect(10, 420, 122, 41))
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setObjectName("splitter_2")
        self.label = QtGui.QLabel(self.splitter_2)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.contrast_threshold = QtGui.QDoubleSpinBox(self.splitter_2)
        self.contrast_threshold.setSingleStep(0.01)
        self.contrast_threshold.setProperty("value", 0.95)
        self.contrast_threshold.setObjectName("contrast_threshold")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.contrast_threshold.valueChanged.connect(self.changeContrastThreshold)
        self.dilation_disk.valueChanged.connect(self.changeDiskSize)
        
        self.image_widget.c.keyPressed.connect(self.keyPress)
        self.image_widget.c.mouseSingleClicked.connect(self.addCell)
        self.image_widget.c.mouseSingleShiftClicked.connect(self.deleteCell)
        
        self.data = cellImage
        if mask is None:
            self.currentMask = np.zeros_like(cellImage, dtype='uint16')
        else:
            self.currentMask = mask.astype('uint16')
        
        if series is None:
            self.series = None
        else:
            self.series = series

        self.listOfMasks = []
        self.listOfMasks.append(self.currentMask)
        self.diskSize = 1
        self.contrastThreshold = 0.95
        self.cellRadius = 3
        self.currentMaskNumber = 1
        
        self.mode = None # can be standard (None), 'poly', or 'square'
        self.modeData = None # or a list of point tuples

        self.makeNewMaskAndBackgroundImage()
        
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("MainWindow", "<html><head/><body><p>Disk Dilation Size</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("MainWindow", "<html><head/><body><p>Contrast Threshold</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))

    # dispatcher method to handle all key presses
    @QtCore.Slot()
    def keyPress(self, keyPressed):
        
        if keyPressed == QtCore.Qt.Key_Return or keyPressed == QtCore.Qt.Key_Enter:
            self.MainWindow.close()
            
        elif keyPressed >= QtCore.Qt.Key_1 and keyPressed <= QtCore.Qt.Key_8:
            # 1 = 49, 8 = 56
            self.currentMaskNumber = keyPressed - 48
            print self.currentMaskNumber

        elif keyPressed == QtCore.Qt.Key_Z:
            self.revert()

        elif keyPressed == QtCore.Qt.Key_Minus:
            self.decrementContrastThreshold()
        elif keyPressed == QtCore.Qt.Key_Equal or keyPressed == QtCore.Qt.Key_Plus:
            self.incrementContrastThreshold()


        elif keyPressed == QtCore.Qt.Key_BracketLeft:
            self.decrementDiskSize()
        elif keyPressed == QtCore.Qt.Key_BracketRight:
            self.incrementDiskSize()

        elif keyPressed == QtCore.Qt.Key_P:
            if self.mode is 'poly':
                self.mode = None
            else:
                self.clearModeData()
                self.mode = 'poly'

        elif keyPressed == QtCore.Qt.Key_T:
            if self.mode is 'poly':
                self.addPolyCell()
                self.clearModeData()

        elif keyPressed == QtCore.Qt.Key_S:
            if self.mode is 'square':
                self.mode = None
            else:
                self.clearModeData()
                self.mode = 'square'

        elif keyPressed == QtCore.Qt.Key_C:
            if self.mode is 'circle':
                self.mode = None
            else:
                self.clearModeData()
                self.mode = 'circle'
                
        elif keyPressed == QtCore.Qt.Key_K:
            self.correlateLastROI()

        elif keyPressed == QtCore.Qt.Key_X:
            self.clearModeData()
            self.mode = None
        else:
            pass

    def mask_from_points(self, vertex_list, size_x, size_y):
        #poly_verts = [(20,0), (50,50), (0,75)]

        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(size_x), np.arange(size_y))
        x, y = x.flatten(), y.flatten()
        point_space = np.vstack((x,y)).T

        poly_mask = nx.points_inside_poly(point_space, vertex_list)
        poly_mask = poly_mask.reshape(size_x, size_y)

        return poly_mask.T


    def addPolyCell(self):
        # build poly_mask
        poly_mask = self.mask_from_points(self.modeData, self.currentMask.shape[0], self.currentMask.shape[1])
        # check if poly_mask interfers with current mask, if so, abort
        if np.any(np.logical_and(poly_mask, self.currentMask)):
            return None

        # add poly_mask to mask
        newMask = (poly_mask * self.currentMaskNumber) + self.currentMask

        self.listOfMasks.append(newMask)
        self.currentMask = self.listOfMasks[-1]

        sys.stdout.flush()
        self.makeNewMaskAndBackgroundImage()

    def clearModeData(self):
        self.modeData = []

    def correlateLastROI(self):
        lastROI = np.logical_xor(listOfMasks[-1], listOfMasks[-2]) 

        # build correlation matrix
        

        # set corr matrix to zero where other ROIs exist

        # multiply with gaussian falloff

        # use threshold t inc. pixels

        # open and close to remove holes

        # update mask and redisplay

        pass


    @QtCore.Slot(tuple)
    def addCell(self, eventTuple):
        x, y = eventTuple
        localValue = self.currentMask[x,y]
        print 'x: ' + str(x) + ', y: ' + str(y) + ', mask val: ' + str(localValue) 
        sys.stdout.flush()


        ########## NORMAL MODE 
        if self.mode is None:
            if localValue > 0 and localValue != self.currentMaskNumber:
                print 'we are altering mask at at %d, %d' % (x, y)
            
                # copy the old mask
                newMask = self.currentMask.copy()

                # make a labeled image of the current mask
                labeledCurrentMask = mahotas.label(newMask)[0]
                roiNumber = labeledCurrentMask[x, y]
            
                # set that ROI to zero
                newMask[labeledCurrentMask == roiNumber] = self.currentMaskNumber
            
                self.listOfMasks.append(newMask)
                self.currentMask = self.listOfMasks[-1]
            elif localValue == 0:
                # build structuring elements
                se = pymorph.sebox()
                se2 = pymorph.sedisk(self.cellRadius, metric='city-block')


                seJunk = pymorph.sedisk(max(np.floor(self.cellRadius/4.0), 1), metric='city-block')
                seExpand = pymorph.sedisk(self.diskSize, metric='city-block')

                # add a disk around selected point, non-overlapping with adjacent cells
                dilatedOrignal = mahotas.dilate(self.currentMask.copy().astype(np.uint16), Bc=se)
                safeUnselected = np.logical_not(dilatedOrignal)
            
                # tempMask is 
                tempMask = np.zeros_like(self.currentMask.copy()).astype(np.uint16)
                tempMask[x, y]= 1
                tempMask = mahotas.dilate(tempMask, Bc=se2)
                tempMask = np.logical_and(tempMask, safeUnselected)
            
                # calculate the area we should add to this disk based on % of a threshold
                cellMean = self.data[tempMask == 1].mean()
                allMeanBw = self.data >= (cellMean * float(self.contrastThreshold))
     
                tempLabel = mahotas.label(np.logical_and(allMeanBw, safeUnselected).astype(np.uint16))[0]
                connMeanBw = tempLabel == tempLabel[x, y]
            
                connMeanBw = np.logical_and(np.logical_or(connMeanBw, tempMask), safeUnselected).astype(np.bool)

                # erode and then dilate to remove sharp bits and edges
            
                erodedMean = mahotas.erode(connMeanBw, Bc=seJunk)
                dilateMean = mahotas.dilate(erodedMean, Bc=seJunk)
                dilateMean = mahotas.dilate(dilateMean, Bc=seExpand)
            
                newCell = np.logical_and(dilateMean, safeUnselected)
                newMask = (newCell * self.currentMaskNumber) + self.currentMask

                self.listOfMasks.append(newMask.copy())
                self.currentMask = newMask.copy()

        elif self.mode is 'square':
            self.modeData.append((x, y))
            if len(self.modeData) == 2:
                square_mask = np.zeros_like(self.currentMask)
                xstart = self.modeData[0][0]
                ystart = self.modeData[0][1]

                xend = self.modeData[1][0]
                yend = self.modeData[1][1]

                square_mask[xstart:xend, ystart:yend] = 1

                # check if square_mask interfers with current mask, if so, abort
                if np.any(np.logical_and(square_mask, self.currentMask)):
                    return None

                # add square_mask to mask
                newMask = (square_mask * self.currentMaskNumber) + self.currentMask

                self.listOfMasks.append(newMask)
                self.currentMask = self.listOfMasks[-1]

                # clear current mode data
                self.clearModeData()

        elif self.mode is 'circle':
            # make a strel and move it in place to make circle_mask
            if self.diskSize < 1:
                return None

            if self.diskSize is 1:
                se = np.ones((1,1))
            elif self.diskSize is 2:
                se = pymorph.secross(r=1)
            else:
                se = pymorph.sedisk(r=(self.diskSize-1))

            se_extent = int(se.shape[0]/2)
            circle_mask = np.zeros_like(self.currentMask)
            circle_mask[x-se_extent:x+se_extent+1, y-se_extent:y+se_extent+1] = se

            # check if circle_mask interfers with current mask, if so, abort
            if np.any(np.logical_and(circle_mask, self.currentMask)):
                return None

            # add circle_mask to mask
            newMask = (circle_mask * self.currentMaskNumber) + self.currentMask

            self.listOfMasks.append(newMask)
            self.currentMask = self.listOfMasks[-1]

        elif self.mode is 'poly':
            self.modeData.append((x, y))

        sys.stdout.flush()
        self.makeNewMaskAndBackgroundImage()
    
    @QtCore.Slot(tuple)
    def deleteCell(self, eventTuple):
        x, y = eventTuple
        localValue = self.currentMask[x,y].copy()
        print 'mask value at %d, %d is %f' % (x, y, localValue)
        sys.stdout.flush()
        
        if self.currentMask[x,y] > 0:
            #print 'we are deleting at %d, %d' % (x, y)
            
            # copy the old mask
            newMask = self.currentMask.copy()

            #make a labeled image of the current mask
            labeledCurrentMask = mahotas.label(newMask)[0]
            roiNumber = labeledCurrentMask[x, y]
            
            # set that ROI to zero
            newMask[labeledCurrentMask == roiNumber] = 0
            
            self.listOfMasks.append(newMask)
            self.currentMask = self.listOfMasks[-1]

        sys.stdout.flush()
        self.makeNewMaskAndBackgroundImage()
    
    # go back to the previous mask
    def revert(self):
        if len(self.listOfMasks) > 1:
            self.listOfMasks.pop()
            self.currentMask = self.listOfMasks[-1]
            self.makeNewMaskAndBackgroundImage()
        else:
            print 'No current mask!'

    # refresh image widget
    def makeNewMaskAndBackgroundImage(self):
        norm = mpl.colors.NoNorm()
        cmap = mpl.cm.jet
        converter = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        if self.currentMask.any():
            color_mask = converter.to_rgba(self.currentMask.astype(float)/self.currentMask.max())
        else:
            color_mask = converter.to_rgba(self.currentMask)
        color_mask[:,:,3] = 0
        color_mask[:,:,3] = (color_mask[:,:,0:2].sum(axis=2) > 0).astype(float) * 0.5

        self.image_widget.updateImage(color_mask)
        
    # update model
    @QtCore.Slot()
    def changeDiskSize(self):#,event):
        self.diskSize = self.dilation_disk.value()

    def decrementDiskSize(self):
        self.dilation_disk.setValue(self.dilation_disk.value() - 1)
        self.diskSize = self.dilation_disk.value()

    def incrementDiskSize(self):
        self.dilation_disk.setValue(self.dilation_disk.value() + 1)
        self.diskSize = self.dilation_disk.value()
        
    def decrementContrastThreshold(self):
        self.contrast_threshold.setValue(self.contrast_threshold.value() - 0.01)
        self.contrastThreshold = self.contrast_threshold.value()

    def incrementContrastThreshold(self):
        self.contrast_threshold.setValue(self.contrast_threshold.value() + 0.01)
        self.contrastThreshold = self.contrast_threshold.value()
    
    # update model
    @QtCore.Slot()
    def changeContrastThreshold(self, event):
        self.contrastThreshold = self.contrast_threshold.value()
        
def pickCells(backgroundImage, mask=None, series=None):
    """This routine is for interactive picking of cells and editing
    a mask.  It takes two arguments- a numpy array for a background image.
    If backgroundImage is 2d, then that is the image used.  If it is
    3D then we assume it is an image series and average over the 3rd
    dimension.

    Returns a mask array.  The mask can have values ranging from 1-8,
    each indicating a different feature.  Use mahotas.label(mask==#) to 
    make a labeled mask for each sub-mask.

    """

    # need to be robust to passing in a stack or a single image
    if backgroundImage.ndim == 3:
        backgroundImage = backgroundImage.mean(axis=2)

    global app
    try:
        app = QtGui.QApplication(sys.argv)
    except RuntimeError:
        pass

    MainWindow = QtGui.QMainWindow()
    gui = CellPickerGUI()
    gui.setupUi(MainWindow, backgroundImage, mask, series)
    MainWindow.show()
    MainWindow.raise_()
    
    app.exec_()
    
    try:
        return gui.currentMask
    except IndexError:
        print 'No mask to return!'
