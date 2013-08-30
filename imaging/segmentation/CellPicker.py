import matplotlib as mpl

from PySide import QtCore, QtGui
import sys
mpl.rcParams['backend.qt4']='PySide'
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np

import scipy.ndimage as nd
import scipy

import pymorph
import mahotas

import matplotlib.nxutils as nx

from sklearn.decomposition import NMF

import pdb

class Communicate(QtCore.QObject):
    keyPressed = QtCore.Signal(tuple)
    mouseSingleClicked = QtCore.Signal(tuple)
    mouseSingleShiftClicked = QtCore.Signal(tuple)
    mouseDoubleClicked = QtCore.Signal(tuple)

class MatplotlibWidget(FigureCanvas):
    """Custom Matplotlib Widget"""
    def __init__(self, data, parent, cutoff):
        super(MatplotlibWidget, self).__init__(Figure())

        self.c = Communicate()
        if data.ndim == 2:
            self.image = data
        elif data.ndim == 3:
            self.image = data.mean(axis=2)
            
        self.setParent(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        self.maxAverageImageVal = self.image.max()

        self.image_ax = self.axes.imshow(self.image, cmap=mpl.cm.gray, vmax=self.maxAverageImageVal*cutoff)
        self.mask = np.zeros((self.image.shape[0], self.image.shape[1],4))
        self.mask_ax = self.axes.imshow(self.mask, cmap=mpl.cm.jet)
        
        self.setGeometry(QtCore.QRect(150, 10, 768, 768))
        self.width = self.geometry().width()
        self.height = self.geometry().height()

        self.setFocus()

    @QtCore.Slot()
    def updateImage(self, image, mask=None):
        """image is a 2d image, mask is a 2d RGBA image"""
        self.image = image
        self.mask = mask
        self.image_ax.set_data(image)
        self.mask_ax.set_data(mask)
        self.draw()    
    
    # signal
    def keyPressEvent(self, event):
        # self.c.keyPressed.emit(event.text())
        self.c.keyPressed.emit(event.key())
        event.accept()
        
    # signal
    def mousePressEvent(self, event):
        self.setFocus()
        
        # we need to map the clicked location into matrix coordinates
        # this is complicated by two factors:  
        # 1) x&y in event coords are switched relative to matrix coordinates
        # 2) we have to account for non-square matrices.  this is done with the x_ and y_offsets

        x_offset = y_offset = 0
        if self.image.shape[0] < self.image.shape[1]:
            margin = (1 - (float(self.image.shape[0]) / float(self.image.shape[1])) ) / 2.0
            x_offset = int(margin * 768)  # 0 if x = y
        if self.image.shape[0] > self.image.shape[1]:
            margin = (1 - (float(self.image.shape[1]) / float(self.image.shape[0])) ) / 2.0
            y_offset = int(margin * 768)  # 0 if x = y

        if self.height == self.width:
            x = int( (float(event.pos().y()) - x_offset) / (self.height - 2 * x_offset) * self.image.shape[0])
            y = int( (float(event.pos().x()) - y_offset) / (self.width - 2 * y_offset) * self.image.shape[1])
        
        # switch here for shift-click (emit different signal)
        if QtCore.Qt.Modifier.SHIFT and event.modifiers():
            self.c.mouseSingleShiftClicked.emit((x, y))
        else: 
            self.c.mouseSingleClicked.emit((x, y))

        event.accept()
      
    def mouseDoubleClickEvent(self, event):
        self.setFocus()
        event.accept()
    
class CellPickerGUI(object):
    def setupUi(self, MainWindow, data, mask, cutoff):        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 900)
        self.MainWindow = MainWindow
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.image_widget = MatplotlibWidget(data, parent=self.centralwidget, cutoff=cutoff)
        # note that the widget size is hardcoded in the class (not the best, but at least it's all
        # in the constructor
        
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.image_widget.sizePolicy().hasHeightForWidth())
        
        # custom image
        self.image_widget.setSizePolicy(sizePolicy)
        self.image_widget.setMinimumSize(QtCore.QSize(512, 512))
        self.image_widget.setObjectName("image_widget")
        self.image_widget.setFocus()

        # splitter for radius selector (organizational)
        self.splitter = QtGui.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(10, 460, 111, 51))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.splitter.setChildrenCollapsible(False)
        # label for radius selector
        self.label_2 = QtGui.QLabel(self.splitter)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        # radius selector
        self.dilation_disk = QtGui.QSpinBox(self.splitter)
        self.dilation_disk.setProperty("value", 4)
        self.dilation_disk.setObjectName("dilation_disk")
        
        # splitter for threshold
        self.splitter_2 = QtGui.QSplitter(self.centralwidget)
        self.splitter_2.setGeometry(QtCore.QRect(10, 400, 122, 51))
        self.splitter_2.setFrameShape(QtGui.QFrame.NoFrame)
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setOpaqueResize(False)
        self.splitter_2.setChildrenCollapsible(False)
        self.splitter_2.setObjectName("splitter_2")
        self.splitter_2.setChildrenCollapsible(False)
        # threshold label
        self.label = QtGui.QLabel(self.splitter_2)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        # threshold selector
        self.contrast_threshold = QtGui.QDoubleSpinBox(self.splitter_2)
        self.contrast_threshold.setSingleStep(0.01)
        self.contrast_threshold.setProperty("value", 0.95)
        self.contrast_threshold.setObjectName("contrast_threshold")
        
        # check boxes to switch modes
        # splitter for mode buttons
        self.splitter_3 = QtGui.QSplitter(self.centralwidget)
        self.splitter_3.setGeometry(QtCore.QRect(10, 260, 141, 131))
        self.splitter_3.setOrientation(QtCore.Qt.Vertical)
        self.splitter_3.setObjectName("splitter_3")
        self.splitter_3.setChildrenCollapsible(False)
        # polygon
        self.radioButton_3 = QtGui.QRadioButton(self.splitter_3)
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_3.setText('Polygon Mode: (p)')        
        # square
        self.radioButton = QtGui.QRadioButton(self.splitter_3)
        self.radioButton.setObjectName("radioButton")
        self.radioButton.setText('Square Mode: (s)')        
        # circle
        self.radioButton_4 = QtGui.QRadioButton(self.splitter_3)
        self.radioButton_4.setObjectName("radioButton_4")
        self.radioButton_4.setText('Circel Mode: (c)')        
        # OGB
        self.radioButton_2 = QtGui.QRadioButton(self.splitter_3)
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_2.setText('OGB Mode: (o)')
        # standard
        self.radioButton_5 = QtGui.QRadioButton(self.splitter_3)
        self.radioButton_5.setObjectName("radioButton_5")
        self.radioButton_5.setText('Standard Mode: (x)')
        self.radioButton_5.setChecked(True)
        # button group for mode radio buttons
        self.buttonGroup = QtGui.QButtonGroup()
        self.buttonGroup.addButton(self.radioButton_3, 1)  #Polygon
        self.buttonGroup.addButton(self.radioButton, 2)    #Square
        self.buttonGroup.addButton(self.radioButton_4, 3)  #Circle
        self.buttonGroup.addButton(self.radioButton_2, 4)  #OGB
        self.buttonGroup.addButton(self.radioButton_5, 5)  #Standard
        # mode switch radio button conecctor
        self.radioButton_3.toggled.connect(self.changeMode)
        self.radioButton.toggled.connect(self.changeMode)
        self.radioButton_4.toggled.connect(self.changeMode)
        self.radioButton_2.toggled.connect(self.changeMode)
        self.radioButton_5.toggled.connect(self.changeMode)
        
        # Hot Key Legend
        # splitter for key legend
        self.splitter_4 = QtGui.QSplitter(self.centralwidget)
        self.splitter_4.setGeometry(QtCore.QRect(10, 10, 141, 241))
        self.splitter_4.setOrientation(QtCore.Qt.Vertical)
        self.splitter_4.setObjectName("splitter_4")
        self.splitter_4.setChildrenCollapsible(False)        
        # title
        self.label_10 = QtGui.QLabel(self.splitter_4)
        self.label_10.setObjectName("label_10")
        self.label_10.setText('HOT KEYS:')        
        # P
        self.label_8 = QtGui.QLabel(self.splitter_4)
        self.label_8.setObjectName("label_8")
        self.label_8.setText('Polygon Mode: (p)')       
        # T
        self.label_9 = QtGui.QLabel(self.splitter_4)
        self.label_9.setObjectName("label_9")
        self.label_9.setText('Terminate Poly.: (t)')       
        # S
        self.label_7 = QtGui.QLabel(self.splitter_4)
        self.label_7.setObjectName("label_7")
        self.label_7.setText('Square: (s)')
        # C
        self.label_5 = QtGui.QLabel(self.splitter_4)
        self.label_5.setObjectName("label_5")
        self.label_5.setText('Circle: (c)')        
        # O
        self.label_6 = QtGui.QLabel(self.splitter_4)
        self.label_6.setObjectName("label_6")
        self.label_6.setText('OGB: (o)')        
        # standard (X)
        self.label_4 = QtGui.QLabel(self.splitter_4)
        self.label_4.setObjectName("label_4")
        self.label_4.setText('Standard: (x)')
        
        self.data = data
        if self.data.ndim == 2:
            self.currentBackgroundImage = self.data
            self.frame = 1
        elif self.data.ndim ==3:
            self.currentBackgroundImage = self.data.mean(axis=2)
            self.frame = self.data.shape[2]
        
        # ave/vid slider gui
        
        # slider label
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(160, 780, 150, 16))
        self.label_3.setObjectName("label_3")
        self.label_3.setText('Slide to Frame in Video')
        self.label_3.setVisible(False)
        # jumper label
        self.label_11 = QtGui.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(10, 780, 150, 16))
        self.label_11.setObjectName("label_11")
        self.label_11.setText('Jump to Frame')
        self.label_11.setVisible(False)
        # ave/vid toggle button
        self.checkBox = QtGui.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(10, 600, 200, 20))
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setText('Ave(On)/Vid(Off)')
        self.checkBox.setChecked(True)
        # video frame slidder
        self.horizontalSlider = QtGui.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(160, 800, 750, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.setVisible(False)
        self.horizontalSlider.setMaximum(self.frame-1)
        self.currentFrame = self.horizontalSlider.value()
        
        # jump to video frame
        self.lineEdit = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(10, 800, 113, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setVisible(False)
        
        # ave/vid checkbox connector
        self.checkBox.stateChanged.connect(self.avgBoxClicked)
        
        # connect value in box and slider
        self.horizontalSlider.valueChanged.connect(self.comScrollToLine)
        self.lineEdit.returnPressed.connect(self.comLineToScroll)
        
        #toggle mask button
        self.checkBox_2 = QtGui.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(10, 620, 200, 20))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_2.setText('Toggle Mask')
        self.checkBox_2.setChecked(True)
        self.checkBox_2.stateChanged.connect(self.flipMask)
        
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
        
        if mask is None:
            self.currentMask = np.zeros_like(self.currentBackgroundImage, dtype='uint16')
        else:
            self.currentMask = mask.astype('uint16')
        

        self.listOfMasks = []
        self.listOfMasks.append(self.currentMask)
        self.diskSize = 3
        self.contrastThreshold = 0.95
        self.cellRadius = 3
        self.currentMaskNumber = 1
        
        self.mode = None # can be standard (None), 'poly', or 'square'
        self.modeData = None # or a list of point tuples       

        self.maskOn = True
        self.useNMF = False

        self.makeNewMaskAndBackgroundImage()
    
    # ave/vid is clicked
    def avgBoxClicked(self, state):
        if state == QtCore.Qt.Checked:
            self.currentBackgroundImage = self.data.mean(axis=2)
            self.makeNewMaskAndBackgroundImage()
            self.horizontalSlider.setVisible(False)
            self.lineEdit.setVisible(False)
            
        else:
            self.currentBackgroundImage = self.data[:,:,self.currentFrame]
            self.makeNewMaskAndBackgroundImage()
            self.horizontalSlider.setVisible(True)
            self.lineEdit.setVisible(True)
            
    # changes the mode from radio buttons
    def changeMode(self):
        state = self.buttonGroup.checkedId()
        if state == 1:
            self.mode = 'poly'
        elif state == 2:
            self.mode = 'square'
        elif state == 3:
            self.mode = 'circle'
        elif state == 4:
            self.mode = 'OGB'
        elif state == 5:
            self.mode = None
    
    # connect the box and slider
    def comLineToScroll(self):
        self.currentFrame = int(self.lineEdit.text())
        self.horizontalSlider.setValue(self.currentFrame)
        self.currentBackgroundImage = self.data[:,:,self.currentFrame]
        self.makeNewMaskAndBackgroundImage()
    def comScrollToLine(self):
        self.currentFrame = self.horizontalSlider.value()
        self.lineEdit.setText(str(self.currentFrame))
        self.currentBackgroundImage = self.data[:,:,self.currentFrame]
        self.makeNewMaskAndBackgroundImage()
               
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("MainWindow", "<html><head/><body><p>Cell Radius Size</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("MainWindow", "<html><head/><body><p>Threshold</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))

    # dispatcher method to handle all key presses
    @QtCore.Slot()
    def keyPress(self, keyPressed):
        
        if keyPressed == QtCore.Qt.Key_Return or keyPressed == QtCore.Qt.Key_Enter:
            plt.close('info')
            self.MainWindow.close()
            
        elif keyPressed >= QtCore.Qt.Key_1 and keyPressed <= QtCore.Qt.Key_8:
            # 1 = 49, 8 = 56
            self.currentMaskNumber = int(keyPressed - 48)
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
                self.radioButton_5.setChecked(True)
            else:
                self.clearModeData()
                self.mode = 'poly'
                self.radioButton_3.setChecked(True)

        elif keyPressed == QtCore.Qt.Key_T:
            if self.mode is 'poly':
                self.addPolyCell()
                self.clearModeData()

        elif keyPressed == QtCore.Qt.Key_S:
            if self.mode is 'square':
                self.mode = None
                self.radioButton_5.setChecked(True)
            else:
                self.clearModeData()
                self.mode = 'square'
                self.radioButton.setChecked(True)

        elif keyPressed == QtCore.Qt.Key_C:
            if self.mode is 'circle':
                self.mode = None
                self.radioButton_5.setChecked(True)
            else:
                self.clearModeData()
                self.mode = 'circle'
                self.radioButton_4.setChecked(True)

        elif keyPressed == QtCore.Qt.Key_O:
            if self.mode is 'OGB':
                self.mode = None
                self.radioButton_5.setChecked(True)
            else:
                self.clearModeData()
                self.mode = 'OGB'
                self.radioButton_2.setChecked(True)
        
        elif keyPressed == QtCore.Qt.Key_A:
            if self.checkBox.isChecked == QtCore.Qt.Checked:
                self.currentBackgroundImage = self.data.mean(axis=2)
                self.makeNewMaskAndBackgroundImage()
                self.horizontalSlider.setVisible(False)
                self.lineEdit.setVisible(False)
                self.checkBox.toggle()
            elif self.checkBox.isChecked != QtCore.Qt.Checked:
                self.currentBackgroundImage = self.data[:,:,self.currentFrame]
                self.makeNewMaskAndBackgroundImage()
                self.horizontalSlider.setVisible(True)
                self.lineEdit.setVisible(True)
                self.checkBox.toggle()
                                
        elif keyPressed == QtCore.Qt.Key_K:
            self.correlateLastROI()

        elif keyPressed == QtCore.Qt.Key_I:
            self.updateInfoPanel()

        elif keyPressed == QtCore.Qt.Key_X:
            self.clearModeData()
            self.mode = None
            self.radioButton_5.setChecked(True)

        elif keyPressed == QtCore.Qt.Key_M:
            self.flipMask()

        elif keyPressed == QtCore.Qt.Key_N:
            self.useNMF = not(self.useNMF)

        else:
            pass

    def flipMask(self):
        if self.maskOn:
            self.maskOn = False
            self.currentMask = np.zeros_like(self.currentMask)
            self.makeNewMaskAndBackgroundImage()
        else:
            self.maskOn = True
            self.currentMask = self.listOfMasks[-1].copy()
            self.makeNewMaskAndBackgroundImage()
        self.checkBox_2.setChecked(self.MaskOn)

    def clearModeData(self):
        self.modeData = []

    def lastROI(self):
        if len(self.listOfMasks) is 0:
            print 'no mask!'
            return None
        elif len(self.listOfMasks) is 1:
            print 'only 1 ROI'
            lastROI = self.currentMask.copy()
        else:
            lastROI = np.logical_xor(self.listOfMasks[-1], self.listOfMasks[-2]) 
        return lastROI

    def maskFromROINumber(self, ROI_number=None):
        if ROI_number is None:
            ROI_mask = self.lastROI()
        else:
            ROI_mask = mahotas.label(self.currentMask)[0] == ROI_number

        assert(np.any(ROI_mask))
        return ROI_mask


    def timeCourseROI(self, ROI_mask):
        if self.data.ndim ==3:
            nPixels=np.sum(ROI_mask)
            return np.sum(self.data[ROI_mask,:],axis=0)/nPixels


    def updateInfoPanel(self, ROI_number=None):
        if self.data.ndim == 2:
            print 'No series information!'
            sys.stdout.flush()
            return None

        
        self.infofig = plt.figure('info')
        box_size = 5*self.dilation_disk.value()

        ROI_mask = self.maskFromROINumber(ROI_number)

        # activity plot
        axes1 = self.infofig.add_axes([0.04, 0.6, 0.92, 0.35]) # main axes
        axes1.cla()
        trace = self.timeCourseROI(ROI_mask)

        try:
            self.max_of_trace = max(trace.max(), self.max_of_trace)
        except:
            self.max_of_trace = trace.max()

        axes1 = plt.plot(trace)
        axes1[0].get_axes().set_xlim(0, trace.shape[0])
        axes1[0].get_axes().set_ylim(self.data.min()*0.9, self.max_of_trace*1.1)
        axes1[0].get_axes().set_title('Activity Plot')

        # Mask display
        axes2 = self.infofig.add_axes([0.04, 0.325, 0.2, 0.2]) # inset axes
        axes2.cla()
        axes2 = plt.imshow(self.currentMask + ROI_mask)
        axes2.get_axes().set_yticklabels([])
        axes2.get_axes().set_xticklabels([])
        axes2.get_axes().set_title('Mask')

        # ROI corralation
        axes3 = self.infofig.add_axes([0.28, 0.325, 0.2, 0.2])
        axes3.cla()
        
        resp_mask = (ROI_mask == 1)
        xvals, yvals = zip(*np.argwhere(resp_mask))
        xmin = min(xvals)
        xmax = max(xvals)
        ymin = min(yvals)
        ymax = max(yvals)

        xcenter = (xmax - xmin) / 2 + xmin
        ycenter = (ymax - ymin) / 2 + ymin
        
        local_data = self.data[xcenter-box_size:xcenter+box_size, ycenter-box_size:ycenter+box_size, :]
        x,y,frame = local_data.shape
        
        x,y,frame = local_data.shape
        
        corr_map = np.empty((x,y))
        for xv in range(x):
            for yv in range(y):
                corr_map[xv,yv] = np.correlate(local_data[xv,yv,:]-local_data[xv,yv,:].mean(), trace-trace.mean())
       
        corr_map[np.isnan(corr_map)] = 0
        corr_map = corr_map/corr_map.max()
        
        axes3.imshow(corr_map, vmax=1)
        
        rgba_mask = np.zeros((box_size*2,box_size*2,4))
        rgba_mask[:,:,0] = resp_mask[xcenter-box_size:xcenter+box_size,ycenter-box_size:ycenter+box_size]
        rgba_mask[:,:,1] = resp_mask[xcenter-box_size:xcenter+box_size,ycenter-box_size:ycenter+box_size]
        rgba_mask[:,:,2] = resp_mask[xcenter-box_size:xcenter+box_size,ycenter-box_size:ycenter+box_size]
        rgba_mask[:,:,3] = (resp_mask[xcenter-box_size:xcenter+box_size,ycenter-box_size:ycenter+box_size]>0).astype(int) * .8
        axes3 = plt.imshow(rgba_mask, cmap=mpl.cm.gist_yarg)
        
        axes3.get_axes().set_yticklabels([])
        axes3.get_axes().set_xticklabels([])
        axes3.get_axes().set_title('ROI Corralation')

        # Histogram
        
        axes4 = self.infofig.add_axes([0.52, 0.325, 0.2, 0.2])
        axes4.cla()
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=45)
        axes4.get_axes().set_title('Corralation Histagram')
        axes4 = plt.hist(corr_map.flatten(), range=(0,1), bins=20)

        axes5 = self.infofig.add_axes([0.76, 0.325, 0.2, 0.2])
        axes5.cla()
        axes5.get_axes().set_yticklabels([])
        axes5.get_axes().set_xticklabels([])
        axes5.get_axes().set_title('Local ROI')       
        axes5.imshow(local_data.mean(axis=2), cmap=mpl.cm.gray)
        
        # NMF Modes

        axes6 = self.infofig.add_axes([0.04, 0.025, 0.2, 0.2])
        axes6.cla()
        axes6.get_axes().set_yticklabels([])
        axes6.get_axes().set_xticklabels([])
        axes6.get_axes().set_title('Mode 1')
                
        axes7 = self.infofig.add_axes([0.28, 0.025, 0.2, 0.2])
        axes7.cla()
        axes7.get_axes().set_yticklabels([])
        axes7.get_axes().set_xticklabels([])
        axes7.get_axes().set_title('Mode 2')
        
        axes8 = self.infofig.add_axes([0.52, 0.025, 0.2, 0.2])
        axes8.cla()
        axes8.get_axes().set_yticklabels([])
        axes8.get_axes().set_xticklabels([])
        axes8.get_axes().set_title('Mode 3')
        
        axes9 = self.infofig.add_axes([0.76, 0.025, 0.2, 0.2])
        axes9.cla()
        axes9.get_axes().set_yticklabels([])
        axes9.get_axes().set_xticklabels([])
        axes9.get_axes().set_title('Mode 4')
        
        modes, this_cell, is_cell = self.doLocalNMF(xcenter, ycenter, ROI_mask)

        for i, (mode, t, is_a_cell, ax) in enumerate(zip(np.rollaxis(modes,2,0)[1:], this_cell[1:], is_cell[1:], [axes6, axes7, axes8, axes9])):
            fit_parameters = self.fitgaussian(mode) 
            gaussian_function = self.gaussian(*fit_parameters)
            xcoords = np.mgrid[0:box_size*2,0:box_size*2][0]
            ycoords = np.mgrid[0:box_size*2,0:box_size*2][1]
            fit_data = gaussian_function(xcoords, ycoords)

            ax.imshow(mode)
            ax.contour(fit_data, cmap=mpl.cm.Pastel1)
            ax.set_xlim(0,mode.shape[0])
            ax.set_ylim(0,mode.shape[1])
            if t:
                mode_is_this_cell = 'this cell'
            else:
                mode_is_this_cell = 'not this cell'

            if is_a_cell:
                mode_is_a_cell = 'is a cell'
            else:
                mode_is_a_cell = 'is not a cell'

            ax.set_title(str(i) + ', ' + mode_is_this_cell + ', ' + mode_is_a_cell)

        plt.draw()
    
    def exitHandler(self):
        self.infofig.close()
    
    def gaussian(self, height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

    def moments(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y

    def fitgaussian(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = self.moments(data)
        errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape)) - data)
        p, success = scipy.optimize.leastsq(errorfunction, params)
        return p
            
    def addPolyCell(self):
        if self.maskOn:
            # build poly_mask
            poly_mask = self.maskFromPoints(self.modeData, self.currentMask.shape[0], self.currentMask.shape[1])
            # check if poly_mask interfers with current mask, if so, abort
            if np.any(np.logical_and(poly_mask, self.currentMask)):
                return None

            self.currentMask = self.currentMask.astype('uint16')

            # need center of mass for polygon
            center_of_mass = scipy.ndimage.measurements.center_of_mass(poly_mask)
            modes, this_cell, is_cell = self.doLocalNMF(center_of_mass[0], center_of_mass[1])

            # add poly_mask to mask
            newMask = (poly_mask * self.currentMaskNumber) + self.currentMask
            newMask = newMask.astype('uint16')

            self.listOfMasks.append(newMask)
            self.currentMask = self.listOfMasks[-1]

            sys.stdout.flush()
            self.makeNewMaskAndBackgroundImage()

    def maskFromPoints(self, vertex_list, size_x, size_y):
        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(size_x), np.arange(size_y))
        x, y = x.flatten(), y.flatten()
        point_space = np.vstack((x,y)).T

        poly_mask = nx.points_inside_poly(point_space, vertex_list)
        poly_mask = poly_mask.reshape(size_x, size_y)

        return poly_mask.T
    
    @QtCore.Slot(tuple)
    def addCell(self, eventTuple):
        if self.maskOn:
            if self.data.ndim == 2:
                self.aveData = self.data.copy()
            else:
                self.aveData = self.data.mean(axis = 2)

            x, y = eventTuple
            localValue = self.currentMask[x,y]
            print str(self.mode) + ' ' + 'x: ' + str(x) + ', y: ' + str(y) + ', mask val: ' + str(localValue) 

            # ensure mask is uint16
            self.currentMask = self.currentMask.astype('uint16')

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
                    newMask = newMask.astype('uint16')

                    self.listOfMasks.append(newMask)
                    self.currentMask = self.listOfMasks[-1]
                elif localValue > 0 and self.data.ndim ==3:
                    # update info panel
                    labeledCurrentMask = mahotas.label(self.currentMask.copy())[0]
                    roiNumber = labeledCurrentMask[x, y]
                    self.updateInfoPanel(ROI_number=roiNumber)

                elif localValue == 0:

                    xmin = int(x - self.diskSize)
                    xmax = int(x + self.diskSize)
                    ymin = int(y - self.diskSize)
                    ymax = int(y + self.diskSize)

                    sub_region_image = self.aveData[xmin:xmax, ymin:ymax].copy()
                    #threshold = mahotas.otsu(self.data[xmin:xmax, ymin:ymax].astype('uint16'))

                    # do a gaussian_laplacian filter to find the edges and the center

                    g_l = nd.gaussian_laplace(sub_region_image, 1)  # second argument is a free parameter, std of gaussian
                    g_l = mahotas.dilate(mahotas.erode(g_l>=0))
                    g_l = mahotas.label(g_l)[0]
                    center = g_l == g_l[g_l.shape[0]/2, g_l.shape[0]/2]
                    #edges = mahotas.dilate(mahotas.dilate(mahotas.dilate(center))) - center


                    newCell = np.zeros_like(self.currentMask)
                    newCell[xmin:xmax, ymin:ymax] = center
                    newCell = mahotas.dilate(newCell)

                    if self.useNMF:
                        modes, this_cell, is_cell = self.doLocalNMF(x,y, newCell)

                        roi_as_selected = newCell.copy()

                        # need to add all modes belonging to this cell first,
                        # then remove the ones nearby.

                        # if a mode is a cell and is this cell, add some of it to the ROI
                        for m, t, i in zip(np.rollaxis(modes, 2, 0)[1:], this_cell[1:], is_cell[1:]):
                            mode_thresh = m > mahotas.otsu(m.astype('uint16'))
                            # need to place it in the right place
                            # have x and y
                            mode_width, mode_height = mode_thresh.shape
                            mode_thresh_fullsize = np.zeros_like(newCell)

                            if x <= mode_width/2:
                                x_range = (0, mode_width)
                            elif x > mode_thresh_fullsize.shape[0] - mode_width/2:
                                x_range = (mode_thresh_fullsize.shape[0]-mode_width, mode_thresh_fullsize.shape[0])
                            else:
                                if mode_width % 2 == 0:
                                    x_range = (x-mode_width/2, x+mode_width/2)
                                else:
                                    x_range = (x-mode_width/2, x+mode_width/2+1)

                            if y <= mode_height/2:
                                y_range = (0, mode_height)
                            elif y > mode_thresh_fullsize.shape[1] - mode_height/2:
                                y_range = (mode_thresh_fullsize.shape[1]-mode_height, mode_thresh_fullsize.shape[1])
                            else:
                                if mode_height % 2 == 0:
                                    y_range = (y-mode_height/2, y+mode_height/2)
                                else:
                                    y_range = (y-mode_height/2, y+mode_height/2+1)

                            mode_thresh_fullsize[x_range[0]:x_range[1], y_range[0]:y_range[1]] = mode_thresh

                            if i and t:
                                valid_area = np.logical_and(mahotas.dilate(mahotas.dilate(mahotas.dilate(mahotas.dilate(newCell.astype(bool))))), mode_thresh_fullsize)
                                newCell = np.logical_or(newCell.astype(bool), valid_area)

                        for m, t, i in zip(np.rollaxis(modes, 2, 0)[1:], this_cell[1:], is_cell[1:]):
                            mode_thresh = m > mahotas.otsu(m.astype('uint16'))
                            # need to place it in the right place
                            # have x and y
                            mode_width, mode_height = mode_thresh.shape
                            mode_thresh_fullsize = np.zeros_like(newCell)

                            if x <= mode_width/2:
                                x_range = (0, mode_width)
                            elif x > mode_thresh_fullsize.shape[0] - mode_width/2:
                                x_range = (mode_thresh_fullsize.shape[0]-mode_width, mode_thresh_fullsize.shape[0])
                            else:
                                if mode_width % 2 == 0:
                                    x_range = (x-mode_width/2, x+mode_width/2)
                                else:
                                    x_range = (x-mode_width/2, x+mode_width/2+1)

                            if y <= mode_height/2:
                                y_range = (0, mode_height)
                            elif y > mode_thresh_fullsize.shape[1] - mode_height/2:
                                y_range = (mode_thresh_fullsize.shape[1]-mode_height, mode_thresh_fullsize.shape[1])
                            else:
                                if mode_height % 2 == 0:
                                    y_range = (y-mode_height/2, y+mode_height/2)
                                else:
                                    y_range = (y-mode_height/2, y+mode_height/2+1)

                            mode_thresh_fullsize[x_range[0]:x_range[1], y_range[0]:y_range[1]] = mode_thresh

                            if i and not t:
                                newCell = np.logical_and(newCell.astype(bool), np.logical_not(mahotas.dilate(mode_thresh_fullsize)))

                        newCell = mahotas.close_holes(newCell.astype(bool))
                        self.excludePixels(newCell, 2)

                    newCell = newCell.astype(self.currentMask.dtype)

                    # remove all pixels in and near current mask and filter for ROI size
                    newCell[mahotas.dilate(self.currentMask>0)] = 0
                    newCell = self.excludePixels(newCell, 10)

                    newMask = (newCell * self.currentMaskNumber) + self.currentMask
                    newMask = newMask.astype('uint16')

                    self.listOfMasks.append(newMask.copy())
                    self.currentMask = newMask.copy()

            elif self.mode is 'OGB':
                # build structuring elements
                se = pymorph.sebox()
                se2 = pymorph.sedisk(self.cellRadius, metric='city-block')
                seJunk = pymorph.sedisk(max(np.floor(self.cellRadius/4.0), 1), metric='city-block')
                seExpand = pymorph.sedisk(self.diskSize, metric='city-block')

                 # add a disk around selected point, non-overlapping with adjacent cells
                dilatedOrignal = mahotas.dilate(self.currentMask.astype(bool), Bc=se)
                safeUnselected = np.logical_not(dilatedOrignal)

                # tempMask is 
                tempMask = np.zeros_like(self.currentMask, dtype=bool)
                tempMask[x, y] = True
                tempMask = mahotas.dilate(tempMask, Bc=se2)
                tempMask = np.logical_and(tempMask, safeUnselected)

                # calculate the area we should add to this disk based on % of a threshold
                cellMean = self.aveData[tempMask == 1.0].mean()
                allMeanBw = self.aveData >= (cellMean * float(self.contrastThreshold))

                tempLabel = mahotas.label(np.logical_and(allMeanBw, safeUnselected).astype(np.uint16))[0]
                connMeanBw = tempLabel == tempLabel[x, y]

                connMeanBw = np.logical_and(np.logical_or(connMeanBw, tempMask), safeUnselected).astype(np.bool)
                # erode and then dilate to remove sharp bits and edges

                erodedMean = mahotas.erode(connMeanBw, Bc=seJunk)
                dilateMean = mahotas.dilate(erodedMean, Bc=seJunk)
                dilateMean = mahotas.dilate(dilateMean, Bc=seExpand)

                modes, this_cell, is_cell = self.doLocaNMF(x,y)

                newCell = np.logical_and(dilateMean, safeUnselected)
                newMask = (newCell * self.currentMaskNumber) + self.currentMask
                newMask = newMask.astype('uint16')

                self.listOfMasks.append(newMask.copy())
                self.currentMask = newMask.copy()

            ########## SQUARE MODE 
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

#                    print ((xend+xstart)/2,(yend+ystart)/2)
                    modes, this_cell, is_cell = self.doLocalNMF((xend+xstart)/2,(yend+ystart)/2)

                    # add square_mask to mask
                    newMask = (square_mask * self.currentMaskNumber) + self.currentMask
                    newMask = newMask.astype('uint16')

                    self.listOfMasks.append(newMask)
                    self.currentMask = self.listOfMasks[-1]

                    # clear current mode data
                    self.clearModeData()

            ########## CIRCLE MODE 
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
                circle_mask[x-se_extent:x+se_extent+1, y-se_extent:y+se_extent+1] = se * 1.0
                circle_mask = circle_mask.astype(bool)

                # check if circle_mask interfers with current mask, if so, abort
                if np.any(np.logical_and(circle_mask, mahotas.dilate(self.currentMask.astype(bool)))):
                    return None

                modes, this_cell, is_cell = self.doLocalNMF(x,y, circle_mask)

                # add circle_mask to mask
                newMask = (circle_mask * self.currentMaskNumber) + self.currentMask
                newMask = newMask.astype('uint16')



                self.listOfMasks.append(newMask)
                self.currentMask = self.listOfMasks[-1]

            ########## POLY MODE 
            elif self.mode is 'poly':
                self.modeData.append((x, y))

            sys.stdout.flush()
            self.makeNewMaskAndBackgroundImage()


    def excludePixels(self, image, size_cutoff=1):
        labeled_image = mahotas.label(image)[0]

        for label_id in range(labeled_image.max()+1):
            label_id_index = labeled_image == label_id
            if label_id_index.sum() <= size_cutoff:
                labeled_image[label_id_index] = 0

        return labeled_image>0

    def doLocalNMF(self, x, y, roi, n_comp=5, diskSizeMultiplier=5):
        # do NMF decomposition
        n = NMF(n_components=n_comp, tol=1e-1)

        xmin_nmf = max(0,int(x - self.diskSize*diskSizeMultiplier))
        xmax_nmf = min(int(x + self.diskSize*diskSizeMultiplier), self.data.shape[0])
        ymin_nmf = max(0, int(y - self.diskSize*diskSizeMultiplier))
        ymax_nmf = min(int(y + self.diskSize*diskSizeMultiplier), self.data.shape[1])

        local_roi = roi[xmin_nmf:xmax_nmf, ymin_nmf:ymax_nmf]

        xcenter_nmf = (xmax_nmf - xmin_nmf) / 2
        ycenter_nmf = (ymax_nmf - ymin_nmf) / 2

        reshaped_sub_region_data = self.data[xmin_nmf:xmax_nmf, ymin_nmf:ymax_nmf, :].reshape(xmax_nmf-xmin_nmf * ymax_nmf-ymin_nmf, self.data.shape[2])
        n.fit(reshaped_sub_region_data-reshaped_sub_region_data.min())
        transformed_sub_region_data = n.transform(reshaped_sub_region_data)
        modes = transformed_sub_region_data.reshape(xmax_nmf-xmin_nmf, ymax_nmf-ymin_nmf, n_comp).copy()

        params = []
        this_cell = []
        is_cell = []
        for i, mode in enumerate(np.rollaxis(modes,2,0)):
            # threshold mode
            thresh_mode = (mode.astype('uint16') > mahotas.otsu(mode.astype('uint16'))).astype(int)

            print 'sum:' + str(thresh_mode.sum())

            # fit thresholded mode
            fit_parameters = self.fitgaussian(thresh_mode) 
            fit_height, fit_xcenter, fit_ycenter, fit_xwidth, fit_ywidth  = fit_parameters
            print 'mode ' + str(i) + ' parameters: ' + str(fit_parameters)
            params.append(fit_parameters)

            # is cell-like?
            if self.diskSize*0.25 < fit_xwidth < 2*self.diskSize and self.diskSize*0.25 < fit_ywidth < 2*self.diskSize:
                if thresh_mode.sum()/float(thresh_mode.size) <= 0.25:
                    is_cell.append(True)
            else:
                is_cell.append(False)

            # is this cell?
            if np.linalg.norm(np.array([xcenter_nmf, ycenter_nmf]) - np.array([fit_xcenter, fit_ycenter])) < self.diskSize*1.5:
                this_cell.append(True)
            else:
                this_cell.append(False)

            fit_gaussian = self.gaussian(*fit_parameters)
            xcoords = np.mgrid[0:xmax_nmf-xmin_nmf,0:ymax_nmf-ymin_nmf][0]
            ycoords =  np.mgrid[0:xmax_nmf-xmin_nmf,0:ymax_nmf-ymin_nmf][1]
            fit_data = fit_gaussian(xcoords, ycoords)

        print 'this cell', this_cell
        print 'is cell', is_cell
        print ' '

        return modes, np.array(this_cell), np.array(is_cell)
    
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
        color_mask[:,:,3] = (color_mask[:,:,0:2].sum(axis=2) > 0).astype(float) * 0.4 # alpha value of 40%

        self.image_widget.updateImage(self.currentBackgroundImage, color_mask)
        
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
        
def pickCells(backgroundImage, mask=None, cutoff=0.8):
    """This routine is for interactive picking of cells and editing
    a mask.  It takes two arguments- a numpy array for a background image.
    If backgroundImage is 2d, then that is the image used.  If it is
    3D then we assume it is an image series and average over the 3rd
    dimension.  We can then toggle between the frames of the movie and the
    average image for the background.  The average image is always used for
    calculations, though.

    Optionally, you can pass in a previous mask (a uint16 image, 0 in the
    background and 1+ for any ROIs).

    Finally, you can pass in a floating point number that is a % of the max
    value in the image--- good for when a stray pixel is very bright and
    destroys the dynamic range of the image.

    Returns a mask array.  The mask can have values ranging from 1-8,
    each indicating a different feature.  Use mahotas.label(mask==#) to 
    make a labeled mask for each sub-mask.

    """

    try:
        app = QtGui.QApplication(sys.argv)
    except RuntimeError:
        app = QtCore.QCoreApplication.instance()

    MainWindow = QtGui.QMainWindow()
    gui = CellPickerGUI()
    gui.setupUi(MainWindow, backgroundImage, mask, cutoff)
    MainWindow.show()
    MainWindow.raise_()
    
    app.exec_()
    
    try:
        return gui.currentMask
    except IndexError:
        print 'No mask to return!'
