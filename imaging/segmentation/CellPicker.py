import matplotlib as mpl

from PySide import QtCore, QtGui
import sys
mpl.rcParams['backend.qt4']='PySide'
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np

import scipy.ndimage as nd

import pymorph
import mahotas

import matplotlib.nxutils as nx

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
    
    #signal
    def keyPressEvent(self, event):
        # self.c.keyPressed.emit(event.text())
        self.c.keyPressed.emit(event.key())
        event.accept()
        
    #signal
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
        
        #custom image
        self.image_widget.setSizePolicy(sizePolicy)
        self.image_widget.setMinimumSize(QtCore.QSize(512, 512))
        self.image_widget.setObjectName("image_widget")
        self.image_widget.setFocus()

        #splitter for radius selector (organizational)
        self.splitter = QtGui.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(10, 460, 111, 51))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.splitter.setChildrenCollapsible(False)
        #label for radius selector
        self.label_2 = QtGui.QLabel(self.splitter)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        #radius selector
        self.dilation_disk = QtGui.QSpinBox(self.splitter)
        self.dilation_disk.setProperty("value", 3)
        self.dilation_disk.setObjectName("dilation_disk")
        
        #splitter for threshold
        self.splitter_2 = QtGui.QSplitter(self.centralwidget)
        self.splitter_2.setGeometry(QtCore.QRect(10, 400, 122, 51))
        self.splitter_2.setFrameShape(QtGui.QFrame.NoFrame)
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setOpaqueResize(False)
        self.splitter_2.setChildrenCollapsible(False)
        self.splitter_2.setObjectName("splitter_2")
        self.splitter_2.setChildrenCollapsible(False)
        #threshold label
        self.label = QtGui.QLabel(self.splitter_2)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        #threshold selector
        self.contrast_threshold = QtGui.QDoubleSpinBox(self.splitter_2)
        self.contrast_threshold.setSingleStep(0.01)
        self.contrast_threshold.setProperty("value", 0.95)
        self.contrast_threshold.setObjectName("contrast_threshold")
        
        #check boxes to switch modes
        #splitter for mode buttons
        self.splitter_3 = QtGui.QSplitter(self.centralwidget)
        self.splitter_3.setGeometry(QtCore.QRect(10, 260, 141, 131))
        self.splitter_3.setOrientation(QtCore.Qt.Vertical)
        self.splitter_3.setObjectName("splitter_3")
        self.splitter_3.setChildrenCollapsible(False)
        #polygon
        self.radioButton_3 = QtGui.QRadioButton(self.splitter_3)
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_3.setText('Polygon Mode: (p)')        
        #square
        self.radioButton = QtGui.QRadioButton(self.splitter_3)
        self.radioButton.setObjectName("radioButton")
        self.radioButton.setText('Square Mode: (s)')        
        #circle
        self.radioButton_4 = QtGui.QRadioButton(self.splitter_3)
        self.radioButton_4.setObjectName("radioButton_4")
        self.radioButton_4.setText('Circel Mode: (c)')        
        #OGB
        self.radioButton_2 = QtGui.QRadioButton(self.splitter_3)
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_2.setText('OGB Mode: (o)')
        #standard
        self.radioButton_5 = QtGui.QRadioButton(self.splitter_3)
        self.radioButton_5.setObjectName("radioButton_5")
        self.radioButton_5.setText('Standard Mode: (x)')
        self.radioButton_5.setChecked(True)
        #button group for mode radio buttons
        self.buttonGroup = QtGui.QButtonGroup()
        self.buttonGroup.addButton(self.radioButton_3, 1)  #Polygon
        self.buttonGroup.addButton(self.radioButton, 2)    #Square
        self.buttonGroup.addButton(self.radioButton_4, 3)  #Circle
        self.buttonGroup.addButton(self.radioButton_2, 4)  #OGB
        self.buttonGroup.addButton(self.radioButton_5, 5)  #Standard
        #mode switch radio button conecctor
        self.radioButton_3.toggled.connect(self.changeMode)
        self.radioButton.toggled.connect(self.changeMode)
        self.radioButton_4.toggled.connect(self.changeMode)
        self.radioButton_2.toggled.connect(self.changeMode)
        self.radioButton_5.toggled.connect(self.changeMode)
        
        #Hot Key Legend
        #splitter for key legend
        self.splitter_4 = QtGui.QSplitter(self.centralwidget)
        self.splitter_4.setGeometry(QtCore.QRect(10, 10, 141, 241))
        self.splitter_4.setOrientation(QtCore.Qt.Vertical)
        self.splitter_4.setObjectName("splitter_4")
        self.splitter_4.setChildrenCollapsible(False)        
        #title
        self.label_10 = QtGui.QLabel(self.splitter_4)
        self.label_10.setObjectName("label_10")
        self.label_10.setText('HOT KEYS:')        
        #P
        self.label_8 = QtGui.QLabel(self.splitter_4)
        self.label_8.setObjectName("label_8")
        self.label_8.setText('Polygon Mode: (p)')       
        #T
        self.label_9 = QtGui.QLabel(self.splitter_4)
        self.label_9.setObjectName("label_9")
        self.label_9.setText('Terminate Poly.: (t)')       
        #S
        self.label_7 = QtGui.QLabel(self.splitter_4)
        self.label_7.setObjectName("label_7")
        self.label_7.setText('Square: (s)')
        #C
        self.label_5 = QtGui.QLabel(self.splitter_4)
        self.label_5.setObjectName("label_5")
        self.label_5.setText('Circle: (c)')        
        #O
        self.label_6 = QtGui.QLabel(self.splitter_4)
        self.label_6.setObjectName("label_6")
        self.label_6.setText('OGB: (o)')        
        #standard (X)
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
        
        #ave/vid slider gui
        
        #slidder label
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(160, 780, 150, 16))
        self.label_3.setObjectName("label_3")
        self.label_3.setText('Slide to Frame in Video')
        self.label_3.setVisible(False)
        #jumper label
        self.label_11 = QtGui.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(10, 780, 150, 16))
        self.label_11.setObjectName("label_11")
        self.label_11.setText('Jump to Frame')
        self.label_11.setVisible(False)
        #ave/vid toggel button
        self.checkBox = QtGui.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(10, 600, 200, 20))
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setText('Ave(On)/Vid(Off)')
        self.checkBox.setChecked(True)
        #video frame slidder
        self.horizontalSlider = QtGui.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(160, 800, 750, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.setVisible(False)
        self.horizontalSlider.setMaximum(self.frame-1)
        self.currentFrame = self.horizontalSlider.value()
        
        #jump to video frame
        self.lineEdit = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(10, 800, 113, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setVisible(False)
        
        #ave/vid checkbox connector
        self.checkBox.stateChanged.connect(self.boxClicked)
        
        #connect value in box and slider
        self.horizontalSlider.valueChanged.connect(self.comScroleToLine)
        self.lineEdit.returnPressed.connect(self.comLineToScrole)
        
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

        self.makeNewMaskAndBackgroundImage()
    
    #ave/vid is clicked
    def boxClicked(self, state):
        if state == QtCore.Qt.Checked:
            self.currentBackgroundImage = self.data.mean(axis=2)
            self.makeNewMaskAndBackgroundImage()
            self.horizontalSlider.setVisible(False)
            self.lineEdit.setVisible(False)
            
        else:
            print self.data.shape
            self.currentBackgroundImage = self.data[:,:,self.currentFrame]
            self.makeNewMaskAndBackgroundImage()
            self.horizontalSlider.setVisible(True)
            self.lineEdit.setVisible(True)
            
    #changes the mode from radio buttons
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
    
    #connect the box and slidder
    def comLineToScrole(self):
        self.currentFrame = int(self.lineEdit.text())
        self.horizontalSlider.setValue(self.currentFrame)
        self.currentBackgroundImage = self.data[:,:,self.currentFrame]
        self.makeNewMaskAndBackgroundImage()
    def comScroleToLine(self):
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
        else:
            pass

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

        # we have two figures, a trace, and masks
        ROI_mask = self.maskFromROINumber(ROI_number)

        self.infofig = plt.figure('info')
        
        axes1 = self.infofig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
        axes1.cla()
        trace = self.timeCourseROI(ROI_mask)

        try:
            self.max_of_trace = max(trace.max(), self.max_of_trace)
        except:
            self.max_of_trace = trace.max()

        axes1 = plt.plot(trace)
        axes1[0].get_axes().set_xlim(0, trace.shape[0])
        axes1[0].get_axes().set_ylim(self.data.min()*0.9, self.max_of_trace*1.1)

        axes2 = self.infofig.add_axes([0.8, 0.75, 0.2, 0.2]) # inset axes
        axes2.cla()
        axes2 = plt.imshow(self.currentMask + ROI_mask)
        axes2.get_axes().set_yticklabels([])
        axes2.get_axes().set_xticklabels([])

        plt.draw()
        

    def averageCorrCoefScore(self, series, mask):
        coef_matrix = np.corrcoef(series[mask, :])
        return coef_matrix[np.triu_indices(coef_matrix.shape[0],1)].mean()

    def addRandomPixelsToEdge(self, mask):
        mask = mask.astype(bool)
        ring = mahotas.dilate(mask) - mask
        rand_ring = np.logical_and(ring, np.random.random((ring.shape[0], ring.shape[1]))>0.5)

        return rand_ring

    def conditionallyDilateMask(self, mask, series, cutoff=0.5, num_guesses=750, topcut=50):
        # assuming mask is a binary array of just one ROI
    
        # cut down the size of the array to a region just around the ROI-
        # speeds up correlation calculation below
        sub_xmin = np.where(mask)[0].min() - 2
        sub_xmax = np.where(mask)[0].max() + 2 
        sub_ymin = np.where(mask)[1].min() - 2
        sub_ymax = np.where(mask)[1].max() + 2
        sub_series = series[sub_xmin:sub_xmax, sub_ymin:sub_ymax, :]
        sub_mask = mask[sub_xmin:sub_xmax, sub_ymin:sub_ymax] > 0

        core = np.corrcoef(sub_series[sub_mask>0,:])
        print 'core corr coef: ' + str(np.mean(core[np.triu_indices(core.shape[0], 1)]))

        # generate a population of possible masks and their average correlation coeffecients
        num_guesses = num_guesses
        masks = np.zeros((sub_series.shape[0], sub_series.shape[1], num_guesses))
        corrs = np.zeros(num_guesses)
        for i in range(num_guesses):
            masks[:,:,i] = self.addRandomPixelsToEdge(sub_mask) + sub_mask>0
            corrs[i] = self.averageCorrCoefScore(sub_series, masks[:,:,i]>0)
    
        # sort masks based on corr coef score
        # and return thresholded average of top 50
        top_population = masks[:,:,np.argsort(corrs)[-topcut:-1]].mean(axis=2)
        top_population_thresh = top_population > cutoff

        # place new mask in place
        mask[sub_xmin:sub_xmax, sub_ymin:sub_ymax] = top_population_thresh

        return mask > 0 


    def maskFromPoints(self, vertex_list, size_x, size_y):
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
        poly_mask = self.maskFromPoints(self.modeData, self.currentMask.shape[0], self.currentMask.shape[1])
        # check if poly_mask interfers with current mask, if so, abort
        if np.any(np.logical_and(poly_mask, self.currentMask)):
            return None

        self.currentMask = self.currentMask.astype('uint16')

        # add poly_mask to mask
        newMask = (poly_mask * self.currentMaskNumber) + self.currentMask
        newMask = newMask.astype('uint16')

        self.listOfMasks.append(newMask)
        self.currentMask = self.listOfMasks[-1]

        sys.stdout.flush()
        self.makeNewMaskAndBackgroundImage()

    
    

    @QtCore.Slot(tuple)
    def addCell(self, eventTuple):
        
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

#                newCell[xmin:xmax, ymin:ymax] = mahotas.erode(np.logical_not(sub_region_image > threshold))
#                newCell = mahotas.dilate(newCell).astype(int)
#                newCell = self.conditionallyDilateMask(newCell, self.series).astype(int)

                # remove all pixels in and near current mask
                newCell[mahotas.dilate(self.currentMask>0)] = 0

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
        color_mask[:,:,3] = (color_mask[:,:,0:2].sum(axis=2) > 0).astype(float) * 0.4 # alpha value of 0.4

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
