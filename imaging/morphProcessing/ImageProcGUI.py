import numpy as np
import matplotlib.pyplot as plt
from manipulateGUI import ManipulateGUI

import scipy.ndimage
import pymorph
import gtk

__all__ = ['threshold', 'subGaussian', 'gaussian', 'median', 'gaussianLaplace', 'hpf', 'laplace', 'binaryErode', 'binaryDialate', 'connectedPixelFilter']

class _ImageProcGUI(ManipulateGUI):
    def __init__(self, inputImage, parameterDict, liveUpdate=True):
        self.inputImage = inputImage
        self.outputImage = inputImage.copy()

        super(_ImageProcGUI, self).__init__(parameterDict, liveUpdate)

        self.inputDisplay = self.inputAxis.imshow(self.inputImage)
        self.outputDisplay = self.outputAxis.imshow(self.outputImage)
        self.figure.canvas.draw()

        plt.ion()
        plt.show()

    def redraw_output(self):
        self.outputAxis.cla()
        self.outputAxis.imshow(self.outputImage)
        self.figure.canvas.draw()
        
    def on_return_clicked(self, widget):
        self.window.destroy()
        gtk.main_quit()

    def eval_function(self):
        #sub class implement
        pass

#################################

class _Threshold(_ImageProcGUI):
    def __init__(self, inputImage, parameterDict=None, liveUpdate=True):
        self.inputImage = inputImage
        if parameterDict:
            self.parameterDict = parameterDict
        else:
            self.parameterDict = {'Threshold Level':{'type':'range', 'lower':0, 'step':1, 'upper':inputImage.max(), 'value':1}}                    
        self.liveUpdate = liveUpdate

        super(_Threshold, self).__init__(self.inputImage, self.parameterDict, self.liveUpdate)

    def eval_function(self):
        self.binaryArray = pymorph.threshad(self.inputImage,self.parameterDict['Threshold Level']['value'],self.inputImage.max())
        self.outputImage = self.inputImage.copy()
        self.outputImage[np.logical_not(self.binaryArray)] = 0

        self.redraw_output()
        
    def on_return_clicked(self, widget):
        self.window.destroy()
        gtk.main_quit()

def threshold(inputImage, parameterDict=None, liveUpdate=True):
    gui = _Threshold(inputImage, parameterDict=None, liveUpdate=True)
    gtk.main()

    print gui.parameterDict
    
    return gui.outputImage

#################################


class _SubGaussian(_ImageProcGUI):
    def __init__(self, inputImage, parameterDict=None, liveUpdate=True):
        self.inputImage = inputImage.astype('float')
        if parameterDict:
            self.parameterDict = parameterDict
        else:
            self.parameterDict = {'sigma':{'type':'range', 'lower':0, 'step':0.1, 'upper':10, 'value':1}}
        self.liveUpdate = liveUpdate

        super(_SubGaussian, self).__init__(self.inputImage, self.parameterDict, self.liveUpdate)

    def eval_function(self):
        self.blurred = scipy.ndimage.gaussian_filter(self.inputImage,sigma=self.parameterDict['sigma']['value'])
        self.outputImage = self.inputImage - self.blurred

        self.redraw_output()
        
    def on_return_clicked(self, widget):
        self.window.destroy()
        gtk.main_quit()

def subGaussian(inputImage, parameterDict=None, liveUpdate=True):
    gui = _SubGaussian(inputImage, parameterDict=None, liveUpdate=True)
    gtk.main()
    
    print gui.parameterDict
    
    return gui.outputImage

################################

class _Gaussian(_ImageProcGUI):
    def __init__(self, inputImage, parameterDict=None, liveUpdate=True):
        self.inputImage = inputImage
        if parameterDict:
            self.parameterDict = parameterDict
        else:
            self.parameterDict = {'sigma':{'type':'range', 'lower':0, 'step':.01, 'upper':4, 'value':1}}
        self.liveUpdate = liveUpdate

        super(_Gaussian, self).__init__(self.inputImage, self.parameterDict, self.liveUpdate)

    def eval_function(self):
        self.filtered = scipy.ndimage.gaussian_filter(self.inputImage,sigma=self.parameterDict['sigma']['value'])
        self.outputImage = self.filtered

        self.redraw_output()
        
    def on_return_clicked(self, widget):
        self.window.destroy()
        gtk.main_quit()

def gaussian(inputImage, parameterDict=None, liveUpdate=True):
    gui = _Gaussian(inputImage, parameterDict=None, liveUpdate=True)
    gtk.main()
    
    print gui.parameterDict
    
    return gui.outputImage

################################

class _Median(_ImageProcGUI):
    def __init__(self, inputImage, parameterDict=None, liveUpdate=False):
        self.inputImage = inputImage
        if parameterDict:
            self.parameterDict = parameterDict
        else:
            self.parameterDict = {'size':{'type':'range', 'lower':1, 'step':1, 'upper':20, 'value':1}}
        self.liveUpdate = liveUpdate

        super(_Median, self).__init__(self.inputImage, self.parameterDict, self.liveUpdate)

    def eval_function(self):
        self.filtered = scipy.ndimage.median_filter(self.inputImage,size=self.parameterDict['size']['value'])
        self.outputImage = self.filtered

        self.redraw_output()
        
    def on_return_clicked(self, widget):
        self.window.destroy()
        gtk.main_quit()

def median(inputImage, parameterDict=None, liveUpdate=True):
    gui = _Median(inputImage, parameterDict=None, liveUpdate=True)
    gtk.main()
    
    print gui.parameterDict
    
    return gui.outputImage

################################

class _GaussianLaplace(_ImageProcGUI):
    def __init__(self, inputImage, parameterDict=None, liveUpdate=True):
        self.inputImage = inputImage
        if parameterDict:
            self.parameterDict = parameterDict
        else:
            self.parameterDict = {'sigma':{'type':'range', 'lower':0, 'step':.01, 'upper':5, 'value':0}}
        self.liveUpdate = liveUpdate

        super(_GaussianLaplace, self).__init__(self.inputImage, self.parameterDict, self.liveUpdate)

    def eval_function(self):
        self.filtered = scipy.ndimage.gaussian_laplace(self.inputImage,sigma=self.parameterDict['sigma']['value'])
        self.outputImage = self.filtered

        self.redraw_output()
        
    def on_return_clicked(self, widget):
        self.window.destroy()
        gtk.main_quit()

def gaussianLaplace(inputImage, parameterDict=None, liveUpdate=True):
    gui = _GaussianLaplace(inputImage, parameterDict=None, liveUpdate=True)
    gtk.main()
    
    print gui.parameterDict
    
    return gui.outputImage

class _HPF(_ImageProcGUI):
    def __init__(self, inputImage, parameterDict=None, liveUpdate=True):
        self.inputImage = inputImage
        if parameterDict:
            self.parameterDict = parameterDict
        else:
            self.parameterDict = {'sigma':{'type':'range', 'lower':0, 'step':1, 'upper':20, 'value':1}}
        self.liveUpdate = liveUpdate

        super(_HPF, self).__init__(self.inputImage, self.parameterDict, self.liveUpdate)

    def eval_function(self):
        self.filtered = scipy.ndimage.gaussian_filter(self.inputImage,sigma=self.parameterDict['sigma']['value'])
        self.outputImage = self.filtered
        
        self.redraw_output()
        
    def on_return_clicked(self, widget):
        self.window.destroy()
        gtk.main_quit()

def hpf(inputImage, parameterDict=None, liveUpdate=True):
    gui = _HPF(inputImage, parameterDict=None, liveUpdate=True)
    gtk.main()
    
    print gui.parameterDict
    
    return gui.outputImage

################################

class _Laplace(_ImageProcGUI):
    def __init__(self, inputImage, parameterDict=None, liveUpdate=True):
        self.inputImage = inputImage
        if parameterDict:
            self.parameterDict = parameterDict
        else:
            self.parameterDict = {'sigma':{'type':'range', 'lower':0, 'step':1, 'upper':20, 'value':1}}
        self.liveUpdate = liveUpdate

        super(_Gaussian, self).__init__(self.inputImage, self.parameterDict, self.liveUpdate)

    def eval_function(self):
        self.filtered = scipy.ndimage.laplace(self.inputImage,sigma=self.parameterDict['sigma']['value'])
        self.outputImage = self.filtered

        self.redraw_output()
        
    def on_return_clicked(self, widget):
        self.window.destroy()
        gtk.main_quit()

def laplace(inputImage, parameterDict=None, liveUpdate=True):
    gui = _Laplace(inputImage, parameterDict=None, liveUpdate=True)
    gtk.main()
    
    print gui.parameterDict
    
    return gui.outputImage

################################

class _BinaryErode(_ImageProcGUI):
    def __init__(self, inputImage, parameterDict=None, liveUpdate=True, se=None):
        self.inputImage = inputImage > 0
        if parameterDict:
            self.parameterDict = parameterDict
        else:
            self.parameterDict = {'iterations':{'type':'range', 'lower':1, 'step':1, 'upper':10, 'value':1}}

        if se:
            self.se = se
        else:
            self.se = np.array([[0,1,0],[1,1,1],[0,1,0]])
        self.liveUpdate = liveUpdate

        super(_BinaryErode, self).__init__(self.inputImage, self.parameterDict, self.liveUpdate)

    def eval_function(self):
        self.filtered = scipy.ndimage.binary_erosion(self.inputImage,structure=self.se,iterations=np.floor(self.parameterDict['iterations']['value']))
        self.outputImage = self.filtered

        self.redraw_output()
        
    def on_return_clicked(self, widget):
        self.window.destroy()
        gtk.main_quit()

def binaryErode(inputImage, parameterDict=None, liveUpdate=True):
    gui = _BinaryErode(inputImage, parameterDict=None, liveUpdate=True)
    gtk.main()
    
    print gui.parameterDict
    
    return gui.outputImage

################################

class _BinaryDilate(_ImageProcGUI):
    def __init__(self, inputImage, parameterDict=None, liveUpdate=True, se=None):
        self.inputImage = inputImage > 0
        if parameterDict:
            self.parameterDict = parameterDict
        else:
            self.parameterDict = {'iterations':{'type':'range', 'lower':1, 'step':1, 'upper':20, 'value':1}}

        if se:
            self.se = se
        else:
            self.se = np.array([[0,1,0],[1,1,1],[0,1,0]])
        self.liveUpdate = liveUpdate

        super(_BinaryDilate, self).__init__(self.inputImage, self.parameterDict, self.liveUpdate)

    def eval_function(self):
        self.filtered = scipy.ndimage.binary_erosion(self.inputImage,structure=self.se,iterations=self.parameterDict['iterations']['value'])
        self.outputImage = self.filtered

        self.redraw_output()
        
    def on_return_clicked(self, widget):
        self.window.destroy()
        gtk.main_quit()

def binaryDialate(inputImage, parameterDict=None, liveUpdate=True):
    gui = _BinaryDilate(inputImage, parameterDict=None, liveUpdate=True)
    gtk.main()
    
    print gui.parameterDict
    
    return gui.outputImage

################################

class _ConnectedPixelFilter(_ImageProcGUI):
    def __init__(self, inputImage, parameterDict=None, liveUpdate=True):
        self.inputImage = inputImage
        self.outputImage = self.inputImage.copy()
        
        # precompute index shit
        self.index = []
        for i in range(self.outputImage.max()):
            self.index.append(((self.outputImage == i), (self.outputImage == i).sum()))

        if parameterDict:
            self.parameterDict = parameterDict
        else:
            self.parameterDict = {'lower cutoff':{'type':'range', 'lower':0, 'step':1, 'upper':25, 'value':0}}
                                  #                                  'upper cutoff':{'type':'range', 'lower':1, 'step':1, 'upper':100, 'value':100}}
        self.liveUpdate = liveUpdate

        super(_ConnectedPixelFilter, self).__init__(self.inputImage, self.parameterDict, self.liveUpdate)

    def eval_function(self):
    # object elim code
    # whole function takes a labeled image
        self.outputImage = self.inputImage.copy()
        for i in range(self.outputImage.max()):
            if (self.parameterDict['lower cutoff']['value'] >= self.index[i][1]):# or self.parameterDict['upper cutoff']['value'] <= self.index[i][1]):
                self.outputImage[self.index[i][0]]=0

        self.redraw_output()
        
    def on_return_clicked(self, widget):
        self.window.destroy()
        gtk.main_quit()

def connectedPixelFilter(inputImage, parameterDict=None, liveUpdate=True):
    gui = _ConnectedPixelFilter(inputImage, parameterDict=None, liveUpdate=True)
    gtk.main()
    
    print gui.parameterDict
    
    return gui.outputImage

################################
