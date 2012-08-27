# Core import / export routines and utilities for all imaging analysis modules
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.signal as sig

from IPython.display import HTML
from IPython.core import display
import IPython.core.pylabtools as pylabtools

import tempfile

import subprocess
import tifffile
try:
    from PIL import Image
except ImportError:
    import Image

__all__ = ['play', 'embed', 'save3dNPArrayAsMovie', 'writeMultiImageStack', 'imread', 'imreadStack', 'imsave', 'imview', 'splitAndResaveChannels', 'readMultiImageTifStack', 'readImagesFromList', 'downsample2d', 'downsample3d']


def save3dNPArrayAsMovie(fileName, npArray, frameRate=6):
    """This method saves a 3d numpy array to a m4v current working directory, using temporary
    tiffs as an intervening step.  See writeMultiImageStack for details.

    A bit defunct due to imview (one can save as an .avi from imagej), but here for reference.
    
    :param fileName: a filename for the movie.  '.m4v' is appended.
    :param npArray: a 3d numpy array
    :param frameRate: optional, defaults to 6
    """
    writeMultiImageStack(fileName,npArray)
    os.system('ffmpeg -r %s -i %s' % (frameRate, fileName) + '_%06d.tif ' + '%s.m4v' % fileName) # you should have ffmpeg by default
    os.system('rm -f %s_*tif' % fileName)

def writeMultiImageStack(fileName, npArray):
    """This method saves a 3d numpy array as a number of .tif files.

    Files are named 'filename_XXXX.tif' where XXXX is an incrementing integer.

    A bit defunct due to imview (one can save as an .avi from imagej), but here for reference.
    
    :param fileName: a filename for the images.  '_XXXX.tif' is appended.
    :param npArray: a 3d numpy array
    """

    for i in range(npArray.shape[2]):
        frame = npArray[:,:,i].T
        imsave(frame, '%s_%06d.tif' % (fileName, i))

def imread(filename, usePIL=False):
    """Simple wrapper to read various file formats.

    We tend to work with single channel tiff files, and as such use tifffile's imread function.
    We've wrapped tifffiles read function to account for the differences in default
    image dimension ordering.  By convention, we use x,y,frame.  The major advantages of
    tifffile are 1) speed and 2) the ability to read multiframe tiffs.

    This function falls back to PIL if the file's mimetype is not 'image/tiff', or if the usePIL flag
    is true.  In this case, the image is X by Y by 4 (R, G, B, Alpha).

    Note that you can convert between numpy arrays and PIL Images as follows:

    import numpy, Image

    i = Image.open('lena.jpg')
    a = numpy.asarray(i) # a is readonly
    i = Image.fromarray(a)

    :param filename: string of the file to load
    :param usePIL: boolean flag to return a PIL Image instance instead of a numpy array
    :returns:  array OR image.  array is a numpy array representation of file.  image is a PIL Image instance.
    """

    filetype = subprocess.Popen("/usr/bin/file -I %s" % filename, shell=True, stdout=subprocess.PIPE).communicate()[0]
    
    if (filetype.find('image/tiff') is not -1) and (not usePIL): # this is a tiff file?  if so, use tifffile
        array=tifffile.imread(filename)
        if len(array.shape) == 3:
            array=np.transpose(array, [1,2,0])
        return array
    else: # otherwise, use PIL
        image = Image.open(filename)
        return image

def imreadStack(filenameList):
    """Simple wrapper to read a list of image series tiffs into a stack.

    Note that this function assumes all images are the same size.

    We tend to work with single channel tiff files, and as such use tifffile's imread function.
    We've wrapped tifffiles read function to account for the differences in default
    image dimension ordering.  By convention, we use x,y,frame.  The major advantages of
    tifffile are 1) speed and 2) the ability to read multiframe tiffs.

    :param filenameList: list of strings representing the files to load
    :returns:  4d numpy array
    """

    firstImageSeries = tifffile.imread(filenameList[0])
    if len(firstImageSeries.shape) == 3:
	firstImageSeries=np.transpose(firstImageSeries, [1,2,0])

    imageStack = np.zeros((firstImageSeries.shape[0], firstImageSeries.shape[1], firstImageSeries.shape[2], len(filenameList)))

    for i, fileName in enumerate(filenameList):
	array=tifffile.imread(fileName)
	if len(array.shape) == 3:
	    array=np.transpose(array, [1,2,0])
        imageStack[:,:,:,i] = array
    return imageStack

def imsave(npArray, filename):
    """Simple for tifffile's imsave to account for our x : y : frame representation.  Can
    take either 2 or 3d numpy arrays.
    
    :param npArray: 2d or 3d numpy array to save.
    :param filename: string of the name to save, ie: 'image.tif'
    """
    if len(npArray.shape) == 3:
        tifffile.imsave(filename, np.transpose(npArray, [2,0,1]))
    else:
        tifffile.imsave(filename, npArray)


def play(npArray, frameRate = 6):
    """IPython Notebook based interface for playing a 3d numpy array using HTML5 and the Ipython HTML() function
    
    Requires ffmpeg to be installed.

    :param npArray: 3d numpy array
    :param frameRate: integer framerate value, defaults to 6
    """
    # ffmpeg is fussy- images must be in the right format to encode right
    uint8_array = np.uint8(npArray.astype('float').copy() / float(npArray.max()) * 2**8-1)
    
    fileName = 'temp'
    temp_dir = tempfile.mkdtemp()
    
    # save the jpeg frames
    for i in range(uint8_array.shape[2]):
        im = Image.fromarray(uint8_array[:,:,i])
        im.save(os.path.join(temp_dir, '%s_%06d.jpg' % (fileName, i)), format='jpeg')

    # encode the video
    command = 'ffmpeg -y -r %s -i %s' % (frameRate, fileName) + '_%06d.jpg ' + '%s.webm' % fileName
    handle = subprocess.Popen(command, shell=True, cwd=temp_dir)
    handle.wait()

    # build the appropriate html from the video file
    video = open(os.path.join(temp_dir, 'temp.webm'), 'rb').read()
    video_encoded = video.encode('base64')
    video_tag = '<video controls alt="test" src="data:video/webm;base64,{0}">'.format(video_encoded)
    
    # kill the temp files
    subprocess.Popen('rm -rf ' + temp_dir, shell=True)

    return HTML(data=video_tag)

def embed():
    """Simple method for embedding all current figures in the current ipython notebook cell.
    Use at the end of a cell.  Only works when the ipython notebook has been started with '--pylab'
    (note: NOT '--pylab=inline')"""

    # getfigs = pylabtools.getfigs

    display.display(*pylabtools.getfigs())
    plt.close('all')


def imview(npArray, timeOut=8):
    """Function to view the numpy array in imageJ.  This function currently only works on OS X.
    Writes the numpy array to a temporary tif file, launches a new imageJ instance, then waits for 8
    seconds to delete the temp file.

    This could clearly be made more robust, checking paths and such.  Requires imageJ to be in the
    /Applications folder and java to be installed.

    :param npArray: 2d or 3d numpy array to view in imageJ.
    """
	
    tempname = str(np.random.randint(100000000))+'.tif'
    imsave(npArray.astype('uint16'), tempname)
    if os.path.exists('/Applications/ImageJ/ImageJ64.app/Contents/Resources/Java/ij.jar'):
	handle = subprocess.Popen(['java', '-jar', '-Xmx1024m', '/Applications/ImageJ/ImageJ64.app/Contents/Resources/Java/ij.jar', tempname])
    elif os.path.exists('/Applications/ImageJ/ImageJ.app/Contents/Resources/Java/ij.jar'):
	handle = subprocess.Popen(['java', '-jar', '-Xmx1024m', '/Applications/ImageJ/ImageJ.app/Contents/Resources/Java/ij.jar', tempname])
    else:
	print "bah, you don't have imagej installed, do you?  aborting..."
    time.sleep(8)
    os.system('rm -f %s' % tempname)

def splitAndResaveChannels(filename, numChannels=2):
    """This function loads a tiff file and resaves it to individual channels, assuming that
    the channels are interlaced on a frame by frame basis, ie: in a two-channel image,
    every other frame is from one channel or the other.

    Resaves the images with the suffix '_chanX.tif' where X is the channel number.

    :param filename: string of the name to load, ie: 'image.tif'
    :param numChannels: number of interlaced channels in the image file.
    """
    image = imread(filename)
    for i in range(numChannels):
        imsave(image[:,:,i::numChannels], filename[:-4]+'_chan' + str(i) + '.tif')

def readMultiImageTifStack(textInName):
    """This function loads all images in a directory that contain a string in part of
    their name, and returns a 3d np array with dimensions corresponding to X,Y, and frameNumber.
    In the case of a single image, it returns a 2d numpy array.

    :param textInName: substring to search for
    :returns: numpy array
    """
    files = glob.glob('*%s*' % textInName)
    npArray = readImagesFromList(files)
    return npArray

def readImagesFromList(listOfFiles):
    """This function takes a list of filename strings, reads them in and concatenates them
    with each other to form a 3d numpy array.
    
    We are making a big assumption that all of these images are monochromatic
    and single frame tifs (like from PrarieView or MetaMorph)

    Generally one wouldn't use this function, but instead use readMultiImageTifStack.

    :param listOfFiles: a list of filename strings like that generated by readMultiImageTifStack
    :returns: 2 or 3d numpy array
    """
    for i, file in enumerate(listOfFiles):
        image = np.squeeze(tifffile.imread(file))
        image = np.expand_dims(image, axis=2)
        if i==0:
            imageArray = image
        else:
            imageArray = np.concatenate((imageArray, image), axis=2)

    try:
        return imageArray
    except:
        return []


def npArrayFromClipboard():
    """This function pulls information off the OS X clipboard and builds a numpy array.
    Generally, this would be data copied from Excel (tab seperated cols, return seperated rows)

    :returns: numpy array of appropriate dimension.
    """
    p = subprocess.Popen(['pbpaste'], stdout=subprocess.PIPE)
    output = p.communicate()
    return np.array([i.split('\t') for i in output[0].split('\r')])
        
def openArrayInExcel(A):
    """This function writes a temporary csv file from a 1 or 2D numpy array
    and uses the 'open' command in OS X to open it in Excel.

    Note that the temporary file is not deleted, which could lead to a leak in
    some cases.  Use with prudence.

    :param: A: 1 or 2d numpy array 
    """

    t = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    np.savetxt(t.file, A, delimiter=',')
    t.file.close()
    os.system('open ' + t.name)

def downsample2d(inputArray, kernelSize):
    """This function downsamples a 2d numpy array by convolving with a flat
    kernel and then sub-sampling the resulting array.

    A kernel size of 2 means convolution with a 2x2 array [[1, 1], [1, 1]] and
    a resulting downsampling of 2-fold.

    :param: inputArray: 2d numpy array
    :param: kernelSize: integer
    """
    average_kernel = np.ones((kernelSize,kernelSize))

    blurred_array = sig.convolve2d(inputArray, average_kernel, mode='same')
    downsampled_array = blurred_array[::kernelSize,::kernelSize]
    return downsampled_array

def downsample3d(inputArray, kernelSize):
    """This function downsamples a 3d numpy array (an image stack)
    by convolving each frame with a flat kernel and then sub-sampling the resulting array,
    re-building a smaller 3d numpy array.

    A kernel size of 2 means convolution with a 2x2 array [[1, 1], [1, 1]] and
    a resulting downsampling of 2-fold.

    The array will be downsampled in the first 2 dimensions, as shown below.

    import numpy as np
    >>> A = np.random.random((100,100,20))
    >>> B = downsample3d(A, 2)
    >>> A.shape
    (100, 100, 20)
    >>> B.shape
    (50, 50, 20)

    :param: inputArray: 2d numpy array
    :param: kernelSize: integer
    """
    first_smaller = downsample2d(inputArray[:,:,0], kernelSize)
    smaller = np.zeros((first_smaller.shape[0], first_smaller.shape[1], inputArray.shape[2]))
    smaller[:,:,0] = first_smaller

    for i in range(1, inputArray.shape[2]):
        smaller[:,:,i] = downsample2d(inputArray[:,:,i], kernelSize)
    return smaller

