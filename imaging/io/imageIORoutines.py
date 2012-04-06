# Core import / export routines and utilities for all imaging analysis modules
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import subprocess
import tifffile
try:
    from PIL import Image
except ImportError:
    import Image

__all__ = ['play3dNpArray', 'save3dNPArrayAsMovie', 'writeMultiImageStack', 'imread', 'imreadStack', 'imsave', 'imview', 'splitAndResaveChannels', 'readMultiImageTifStack', 'readImagesFromList']

def play3dNpArray(npArray, frameRate=6):
    """This method loops a 3d numpy array in a matplotlib window.

    A bit defunct due to imview, but here for reference.
    
    :param npArray: a 3d numpy array
    :param frameRate: optional, defaults to 6
    """
    plt.figure()
    axis=plt.imshow(npArray[:,:,0])
    for i in range(1, npArray.shape[2]):
        axis.set_data(npArray[:,:,i])
        plt.draw()
        time.sleep(1.0/frameRate)

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
    os.system('rm -f %s_*png' % fileName)

def writeMultiImageStack(fileName, npArray):
    """This method saves a 3d numpy array as a number of .tif files.

    Files are named 'filename_XXXX.tif' where XXXX is an incrementing integer.

    A bit defunct due to imview (one can save as an .avi from imagej), but here for reference.
    
    :param fileName: a filename for the images.  '_XXXX.tif' is appended.
    :param npArray: a 3d numpy array
    """

    for i in range(npArray.shape[2]):
        frame = npArray[:,:,i].T
        imsave('%s_%06d.tif' % (fileName, i), frame)

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
