# segmentation and ROI extraction code
import numpy as np
import pymorph
import mahotas

import scipy.stats as stats
import scipy.signal as sig
import scipy.ndimage as nd

from CellPicker import pickCells

__all__ = ['pickCells', 'extractTimeCoursesFromSeries', 
           'extractTimeCoursesFromStack', 'avgFromROIInSeries', 
           'avgFromROIInStack', 'allPixelsFromROIInSeries', 'watershedSegment']

def extractTimeCoursesFromSeries(imageSeries, mask):
    """Get timecourses of stack regions defined by a index 2-D array

    Really just a wrapper around extractTimeCoursesFromStack
    adds a dimension, extracts, and squeezes.

    Returns a N by time numpy array where N is the number of objects
    in labelImage.

    :param series: X by Y by time
    :param mask: 2-D labeled image of cell masks
    :returns: numobjects by time numpy array
    """
    return np.squeeze(extractTimeCoursesFromStack(np.expand_dims(imageSeries, axis=3), mask))
    
def extractTimeCoursesFromStack(imageStack, mask):
    """Get timecourses of stack regions defined by a index 2-D array

    Returns a N by time by trial numpy array where N is the number of objects
    in labelImage.

    :param stack: X by Y by time by trial
    :param labelImage: 2-D labeled image of cell masks
    :returns: traces: a time by numobjects by trial numpy array
    """

    Xsize, Ysize, nTimePoints, nTrials = imageStack.shape

    objectLabels = set(mask.ravel())
    nObjects = max(objectLabels) + 1
    traces = np.zeros((nTimePoints, nObjects, nTrials))

    for obj in objectLabels:
        index = mask == obj
        traces[:, obj, :] = avgFromROIInStack(imageStack, index)

    return traces

def avgFromROIInSeries(imageSeries, binaryMask):
    """Computes an avgerage time series across all pixels in the mask.  

    :param imageSeries: X by Y by time
    :param binaryMask: binary mask image
    :returns: objectTimeCourse: numpy array, 1D, timeseries
    """
    nPixels = np.sum(binaryMask)
    objectTimeCourse = np.sum(imageSeries[binaryMask,:],axis=0)/nPixels # all the time points, all the trials, for every X and Y which == object
    return objectTimeCourse

def avgFromROIInStack(imageStack, binaryMask):
    """Computes an avgerage timeseries across all pixels in the mask, for all trials in the stack.  

    :param imageStack: X by Y by time by trial
    :param binaryMask: binary mask image
    :returns: objectTimeCourse: numpy array, 2D, timeseries by object
    """
    nPixels=np.sum(binaryMask)
    objectTimeCourse = np.sum(imageStack[binaryMask,:,:],axis=0)/nPixels # all the time points, all the trials, for every X and Y which == object
    return objectTimeCourse

def allPixelsFromROIInSeries(imageSeries, binaryMask):
    """Computes an pixel by pixel timeseries for all pixels in the mask, for all trials in the image series.
    Similar to avgFromROIInSeries, but doesn't average.
    
    :param imageSeries: X by Y by time
    :param binaryMask: binary mask image
    :returns: pixelValues: numpy array, 2D, pixel number by time
    """
    pixelValues = np.zeros((binaryMask.sum(), imageSeries.shape[2]))
    for i in range(imageSeries.shape[2]):
        iii = imageSeries[:,:,i]
        pixelValues[:,i] = iii[binaryMask.astype('bool')]
    return pixelValues

def watershedSegment(image, diskSize=20):
    """This routine implements the watershed example from 
    http://www.mathworks.com/help/images/examples/marker-controlled-watershed-segmentation.html, 
    but using pymorph and mahotas.

    :param image: an image (2d numpy array) to be segemented
    :param diskSize: an integer used as a size for a structuring element used 
                     for morphological preprocessing.
    :returns: tuple of binarized and labeled segmention masks
    """

    def gradientMagnitudue(image):
        sobel_x = nd.sobel(image.astype('double'), 0)
        sobel_y = nd.sobel(image.astype('double'), 1)
        return np.sqrt((sobel_x * sobel_x) + (sobel_y * sobel_y))    

    def imimposemin(image, mask, connectivity):
        fm = image.copy()
        fm[mask] = -9223372036854775800
        fm[np.logical_not(mask)] = 9223372036854775800

        fp1 = image + 1
        
        g = np.minimum(fp1, fm)
        
        j = infrec(fm, g)
        return j

    def infrec(f, g, Bc=None):
        if Bc is None: Bc = pymorph.secross()
        n = f.size
        return fast_conditional_dilate(f, g, Bc, n);

    def fast_conditional_dilate(f, g, Bc=None, n=1):
        if Bc is None:
            Bc = pymorph.secross()
        f = pymorph.intersec(f,g)
        for i in xrange(n):
            prev = f
            f = pymorph.intersec(mahotas.dilate(f, Bc), g)
            if pymorph.isequal(f, prev):
                break
        return f

    gradmag = gradientMagnitudue(image)

    ## compute foreground markers

    # open image to create flat regions at cell centers
    se_disk = pymorph.sedisk(diskSize) 
    image_opened = mahotas.open(image, se_disk);

    # define foreground markers as regional maxes of cells
    # this step is slow!
    foreground_markers = mahotas.regmax(image_opened)

    ## compute background markers

    # Threshold the image, cast it to the right datatype, and then calculate the distance image
    image_black_white = image_opened > mahotas.otsu(image_opened)
    image_black_white = image_black_white.astype('uint16')

    # note the inversion here- a key difference from the matlab algorithm
    # matlab distance is to nearest non-zero pixel
    # python distance is to nearest 0 pixel
    image_distance = pymorph.to_uint16(nd.distance_transform_edt(np.logical_not(image_black_white)))
    eight_conn = pymorph.sebox()

    distance_markers = mahotas.label(mahotas.regmin(image_distance, eight_conn))[0]
    image_dist_wshed, image_dist_wshed_lines = mahotas.cwatershed(image_distance, distance_markers, eight_conn, return_lines=True)
    background_markers = image_dist_wshed_lines - image_black_white

    all_markers = np.logical_or(foreground_markers, background_markers)

    # impose a min on the gradient image.  assumes int64
    gradmag2 = imimposemin(gradmag.astype(int), all_markers, eight_conn)

    # call watershed
    segmented_cells, segmented_cell_lines = mahotas.cwatershed(gradmag2, mahotas.label(all_markers)[0], eight_conn, return_lines=True)
    segmented_cells -= 1
    
    # seperate watershed regions
    segmented_cells[gradientMagnitudue(segmented_cells) > 0] = 0
    return segmented_cells > 0, segmented_cells
