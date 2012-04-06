import numpy as np
import ImageProcGUI as ipg
import pymorph
import scipy.ndimage as nd

__all__ = ['boxcar', 'highPassFilter', 'buildSeedPoints', 'regionProps']

def boxcar(imageSeries, boxWidth=3, axis=2):
    """Boxcar smoothes an image series with a boxcar of a specified width.
    Does this by convolving a flat array.

    :param imageSeries: a 3d numpy array (X by Y by time)
    :param boxWidth: an optional int, specifying the width of the boxcar
    :param axis: an optional int, specifying the axis to smooth over
    :returns: 3d numpy array, a smoothed version of imageSeries.
    """
    return nd.convolve1d(imageSeries, np.array([1]*boxWidth)/float(boxWidth), axis=axis)

def highPassFilter(imageSeries, averageRange=None, neg=False):
    """High pass filters an image series based on the subtraction of a gaussian filtered
    version of the image.  Returns a single image to use for segmentation.

    Optional parameters are for averaging only a portion of the image, or for inverting the image.

    :param imageSeries: a 3d numpy array (X by Y by time)
    :param averageRange: an optional list of ints, specifying the frames over which to pre- average
    :returns: 2d numpy array, a high pass filtered version of the average of imageSeries.
    """
    if not averageRange:
        image = np.floor(np.mean(imageSeries, axis=2)).astype('uint16')
    else:
        image = np.floor(np.mean(image[:,:,averageRange[0]:averageRange[1]])).astype('uint16')
    if neg:
        image = pymorph.neg(image)

    return ipg.subGaussian(image)

def buildSeedPoints(image, mode='com'):
    """ Successive set of filters to take a still image, isolate cell-like ROIs, and return a
    list of points representing center points to pass into core.segmentation.pickCells.

    Filters are, in order:  thresholding, morphological closing, and then a connected pixel cutoff filter.

    Often, the best thing to pass into this image as a high pass filtered version of your field of view.

    :param image: a 2d numpy array to build points from
    :param mode: an optional string, either: 'centriod' or 'com'
    :returns: tuple, (seedPoints, seedPointImage) 2d and 3d numpy array, containing coordinates and an image of points, respectively
    """
    #    seedMask = ipg.binaryErode(ipg.connectedPixelFilter(pymorph.label(ipg.threshold(ipg.subGaussian(image))>0)))
    #    binarySeedMask = ipg.connectedPixelFilter(pymorph.label(pymorph.close(ipg.threshold(ipg.subGaussian(image))>0)))
    binarySeedMask = ipg.connectedPixelFilter(pymorph.label(pymorph.close(ipg.threshold(image))))
    seedMask = pymorph.label(binarySeedMask)
    seedingRegionProps = regionProps(image, seedMask)

    if mode is 'centroid':
        seedPoints = [r['centroid'] for r in sorted(seedingRegionProps, key=lambda x: x['meanIntensity'],reverse=True)]
    elif mode is 'com':
        seedPoints = [r['com'] for r in sorted(seedingRegionProps, key=lambda x: x['meanIntensity'],reverse=True)]

    #    pdb.set_trace()

    seedPoints = np.floor(np.array(seedPoints))
    seedPointImage = np.zeros_like(seedMask)        
    for point in seedPoints:
        seedPointImage[point] = 255

    return seedPoints, seedPointImage

def regionProps(mask, image=None):
    """Calculates some basic properties of ROIs in a mask.

    Properties calculated:  centroids, boxes, areas.  If the original image is passed in,
    this also calcluate the mean value in each region.

    Keys of returned dictionary:  'centroid', 'boundingBox', 'area', and optionally, 'meanIntensity'

    :param mask: a 2d labeled image
    :param image: an optional 2p numpy array, original image, for calculation of mean intensity values
    :returns: a dictionary of lists, each containing property values
    """

    if image is not None:
        if len(image.shape) >= 3:
            meanImage = np.mean(image, axis=2)
        else:
            meanImage = image

        min = meanImage.min()
        if min < 0:
            meanImage -= meanImage.min()

        means = pymorph.grain(meanImage, labels=mask, measurement='mean', option='data')
    else:
        means = None
        
    numLabels = mask.max()
    centroids = pymorph.blob(mask, measurement='centroid', output='data')
    boxes = pymorph.blob(mask, measurement='boundingbox', output='data')
    area = pymorph.blob(mask, measurement='area', output='data')

    coms = []
    for i in range(1, numLabels+1):
        coms.append(nd.measurements.center_of_mass(meanImage, labels=mask==i))

    if means is not None:
        props = [{'meanIntensity':means[i], 'centroid':centroids[i], 'com':coms[i], 'boundingBox':boxes[i], 'area':area[i]} for i in range(numLabels)]
    else:
        props = [{'centroid':centroids[i], 'com':coms[i], 'boundingBox':boxes[i], 'area':area[i]} for i in range(numLabels)]
    return props

