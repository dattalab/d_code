import numpy as np
import pymorph
import scipy.ndimage as nd

__all__ = ['boxcar', 'regionProps']

def boxcar(imageSeries, boxWidth=3, axis=2):
    """Boxcar smoothes an image series with a boxcar of a specified width.
    Does this by convolving a flat array.

    :param imageSeries: a 3d numpy array (X by Y by time)
    :param boxWidth: an optional int, specifying the width of the boxcar
    :param axis: an optional int, specifying the axis to smooth over
    :returns: 3d numpy array, a smoothed version of imageSeries.
    """
    return nd.convolve1d(imageSeries, np.array([1]*boxWidth)/float(boxWidth), axis=axis)

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

