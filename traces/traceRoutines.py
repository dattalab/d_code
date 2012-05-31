import numpy as np
from scipy.interpolate import interp1d
import scipy.ndimage as nd
import scipy.stats
import scipy

__all__ = ['baseline', 'normalize', 'normalizeAndBaseline', 'findLevels', \
       'subBaseline', 'normAndCorrectBaseline', 'correctionMatrixLinearFit', \
       'correctionMatrixSmoothed', 'boxcar', 'smooth', \
       'calcCorrectedSTDs', 'calcTracesOverThreshhold',  'calcPosTraces', 'partitionTracesBySTD',
       'findLevels1d', 'findLevelsNd']

def baseline(A, baseRange, baseAxis):
    """Baseline a numpy array using a given range over a specfied axis.

    :param A: numpy array of arbitrary dimension
    :param baseRange: list of 2 numbers, specifying the range over which to compute the average for baselining.
    :param baseAxis: the axis of the np to baseline over
    :returns: basedlined array
    """

    shape = A.shape
    # make a slice to take the mean in the right dimension
    baseSlice = [slice(baseRange[0], baseRange[1]) if a is baseAxis else slice(None) for a in range(len(shape))]
    base = np.mean(A[baseSlice], axis=baseAxis)

    # make a slice to pad the numbers to make the broadcasting work
    try:
        return A - base
    except:
        subSlice=[slice(None) if axis == base.shape[0] else None for axis in shape]
        return A - base[subSlice]
    finally:
        pass
    
def normalize(A, normRange, normAxis):
    """Normalize a numpy array using a given range over a specfied axis.

    :param A: numpy array of arbitrary dimension
    :param normRange: list of 2 numbers, specifying the range over which to compute the average for normalization.
    :param normAxis: the axis of the np to normalize over
    :returns: normalized array
    """

    shape = A.shape
    # make a slice to take the mean in the right dimension
    # slice(None) effectively means ':', or all the elements
    normSlice = [slice(normRange[0], normRange[1]) if a is normAxis else slice(None) for a in range(len(shape))]
    norm = np.mean(A[normSlice], axis=normAxis)

    # make a slice to pad the numbers to make the broadcasting work
    # again, slice(None) means ':' and None means an empty dimension (note difference!)

    try:
        return A/norm
    except:
        subSlice=[slice(None) if axis == norm.shape[0] else None for axis in shape]
        return A / norm[subSlice]
    finally:
        pass
    
def normalizeAndBaseline(A, baseRange, baseAxis):
    """Normalize, then baseline a numpy array over a given range and on a specfied axis.

    Calls normalize, and then baseline

    :param A: numpy array of arbitrary dimension
    :param baseRange: list of 2 numbers, specifying the range over which to compute the average for baselining and normalization
    :param baseAxis: the axis of the np to baseline and normalize over
    :returns: normalized, baselined array
    """
    return baseline(normalize(A, baseRange, baseAxis), baseRange, baseAxis)

def findLevels(A, level, mode='rising', boxWidth=0, rangeSubset=None):
    """Function to find level crossings in an 1d numpy array.  Based on the Igor
    function FindLevel. 

    Can find rising and/or falling crossings, control with the 'mode' paramter.

    Returns an numpy array of all crossings found and the number of crossings

    :param A: 1d numpy array
    :param level: floating point to search for in A
    :param mode: optional string: mode specfication. one of 'rising', 'falling' or 'both'
    :param boxWidth: optional int for local boxcar smoothing
    :param rangeSubset: optional list of ints to limit the search
    :returns: tuple, a numpy array of level crossings and the number of crossings
    """
    assert mode in ('rising', 'falling', 'both'), 'traceManip.findLevels: Unknown mode \'%s\'' % mode

    if boxWidth is not 0:
        A = np.convolve(A, np.array([1]*boxWidth)/float(boxWidth))

    crossings = np.diff(np.sign(A-level), axis=0)
    
    if mode is 'rising':
        rising_points = np.where(crossings > 0)
        return rising_points[0], len(rising_points[0])
    elif mode is 'falling':
        falling_points = np.where(crossings < 0)
        return falling_points[0], len(falling_points[0])
    else:
        all_crossing_points = np.where(np.abs(crossings) > 0)
        return all_crossing_points, len(all_crossing_points)


def findLevels1d(A, level, mode='rising', boxWidth=0):
    return findLevelsNd(A, level, mode='rising', axis=0, boxWidth=0)

def findLevelsNd(A, level, mode='rising', axis=0, boxWidth=0):
    """Function to find level crossings in an Nd numpy array. 

    Can find rising and/or falling crossings, control with the 'mode' paramter.

    Returns a binary array of level crossings, with true elements right AFTER a crossing.

    NOTE THAT THIS RETURNS DIFFERENT VALUES THAN findLevels().  if you want to get a list of
    locations where the crossings occurs, then use the following syntax:

    levels = findLevelsNd(array, level)
    level_crossings_locations = levels.nonzero()
    number_of_level_crossings = len(level_crossing_locations[0])

    Often, the crossings are noisy.  You can use np.diff() and findLevelsNd() again to help yourself out.


    :param A: 1d numpy array
    :param level: floating point to search for in A
    :param mode: optional string: mode specfication. one of 'rising', 'falling' or 'both'
    :param axis: optional integer, specifies dimension
    :param boxWidth: optional int for local boxcar smoothing
    :returns: binary array of level crossing locations
    """
    assert mode in ('rising', 'falling', 'both'), 'traceManip.findLevels: Unknown mode \'%s\'' % mode

    if boxWidth is not 0:
        A = nd.convolve1d(A, np.array([1]*boxWidth)/float(boxWidth), axis=axis)

    crossings = np.diff(np.sign(A-level), axis=axis)
    
    if mode is 'rising':
        return crossings>0
    elif mode is 'falling':
        return crossings<0
    else:
        return np.abs(crossings>0)
    
def subBaseline(A):
    X = correctionMatrixSmoothed(A)
    return (A - X)

def normAndCorrectBaseline(A):
    X = correctionMatrixSmoothed(A)
    return (A - X + np.mean(X, axis=0)) / np.mean(X, axis=0)

def correctionMatrixLinearFit(A):
    """This routine could be improved quite a bit.  presumes that the first
    and last lines of the trace near the baseline and builds a 5 point
    average of those and a line which linearlly interpolates between them.

    This is done for every trace in the array, building a correction for baseline
    drift over a given axix (axis 0)
    """
    
    correctionMatrix = np.zeros_like(A)
    xRange = np.array([0, A.shape[0]])
    for traceNum in range(A.shape[1]):
        t = A[:,traceNum]
        traceLength = t.shape[0]
        start_window = np.array([0, 3])
        stop_window = np.array([traceLength-5, traceLength-1])

        start = np.mean(t[slice(start_window[0], start_window[1])])
        stop = np.mean(t[slice(stop_window[0], stop_window[1])])
        
        endpointRatio = start/stop
        numStartShifts = 0
        numStopShifts = 0
        start_std = np.std(t[slice(start_window[0], start_window[1])])
        stop_std = np.std(t[slice(stop_window[0], stop_window[1])])

        def shiftWin(window, shift_amount, trace):
            return window + shift_amount, np.mean(t[slice(window[0], window[1])]), np.std(t[slice(window[0], window[1])])

        while endpointRatio>1.2 or endpointRatio <0.85:
            if endpointRatio<1:
                print 'endpoint ratio >1, shifting...'
                stop_window, stop, stop_std = shiftWin(stop_window, -1, t)
                endpointRatio = start/stop
                numStopShifts += 1
            if endpointRatio>1:
                print 'endpoint ratio < 0.85, shifting...'
                start_window, start, start_std = shiftWin(start_window, 1, t)
                endpointRatio = start/stop
                numStartShifts += 1
                
        if numStartShifts > 0 or numStopShifts > 0:
            print 'shifted start %d times, shifted stop %d times.  final endpoints:' % (numStartShifts, numStopShifts)
            print start, stop
            print 'std of ranges: %f, %f' % (start_std,stop_std)
            if stop_std > 10:
                stop_window, stop, stop_std = shiftWin(stop_window, -1, t)
                print 'Stop window STD still high, shift by two, and now: %f' % stop_std
            if start_std > 10:
                start_window, start, start_std = shiftWin(start_window, 1, t)
                print 'Start window STD still high, shift by two, and now: %f' % start_std
                
        yRange = np.array([start, stop])
        f = scipy.interpolate.interp1d(xRange, yRange)
        correctionMatrix[:,traceNum] = f(np.linspace(0,A.shape[0],A.shape[0]))
    return correctionMatrix

def correctionMatrixSmoothed(A):
    """
    """
    correctionMatrix = np.zeros_like(A)
    for traceNum in range(A.shape[1]):
        t = A[:,traceNum]
        sm = lowessPy(np.arange(float(len(t))), t, f=1.8, iters=10)
        correctionMatrix[:,traceNum] = sm
    return correctionMatrix

def lowessPy(x, y, f=2./3., iters=3): 
    """lowess(x, y, f=2./3., iter=3) -> yest 

    Lowess smoother: Robust locally weighted regression. 
    The lowess function fits a nonparametric regression curve to a scatterplot. 
    The arrays x and y contain an equal number of elements; each pair 
    (x[i], y[i]) defines a data point in the scatterplot. The function returns 
    the estimated (smooth) values of y. 

    The smoothing span is given by f. A larger value for f will result in a 
    smoother curve. The number of robustifying iterations is given by iter. The 
    function will run faster with a smaller number of iterations. 

    x and y should be numpy float arrays of equal length.  The return value is 
    also a numpy float array of that length. 

    This is a version in C, but seems to yield different values....
    https://github.com/brentp/bio-playground/tree/master/lowess

    e.g. 
    >>> import numpy 
    >>> x = np.array([4,  4,  7,  7,  8,  9, 10, 10, 10, 11, 11, 12, 12, 12, 
    ...                 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 
    ...                 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 
    ...                 20, 22, 23, 24, 24, 24, 24, 25], np.float) 
    >>> y = np.array([2, 10,  4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 
    ...                 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 
    ...                 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 
    ...                 64, 66, 54, 70, 92, 93, 120, 85], np.float) 
    >>> result = lowess(x, y) 
    >>> len(result) 
    50 
    >>> print "[%0.2f, ..., %0.2f]" % (result[0], result[-1]) 
    [4.85, ..., 84.98] 
    """ 
    n = len(x) 
    r = int(np.ceil(f*n)) 
    h = [np.sort(abs(x-x[i]))[r] for i in range(n)] 
    w = np.clip(abs(([x]-np.transpose([x]))/h),0.0,1.0) 
    w = 1-w*w*w 
    w = w*w*w 
    yest = np.zeros(n) 
    delta = np.ones(n) 
    for iteration in range(iters): 
        for i in xrange(n): 
            weights = delta * w[:,i] 
            weights_mul_x = weights * x 
            b1 = np.dot(weights,y) 
            b2 = np.dot(weights_mul_x,y) 
            A11 = sum(weights) 
            A12 = sum(weights_mul_x) 
            A21 = A12 
            A22 = np.dot(weights_mul_x,x) 
            determinant = A11*A22 - A12*A21 
            beta1 = (A22*b1-A12*b2) / determinant 
            beta2 = (A11*b2-A21*b1) / determinant 
            yest[i] = beta1 + beta2*x[i] 
        residuals = y-yest 
        s = np.median(abs(residuals)) 
        delta[:] = np.clip(residuals/(6*s),-1,1) 
        delta[:] = 1-delta*delta 
        delta[:] = delta*delta 
    return yest 


# -------------------- SMOOTHING ROUTINES------------------------------------------

def boxcar(A, boxWidth=3, axis=1):
    """Boxcar smoothes a matrix of 1d traces with a boxcar of a specified width.
    Does this by convolving the traces with another flat array.

    :param A: a 1d (time) or 2d numpy array (traces by time)
    :param boxWidth: an optional int, specifying the width of the boxcar
    :returns: 2d numpy array, a smoothed version of A.
    """
    if A.ndim is 1:
        axis=0
    
    return nd.convolve1d(A, np.array([1]*boxWidth)/float(boxWidth), axis=axis)

def smooth(A, window_len=11, window='hanning'):
    """smooth the data (1D numpy array) using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    ripped off from http://www.scipy.org/Cookbook/SignalSmooth
  
    :param A: the input signal 
    :param window_len: the dimension of the smoothing window; should be an odd integer
    :param window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman' flat window will produce a moving average smoothing.
    :returns: the smoothed signal
        
    See also: numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    """

    if A.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if A.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return A

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[A[window_len-1:0:-1],A,A[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



def calcCorrectedSTDs(A, expectedMean):
    A = np.atleast_2d(A)
    STDs = np.zeros((A.shape[1]))

    for i in range(A.shape[1]):
        trace = A[:,i].copy()

        negValues = np.array([value for value in trace if value < expectedMean])
        flippedNegValues = np.abs(negValues-expectedMean)+expectedMean
        STDs[i] = np.std(np.concatenate((negValues, flippedNegValues)))

    return STDs

def calcTracesOverThreshhold(A, thresh):
    A = np.atleast_2d(A)
    overList = np.zeros((A.shape[1]))

    # if threshList is just a number, then it's a global comparison
    if type(thresh) is not np.ndarray:
        threshList = np.ones(A.shape[1]) * thresh
    else:
        threshList = thresh.copy()
        
    overList = np.zeros(A.shape[0])
    for i in range(A.shape[1]):
        overList[i] = (A[:,i] >= threshList[i]).any()

    return overList.astype('bool')

# -------------------- PLAYING ROUTINES------------------------------------------

def calcPosTraces(A, stdList, threshRange, numThresh=100, baseLevel=1.0):
    
    threshList = np.arange(threshRange[0], threshRange[1], (threshRange[1]-threshRange[0])/float(numThresh)) 

    numPos = np.zeros(len(threshList))
    posMat = np.zeros((A.shape[1], numThresh))


    for i, threshLevel in enumerate(threshList):
        s = stdList*threshLevel
        overT = calcTracesOverThreshhold(A, s)

        numPos[i] = np.sum(overT)
        posMat[:,i] = overT.copy()
    
    return numPos, posMat.astype('bool')

def partitionTracesBySTD(posMatrix, lowValue, highValue):
    diff = np.diff(np.logical_not(posMatrix), axis=1)
    dropLevels = np.argmax(diff, 1)
    
    # a dropLevel of '0' means either it always succeed so let's set it to the max
    dropLevels[dropLevels == 0] = posMatrix.shape[1] 
    
    low  = [index for index, cutLevel in enumerate(dropLevels) if cutLevel < lowValue]
    mid  = [index for index, cutLevel in enumerate(dropLevels) if lowValue <= cutLevel < highValue]
    high = [index for index, cutLevel in enumerate(dropLevels) if cutLevel >= highValue]

    return low, mid, high


