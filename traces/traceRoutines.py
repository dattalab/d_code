import numpy as np

from scipy.interpolate import interp1d, splrep, splev
import scipy.ndimage as nd
import scipy.stats
import scipy

from scipy.signal import butter, lfilter

import matplotlib as mpl
from matplotlib import mlab
import bisect

__all__ = ['baseline', 'normalize', 'normalizeAndBaseline', 'findLevels', \
               'subBaseline', 'normAndCorrectBaseline', 'correctionMatrixLinearFit', \
               'correctionMatrixSmoothed', 'boxcar', 'smooth', \
               'calcCorrectedSTDs', 'calcTracesOverThreshhold',  'calcPosTraces', 'partitionTracesBySTD', \
               'findLevels1d', 'findLevelsNd', \
               'fir_filter', 'butter_bandpass_filter', 'psd', 'specgram',\
               'mask_deviations', 'baseline_splines']

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
        f = interp1d(xRange, yRange)
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



# -------------------- Filtering Routines ------------------------------------------
# from http://code.google.com/p/python-neural-analysis-scripts/source/browse/trunk/scripts/Filtering/Fir.py

def spectral_inversion(kernel):
    kernel = -kernel
    kernel[len(kernel)/2] += 1.0
    return kernel

def make_fir_filter(sampling_freq, critical_freq, kernel_window, taps, kind, **kwargs):
    nyquist_freq = sampling_freq/2
    critical_freq = np.array(critical_freq, dtype = np.float64)
    normalized_critical_freq = critical_freq/nyquist_freq

    if not taps % 2: #The order must be even for high and bandpass
        taps += 1

    if kind.lower() in ['low','low pass', 'low_pass']:
        kernel = scipy.signal.firwin(taps, normalized_critical_freq,
                               window=kernel_window, **kwargs)

    elif kind.lower() in ['high','high pass', 'high_pass']:
        lp_kernel = scipy.signal.firwin(taps, normalized_critical_freq,
                                  window = kernel_window, **kwargs)
        kernel = spectral_inversion(lp_kernel)
          
    elif kind.lower() in ['band','band pass', 'band_pass']:
        lp_kernel = scipy.signal.firwin(taps, normalized_critical_freq[0],
                                  window = kernel_window, **kwargs)
        hp_kernel = scipy.signal.firwin(taps, normalized_critical_freq[1],
                                  window = kernel_window, **kwargs)
        hp_kernel = spectral_inversion(hp_kernel)
        
        bp_kernel = spectral_inversion(lp_kernel + hp_kernel)
        kernel = bp_kernel
    
    return kernel

def fir_filter(sig, sampling_freq, critical_freq, kernel_window = 'hamming', taps = 101, kind = 'band', **kwargs):
    """
    Build a filter kernel of type <kind> and apply it to the signal
    Returns the filtered signal.

    Inputs:
        sig          : an n element sequence
        sampling_freq   : rate of data collection (Hz)
        critical_freq   : high and low cutoffs for filtering 
                        -- for bandpass this is a 2 element seq.
        kernel_window   : a string from the list - boxcar, triang, blackman,
                             hamming, bartlett, parzen, bohman, blackmanharris, nuttall, barthann
        taps            : the number of taps in the kernel (integer)
        kind            : the kind of filtering to be performed (high,low,band)
        **kwargs        : keywords passed onto scipy.firwin
    Returns:
        filtered signal : an n element seq
    """

    kernel = make_fir_filter(sampling_freq, critical_freq, kernel_window, taps, kind, **kwargs) 


    return np.roll(scipy.signal.lfilter(kernel, [1], sig), -taps/2+1)

# BUTTERWORTH bandpass

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y




# -------------------- SPECTROGRAM ROUTINES------------------------------------------
# modified from http://code.google.com/p/python-neural-analysis-scripts/source/browse/trunk/LFP/signal_utils.py

def find_NFFT(frequency_resolution, sampling_frequency, 
              force_power_of_two=False):
    #This function returns the NFFT
    NFFT = (sampling_frequency*1.0)/frequency_resolution-2
    if force_power_of_two:
        pow_of_two = 1
        pot_nfft = 2**pow_of_two
        while pot_nfft < NFFT:
            pow_of_two += 1
            pot_nfft = 2**pow_of_two
        return pot_nfft
    else:
        return NFFT
        
def find_frequency_resolution(NFFT, sampling_frequency):
    return (sampling_frequency*1.0)/(NFFT + 2)

def find_NFFT_and_noverlap(frequency_resolution, sampling_frequency,
                           time_resolution, num_data_samples):
    NFFT =  find_NFFT(frequency_resolution, sampling_frequency)
    
    # finds the power of two which is just greater than NFFT
    pow_of_two = 1
    pot_nfft = 2**pow_of_two
    noverlap = pot_nfft-sampling_frequency*time_resolution
    while pot_nfft < NFFT or noverlap < 0:
        pow_of_two += 1
        pot_nfft = 2**pow_of_two
        noverlap = pot_nfft-sampling_frequency*time_resolution

    pot_frequency_resolution = find_frequency_resolution(pot_nfft, 
                                                         sampling_frequency)
    
    return {'NFFT':int(NFFT), 'power_of_two_NFFT':int(pot_nfft), 
            'noverlap':int(noverlap), 
            'power_of_two_frequency_resolution':pot_frequency_resolution} 

def resample_signal(signal, prev_sample_rate, new_sample_rate):
    rate_factor = new_sample_rate/float(prev_sample_rate)
    return scipy.signal.resample(signal, int(len(signal)*rate_factor))    

def psd(signal, sampling_frequency, frequency_resolution,
        high_frequency_cutoff=None,  axes=None, **kwargs):
    """
    This function wraps matplotlib.mlab.psd to provide a more intuitive 
        interface.
    Inputs:
        signal                  : the input signal (a one dimensional array)
        sampling_frequency      : the sampling frequency of signal
        frequency_resolution    : the desired frequency resolution of the 
                                    specgram.  this is the guaranteed worst
                                    frequency resolution.
        --keyword arguments--
        **kwargs                : Arguments passed on to 
                                   matplotlib.mlab.specgram
    Returns:
        Pxx
        freqs
    """
    if (high_frequency_cutoff is not None 
        and high_frequency_cutoff < sampling_frequency):
        resampled_signal = resample_signal(signal, sampling_frequency, 
                                                    high_frequency_cutoff)
    else:
        high_frequency_cutoff = sampling_frequency
        resampled_signal = signal
    num_data_samples = len(resampled_signal)
    NFFT= find_NFFT(frequency_resolution, high_frequency_cutoff, 
                    force_power_of_two=True) 
    
    return mlab.psd(resampled_signal, NFFT=NFFT, 
                    Fs=high_frequency_cutoff, 
                    noverlap=0, **kwargs)

def specgram(signal, sampling_frequency, time_resolution, 
             frequency_resolution, bath_signals=[], 
             high_frequency_cutoff=None,  axes=None, logscale=True, **kwargs):
    """
    This function wraps matplotlib.mlab.specgram to provide a more intuitive 
        interface.
    Inputs:
        signal                  : the input signal (a one dimensional array)
        sampling_frequency      : the sampling frequency of signal
        time_resolution         : the desired time resolution of the specgram
                                    this is the guaranteed worst time resolution
        frequency_resolution    : the desired frequency resolution of the 
                                    specgram.  this is the guaranteed worst
                                    frequency resolution.
        --keyword arguments--
        **kwargs                : Arguments passed on to 
                                   matplotlib.mlab.specgram
    Returns:
            Pxx
            freqs
            bins

    Plot with: 
        points, freqs, bins = specgram(...)
        extent = (bins[0], bins[-1], freqs[0], freqs[-1])
        imshow(points, aspect='auto', origin='lower', extent=extent) # from pyplot

    """
    if (high_frequency_cutoff is not None 
        and high_frequency_cutoff < sampling_frequency):
        resampled_signal = resample_signal(signal, sampling_frequency, 
                                                    high_frequency_cutoff)
    else:
        high_frequency_cutoff = sampling_frequency
        resampled_signal = signal
    num_data_samples = len(resampled_signal)
    specgram_settings = find_NFFT_and_noverlap(frequency_resolution, 
                                               high_frequency_cutoff, 
                                               time_resolution, 
                                               num_data_samples)
    NFFT     = specgram_settings['power_of_two_NFFT']
    noverlap = specgram_settings['noverlap']
    Pxx, freqs, bins = mlab.specgram(resampled_signal, 
                                                NFFT=NFFT, 
                                                Fs=high_frequency_cutoff, 
                                                noverlap=noverlap, **kwargs)


    if logscale:
        Pxx = 10*np.log10(Pxx)

    return Pxx, freqs, bins

def mask_deviations(traces, std_cutoff=2.25, axis=0, iterations=40):
    """This routine takes a 1 or 2d array and masks large positive deviations from the mean.
    It works by calculating the mean and std of the trace in the given axis, then making a masked
    numpy array where every value more than std_cutoff*std above the mean is masked.  It iterates
    a set number of times, but could be altered to take

    could be altered to mask both negative and postive deviations, and to go till convergence
    with a tolerance, rather than a fixed number of iterations.

    :param traces: a 1 or 2d numpy array (traces by time)
    :param std_cutoff: optional floating point number, used for masking
    :param axis: optional integer, axis over which to calculate mean and std
    :param interations: times to repeat the masking process
    :returns: 2d masked numpy array, same size as traces
    """

    temp_traces = traces.copy()

    cutoffs = temp_traces.mean(axis=axis) + temp_traces.std(axis=axis)*std_cutoff
    masked_traces = np.ma.masked_array(traces, traces>=cutoffs)

    # could go until some sort of convergence in STD
    # but light empirical testing shows convergence after ~5 iterations
    # going with 20 for the default for overkill (still is fast)
    for i in range(iterations):   
        cutoffs = masked_traces.mean(axis=axis) + masked_traces.std(axis=axis)*std_cutoff
        masked_traces = np.ma.masked_array(masked_traces, traces>=cutoffs)

    return masked_traces

def baseline_splines(traces, n_control_points, std_cutoff=2.25):
    """This routine takes a 1 or 2d array and fits a spline to the baseline.
    To pick points for the spline fit, the baseline is first calcuated by 
    mask_deviations().  Then, the trace is split into n_control_points-2 
    segments, and the centers of those segments and the local mean are used
    for x and y values, respectively.  Additionally, the endpoints of the 
    trace and the mean values of the last 10 points are used, as well.

    This routine can fail if there are no valid points (i.e., if all points
    in a segment are masked).  In light testing, the most common case of this
    happening is on the ends, and this is checked for, but could be more adaptive.

    The return value of this function is a numpy array the same size and shape
    as traces, which contains spline approximated baselines.  Dividing the
    original traces by the baseline splines will normalize the traces.

    The spline is smoothed, but the number of control points will affect 
    how 'responsive' the spline is to deviations.  A good starting point
    is 5 control points or so.

    :param traces: a 1 or 2d numpy array (traces by time)
    :param n_control_points: integer for number of control points in spline.
    :returns: 2d numpy array, same size as traces
    """

    # assuming time by traces x trials
    if traces.ndim is 1:
        traces = np.atleast_2d(traces).T
    if traces.ndim is 2:
        traces = np.atleast_3d(traces)
    
    num_points = traces.shape[0]
    num_traces = traces.shape[1]
    num_trials = traces.shape[2]

    fit_baselines = np.zeros_like(traces)
    
    for trial in range(num_trials):
        masked_traces = mask_deviations(traces[:,:,trial], std_cutoff=std_cutoff)
    
        for trace in range(num_traces):
            num_segments = n_control_points - 2
            edge_size = int(np.ceil(masked_traces.shape[0] * 0.1))
            if num_segments>0:
                trace_in_parts = np.array_split(masked_traces[edge_size:-edge_size,trace], n_control_points-2)

                means = [x.mean() for x in trace_in_parts] # could also consider the median point

                segment_length = len(trace_in_parts[0])
                center_of_first = segment_length / 2 
                xs = [center_of_first+segment_length*i for i in range(num_segments)]
            else: # only using endpoints
                means = []
                xs = []

            # *** needs to be more general to drop any value in means that is false

            # add the average of the first ten and last ten points to the spline
            if masked_traces[0:edge_size,trace].mean(axis=0):
                means.insert(0, masked_traces[0:edge_size,trace].mean(axis=0))
            else:
                print 'first control point undefined (masked away everything?)'
                means.append(means[0])
            xs.insert(0,0)

            if masked_traces[-edge_size:,trace].mean(axis=0):
                means.append(masked_traces[-edge_size:,trace].mean(axis=0))
            else:
                print 'last control point undefined (masked away everything?)'
                means.append(means[-1])
            xs.append(num_points)

            # fit spline and generate a baseline
            if n_control_points<=3:
                k=1
            else:
                k=3
            tck = splrep(xs,means,k=k)#, w=weights)#,s=20)
            xnew = np.arange(0,num_points)

            fit_baselines[:,trace,trial] = splev(xnew,tck)

    return np.squeeze(fit_baselines)

