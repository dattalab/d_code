"""Event arrays are 2D label arrays (time x ROI) that are generated from an
array of fluorescent traces of the same size.

    Uses the following inequality to determine if an event occured at a specific time in a cell:
        dF/F of cell > (baseline of cell + std_threshold * std of cell * alpha)

    See the findEvents() docstring for more info.

These routines are used to create and analyze event arrays.  Note that
some of the event utility functions return masked numpy arrays.  This
is because generally, there are different number of events in each
cell during each trial.  Anywhere there wasn't an event is a 'np.nan'
value, and the mask will ensure that it isn't used to calcuate things
like mean(), min(), max() etc.
"""

import numpy as np
import traces as tm
from sklearn.mixture import GMM
import scipy.ndimage as nd
import mahotas

__all__ = ['findEvents', 'findEventsGMM', 'findEventsBackground',
           'getCounts', 'getStartsAndStops', 'getDurations', 'getAvgAmplitudes', 'getWeightedEvents',
           'fitGaussianMixture1D', 'getGMMBaselines']

#----------------------------------------EVENT FINDING FUNCTIONS AND WRAPPERS-----------------------------------

def findEvents(traces, stds, std_threshold=2.5, falling_std_threshold=None, baselines=None, boxWidth=3, minimum_length=2, alpha=None):
    """Core event finding routine with flexible syntax.

    Uses the following inequality to determine if an event occured at a specific time in a cell:
        dF/F of cell > (baseline of cell + std_threshold * std of cell * alpha)
    
    By default, the baseline is 0.0 (the dF/F traces have been baselined).  This baseline can be 
    explicitly specified using the `baselines` parameter.  If `baselines` is a 1d array, it is a 
    global correction value.  If `baselines` is exactly the same size as `traces`, the routine 
    assumes that the baselines have been explicitly specificed across all cells, trials and frames.  
    If `baselines` is of size (time x trials), then the routine assumes that the basline value has
    been determined for the whole population on a trial by trial basis.  This is done in the routines
    `findEventsBackground` and `findEventsGMM`.

    The `alpha` parameter is here for flexibility.  It allows for the scaling of the threshold of detection
    on a cell by cell, frame by frame basis indepedent of the noise of a cell or it's baseline value.  
    If specified it must be the exact same size as `traces`.  By default it is set to 1.0.

    The routine returns an event array exactly the same size as `traces`, where each event is labeled with
    a unique number (an integer).  The background is labeled with '0'.  This can be used in all the utility
    routines below.

    :param: traces - 2 or 3d numpy array of baselined and normalized traces (time x cells, or time x cells x trials)
    :param: stds - 1 or 2d numpy event array of per-cell std values (cells, or cells x trials)
    :param: std_threshold - multiple of per-cell STD to use for an event (float)
    :param: falling_std_threshold - optional multiple of per-cell STD to use as end of an event (float)
    :param: baselines - optional estimation of the baseline values of the cells 
    :param: boxWidth - filter size for smoothing traces and background values before detection
    :param: minimum_length - minimum length of an event
    :param: alpha - optional scaling parameter for adjusting thresholds

    :returns: numpy array same shape and size of traces, with each event given a unique integer label
    """

    if traces.ndim == 2:
        traces = np.atleast_3d(traces) # time x cells x trials
        stds = np.atleast_2d(stds).T # cells x trials
    time, cells, trials = traces.shape
    events = np.zeros_like(traces)

    # broadcasting of baselines.  ends up as time x cells x trials.  this is really annoying,
    # but relying on numpy to broadcast things was tricky and problembatic.  idea here is to
    # get baselines identical to traces

    if baselines is None: # no baseline correction, default
        full_baselines = np.zeros_like(traces)

    elif baselines.shape == (time): # one global correction
        full_baselines = np.zeros_like(traces)
        for trial in range(trials):
            for cell in range(cells):
                full_baselines[:,cell,trial] = baselines

    elif baselines.shape ==(time, cells): # full, but only one trial
        full_baselines = baselines[:,:,None] 
    
    elif baselines.shape == (time, trials): # modeled on a trial by trial basis
            full_baselines = np.zeros_like(traces)
            for trial in range(trials):
                for cell in range(cells):
                    full_baselines[:,cell,trial] = baselines[:,trial]

    # this is a check to prevent a dip in the global population from calling stuff responders 
    # basically, if the estimated baseline falls below zero, we fall back to the implicit background
    # value of 0.0
    full_baselines[full_baselines<0.0] = 0.0  

    # alpha is a scaling factor for event detection. if used it has to be the same size and shape as traces.
    # no broadcasting is done here.  it scales the threshold for detection so by default it is 1.0 everywhere.
    if alpha is None:
        alpha = np.ones_like(full_baselines)

    # smooth traces and baselines
    if boxWidth is not 0:
        traces_smoothed = nd.convolve1d(traces, np.array([1]*boxWidth)/float(boxWidth), axis=0)
        baselines_smoothed = nd.convolve1d(full_baselines, np.array([1]*boxWidth)/float(boxWidth), axis=0)

    # detect events
    for trial in range(trials):
        for cell in range(cells):
            events[:,cell,trial] = traces_smoothed[:,cell,trial] > baselines_smoothed[:,cell,trial] + (stds[cell, trial] * float(std_threshold) * alpha[:,cell,trial])

    # filter for minimum length
    events = mahotas.label(events, np.array([1,1])[:,np.newaxis,np.newaxis])[0]
    for single_event in range(1, events.max()+1):
        if (events == single_event).sum() <= minimum_length:
            events[events == single_event] = 0
    events = events>0

    # if a falling std is specified, extend events until they drop below that threshold
    if falling_std_threshold is not None:
        for trial in range(trials):
            for cell in range(cells):
                falling_thresh_events = traces_smoothed[:,cell,trial] > baselines_smoothed[:,cell,trial] + (stds[cell, trial] * float(falling_std_threshold) * alpha[:,cell,trial])

                for event_end in np.argwhere(np.diff(events[:,cell,trial].astype(int)) == -1):
                    j = event_end
                    while ((events[j,cell,trial]) or (falling_thresh_events[j])) and (j < cells):
                        events[j,cell,trial] = events[j-1,cell,trial]
                        j = j + 1

    # finally label the event array and return it. 
    events = mahotas.label(events>0, np.array([1,1])[:,np.newaxis,np.newaxis])[0]
    return np.squeeze(events) 

def findEventsGMM(traces, stds, std_threshold=2.5, falling_std_threshold=None, boxWidth=3, minimum_length=2):
    """Wrapper for findEvents with baseline estimation using a mixture of gaussians model.

    The major idea here is to use a mixture of two gaussians to model
    the baselines within each trial as a mixture of two gaussians -
    one for the 'baseline' and one for all the 'bright' responding
    pixels.  At each time point, the ROI brightnesses are fit with
    with this GMM.  The means of the two distributions are initialized
    to the background 'cell' and all points brighter than the mean of
    all ROIs.  After fitting, the smaller of the two means at every
    point is taken to be the 'background'.  This generally is very
    close to the average of the entire frame, but is generally smaller
    during full field events, because the larger gaussian 'sucks up'
    the spurious bright pixels.
    
    See getGMMBaselines() for more information.

    :param: traces - 2 or 3d numpy array of baselined and normalized traces (time x cells, or time x cells x trials)
    :param: stds - 1 or 2d numpy event array of per-cell std values (cells, or cells x trials)
    :param: std_threshold - multiple of per-cell STD to use for an event (float)
    :param: falling_std_threshold - optional multiple of per-cell STD to use as end of an event (float)
    :param: baselines - optional estimation of the baseline values of the cells 
    :param: boxWidth - filter size for smoothing traces and background values before detection
    :param: minimum_length - minimum length of an event

    :returns: numpy array same shape and size of traces, with each event given a unique integer label
    """

    if traces.ndim == 2:
        traces = np.atleast_3d(traces) # time x cells x trials
        stds = np.atleast_2d(stds).T # cells x trials
    baselines = getGMMBaselines(traces) # time x trials (one population baseline trace for all cells)
    return findEvents(traces, stds, std_threshold, falling_std_threshold, baselines, boxWidth, minimum_length)

def findEventsBackground(traces, stds, std_threshold=2.5, falling_std_threshold=None, boxWidth=3, minimum_length=2):
    """Wrapper for findEvents with baseline estimation using the background..

    Here, we estimate the population baseline for all the cells as the
    'background cell', or cell 0.  It is generally a fair estimation
    of the general response of the field of view, but is imperfect due
    to segmentation errors.

    :param: traces - 2 or 3d numpy array of baselined and normalized traces (time x cells, or time x cells x trials)
    :param: stds - 1 or 2d numpy event array of per-cell std values (cells, or cells x trials)
    :param: std_threshold - multiple of per-cell STD to use for an event (float)
    :param: falling_std_threshold - optional multiple of per-cell STD to use as end of an event (float)
    :param: baselines - optional estimation of the baseline values of the cells 
    :param: boxWidth - filter size for smoothing traces and background values before detection
    :param: minimum_length - minimum length of an event

    :returns: numpy array same shape and size of traces, with each event given a unique integer label
    """
    if traces.ndim == 2:
        traces = np.atleast_3d(traces) # time x cells x trials
        stds = np.atleast_2d(stds).T # cells x trials
    baselines = traces[:,0,:].copy() # time x trials (one population baseline trace for all cells)
    return findEvents(traces, stds, std_threshold, falling_std_threshold, baselines, boxWidth, minimum_length)

#----------------------------------------EVENT UTILITY FUNCTIONS-----------------------------------

def getStartsAndStops(event_array):
    """This routine takes an event_array and returns the starting and
    stopping times for all events in the array.

    :param: event_array - 2d or 3d numpy event array (time x cells, or time x cells x trials))
    :returns: masked numpy arrays, one for starting times and stopping times.
              size is cells x max event number or cells x trials x max event number.
              masked array is to account for the variable number of events in each cell
    """

    event_array = np.atleast_3d(event_array)
    max_num_events = getCounts(event_array).max()
    time, cells, trials = event_array.shape

    starts = np.zeros((cells, trials, int(max_num_events)))
    stops = np.zeros((cells, trials, int(max_num_events)))

    starts[:] = np.nan
    stops[:] = np.nan

    for cell in range(cells):
        for trial in range(trials):
            event_ids = np.unique(event_array[:,cell,trial])[1:]
            for i, event_id in enumerate(event_ids):
                starts[cell, trial, i] = np.argwhere(event_array[:,cell,trial] == event_id).flatten()[0]
                stops[cell, trial, i] = np.argwhere(event_array[:,cell,trial] == event_id).flatten()[-1]

    starts = np.ma.array(starts, mask=np.isnan(starts))
    starts = np.squeeze(starts)

    stops = np.ma.array(stops, mask=np.isnan(stops))
    stops = np.squeeze(stops)

    return starts, stops

def getCounts(event_array, time_range=None):
    """This routine takes an event_array and optionally a time range
    and returns the number of events in each cell.

    :param: event_array - 2 or 3d numpy event array (time x cells or time x cells x trials)
    :param: time_range - optional list of 2 numbers limiting the time range to count events
    :returns: 1d or 2d numpy array of counts (cells or cells x trials)
    """
    if time_range is not None:
        event_array = event_array[time_range[0]:time_range[1],:] # note that this works for 2 or 3d arrays...

    if event_array.ndim is 2:
        event_array = event_array[:,:,np.newaxis]
    time, cells, trials = event_array.shape
    
    counts = np.zeros((cells,trials))
    for trial in range(trials):
        for cell in range(cells):
            counts[cell, trial] = np.unique(event_array[:,cell,trial]).size - 1
    return np.squeeze(counts)

def getDurations(event_array, time_range=None):
    """This routine takes an event_array (time x cells) and returns
    the duration of events in each cell.

    :param: event_array - 2 or 3d numpy event array (time x cells, or time x cells x trials)
    :param: time_range - optional list of 2 numbers limiting the time range to count events
    :returns: 2d masked numpy array of event durations. size is cells x largest number of events.
              masked entries are to account for variable number of events
    """
    event_array = np.atleast_3d(event_array)
    max_num_events = getCounts(event_array).max()
    time, cells, trials = event_array.shape

    durations = np.zeros((cells, trials, int(max_num_events)))
    durations[:] = np.nan

    for cell in range(cells):
        for trial in range(trials):
            event_ids = np.unique(event_array[:,cell,trial])[1:]
            for i, event_id in enumerate(event_ids):
                durations[cell, trial, i] = np.argwhere(event_array[:,cell,trial] == event_id).size
    durations = np.ma.array(durations, mask=np.isnan(durations))
    durations = np.squeeze(durations)
    
    return durations

def getAvgAmplitudes(event_array, trace_array, time_range=None):
    """This routine takes an event_array (time x cells) and
    corresponding trace array and returns the average amplitudes of
    events in each cell.

    :param: event_array - 2 or 3d numpy event array (time x cells, or time x cells x trials)
    :param: time_range - optional list of 2 numbers limiting the time range to count events
    :returns: 2d masked numpy array of event average amplitudes. size is cells x largest number of events.
              masked entries are account for variable number of events
    """
    event_array = np.atleast_3d(event_array)
    trace_array= np.atleast_3d(trace_array)

    max_num_events = getCounts(event_array).max()
    time, cells, trials = event_array.shape

    amps = np.zeros((cells, trials, int(max_num_events)))
    amps[:] = np.nan

    for cell in range(cells):
        for trial in range(trials):
            event_ids = np.unique(event_array[:,cell,trial])[1:]
            for i, event_id in enumerate(event_ids):
                amps[cell, trial, i] = trace_array[event_array == event_id].mean()
    amps = np.ma.array(amps, mask=np.isnan(amps))
    amps = np.squeeze(amps)

    return np.ma.masked_array(amps, np.isnan(amps))

def getWeightedEvents(event_array, trace_array):
    """This routine takes an event array and corresponding trace array
    and replaces the event labels with the average amplitude of the
    event.

    :param: event_array - 2 or 3d numpy event array (time x cells, or time x cells x trials)
    :param: trace_array - 2 or 3d numpy event array (time x cells, or time x cells x trials)
    :returns: 2d numpy array same shape and size of event_array, zero where there
              weren't events, and the average event amplitude for the event otherwise.
    """
    weighted_events = np.zeros_like(event_array, dtype=float)
    
    for i in np.unique(event_array)[1:]:
        weighted_events[event_array==i] = trace_array[event_array==i].mean()
    return weighted_events

#----------------------------------------GMM UTILITY FUNCTIONS-----------------------------------

def fitGaussianMixture1D(data, n):
    """Routine for fitting a 1d array to a mixture of `n` gaussians.

    We initialize the GMM model with means equal to the first point
    (the 'background' cell) and all ROIs larger than the mean.

    After fitting, we return the means, stds, and weights of the GMM,
    along with the BIC, AIC, and the model itself.

    :param: data - 1d array of data to fit
    :param: n - number of gaussians to fit
    :returns: tuple of (means, stds, weights, BIC, AIC, GMM model object)
    """
    
    g = GMM(n_components=n, init_params='wc', n_init=5)
    
    g.means_ = np.zeros((2, 1))
    g.means_[0,0] = data[0] # first datapoint is the background value... should be near 0.0
    g.means_[1,0] = data[data > data[0]].mean()

    g.fit(data)

    return (np.squeeze(g.means_.flatten()), 
            np.squeeze(np.sqrt(g.covars_).flatten()), 
            np.squeeze(g.weights_).flatten(),
            g.bic(data), 
            g.aic(data), 
            g)

def getGMMBaselines(traces):
    """Wrapper for fitGaussianMixture1D() for findEventsGMM().
    
    :param: traces - 2 or 3d numpy array of dF/F (time x cells, or time x cells x trials)
    :returns: 1 or 2d numpy array of estimated baseline (time or time x trials).
    """
    traces = np.atleast_3d(traces) # time x cells x trials
    time, cells, trials = traces.shape
    gmmBaselines = np.zeros((time, trials)) # one baseline estimation for each trial

    for trial in range(trials):
        for frame in range(time):
            means, stds, weights, bic, aic, model = fitGaussianMixture1D(traces[frame,:,trial], 2)
            gmmBaselines[frame, trial] = means.min()

    return gmmBaselines

#----------------------------------------DEPRECATED EVENT FINDING FUNCTIONS-----------------------------------

def findEventsAtThreshold(traces, stds, rising_threshold, falling_threshold=0.75, first_mode='rising', second_mode='falling', boxWidth=3, distance_cutoff=2):
    """----------------DEPRECATED-----------------------------

    Routine to find events based on the method in Dombeck et al., 2007.  
    Relies on the multi-dimensional findLevels function in traceRoutines.

    Finds all two sets of points in `traces` that cross threshold multiples
    of `stds`.  The first_mode and second_mode parameters determine if the
    crossings are rising, or falling.  The trace is filtered with a flat
    kernel of width `boxWidth` and successive crossings are paired.  Any
    crossings less that `distance_cutoff` apart are discarded.

    This routine is called by findEventsDombeck().

    :param: traces - 2 or 3d numpy array of dF/F traces (time x cells, or time x cells x trial)
    :param: stds - 1 or 2d numpy array of values representing noise levels in the data (cells, or cells x trials)
    :param: rising_threshold - float used for first crossings
    :param: falling_threshold - float used for second crossings
    :param: boxWidth - filter size
    :param: distance_cutoff - eliminate crossings pairs closer than this- eliminates noise

    :returns: 2d or 3d array same size and dimension as traces, labeled with event number
    """

    # insure that we have at least one 'trial' dimension.
    if traces.ndim == 2:
        traces = np.atleast_3d(traces)
        stds = np.atleast_2d(stds)

    time, cells, trials = traces.shape
    # normally tm.findLevels works with a single number, but if the shapes are right then it will broadcast correctly with a larger array
    first_crossings = tm.findLevelsNd(traces, np.array(stds)*rising_threshold, mode=first_mode, axis=0, boxWidth=boxWidth)
    second_crossings = tm.findLevelsNd(traces, np.array(stds)*falling_threshold, mode=second_mode, axis=0, boxWidth=boxWidth)
    
    events = np.zeros_like(traces)
    i=1
    for cell in range(cells):
        for trial in range(trials):
            rising_event_locations = np.where(first_crossings[:,cell,trial])[0] # peel off the tuple
            falling_event_locations = np.where(second_crossings[:,cell,trial])[0] # peel off the tuple
        
            possible_pairs = []
            for r in rising_event_locations:
                if possible_pairs:
                    prev_rising = zip(*possible_pairs)[0]
                    prev_falling = zip(*possible_pairs)[1] 
                    
                    if r <= prev_falling[-1]:
                        continue
                
                try:
                    f = falling_event_locations[np.searchsorted(falling_event_locations, r)]
                    possible_pairs.append([r,f])
                except IndexError:
                    possible_pairs.append([r,time])
                    
            for pair in possible_pairs:
                if pair[1]-pair[0] > distance_cutoff:
                    events[pair[0]:pair[1], cell, trial] = i
                    i = i+1

    return np.squeeze(events)

def findEventsDombeck(traces, stds, false_positive_rate=0.05, lower_sigma=1, upper_sigma=5, boxWidth=3, distance_cutoff=2):
    """----------------DEPRECATED-----------------------------
    
    This routine uses findEventsAtThreshold() at a range of thresholds to
    detect both postive and going events, and calculates a false positive
    rate based on the percentage of total negative events 
    (see Dombeck et al. 2007).  It then calculates the threshold closest to
    the specificed false postive rate and returns that event array for 
    positive going events.

    The falling value is hardcoded at 0.75 * std of baseline, as per Dombeck et al. 2007.

    :param: traces - 2 or 3d numpy array of traces (time x cells or time x cells x trials)
    :param: stds - 1 or 2d numpy array of values representing noise levels in the data (cells, or cells x trials)
    :param: false_positive_rate - float value of desired false positive rate (0.05 = 5%)
    :param: lower_sigma - starting point for scan
    :param: upper_sigma - stopping point for scan
    :param: boxWidth - window size for pre-smoothing
    :param: distance_cutoff - minimum length of event

    :returns: events array for traces at desired false positive rate
    """
    all_events = []
    
    for sigma in np.arange(lower_sigma, upper_sigma, 0.125):
        pos_events = findEventsAtThreshold(traces, stds, sigma, 0.75, first_mode='rising',  second_mode='falling', boxWidth=boxWidth, distance_cutoff=distance_cutoff)
        neg_events = findEventsAtThreshold(traces, stds, -sigma, -0.75, first_mode='falling',  second_mode='rising', boxWidth=boxWidth, distance_cutoff=distance_cutoff)

        temp_false_positive_rate = neg_events.max() / (pos_events.max() + neg_events.max())

        all_events.append((sigma, pos_events.max(), neg_events.max(), temp_false_positive_rate, pos_events, neg_events))

    closest_to_false_pos = np.argmin(np.abs(np.array(zip(*all_events)[3])-false_positive_rate)) # get all false positive rates, find index closest to 0.05
    print 'Using sigma cutoff of: ' + str(all_events[closest_to_false_pos][0]) # get the right sigma

    return all_events[closest_to_false_pos][4] # pos events are 4th in tuple   
