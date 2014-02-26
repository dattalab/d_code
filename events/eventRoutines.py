"""Event arrays are 2D label arrays (time x ROI) that are generated from an
array of fluorescent traces of the same size.  Event dection is based on the
method of Dombeck et al., 2007, and looks for events that cross a multiple
of the standard deviation of the baseline, and last until the traces goes 
back down to 0.75 times the standard deviation of the baseline.  By
calculating postive and negative going events at different thresholds, one
can estimate an adaptive threshold that gives ~5% false positive rate.  
Typical values range from 1.75 to 2.5 times the standard deviations.

These routines are used to create and analyze event arrays.  Note that the 
routines that find events can deal with single or multiple trials, but the
other functions here (getCounts(), getStartsAndStops(), etc.) use single
trial event arrays i.e. 2d arrays (time x cells)."""

import numpy as np
import traces as tm
from sklearn.mixture import GMM
import scipy.ndimage as nd
import mahotas

__all__ = ['findEventsAtThreshold', 'findEventsDombeck', 'getCounts', 'getStartsAndStops', 'getDurations', 'getAvgAmplitudes', 'getWeightedEvents', 'findEvents', 'findEventsGMM', 'findEventsBackground']

def findEventsAtThreshold(traces, stds, rising_threshold, falling_threshold=0.75, first_mode='rising', second_mode='falling', boxWidth=3, distance_cutoff=2):
    """Routine to find events based on the method in Dombeck et al., 2007.  
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
    """This routine uses findEventsd() at a range of thresholds to
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


def getStartsAndStops(event_array):
    """This routine takes an event_array and returns the starting and stopping times for all events in the array.

    :param: event_array - 2d or 3d numpy event array (time x cells, or time x cells x trials))
    :returns: masked numpy arrays, one for starting times and stopping times.
              size is cells x max event number or cells x trials x max event number.
              masked array is to account for the variable number of events in each cell
    """

    event_array = np.atleast_3d(event_array)
    max_num_events = getCounts(event_array).max()
    time, cells, trials = event_array.shape

    starts = np.zeros((cells, trials, max_num_events))
    stops = np.zeros((cells, trials, max_num_events))

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
    """This routine takes an event_array and optionally a time range and returns the number
    of events in each cell.

    :param: event_array - 2 or 3d numpy event array (time x cells or time x cells x trials)
    :param: time_range - optional list of 2 numbers limiting the time range to count events
    :returns: 1d or 2d numpy array of counts (cells or cells x trials)
    """
    if time_range is not None:
        event_array = event_array[time_range[0]:time_range[1],:] # note that this works for 2 or 3d arrays...
    counts = (event_array>0).sum(axis=0)
    return counts

def getDurations(event_array, time_range=None):
    """This routine takes an event_array (time x cells) and returns the duration
    of events in each cell.

    :param: event_array - 2 or 3d numpy event array (time x cells, or time x cells x trials)
    :param: time_range - optional list of 2 numbers limiting the time range to count events
    :returns: 2d masked numpy array of event durations. size is cells x largest number of events.
              masked entries are to account for variable number of events
    """
    event_array = np.atleast_3d(event_array)
    max_num_events = getCounts(event_array).max()
    time, cells, trials = event_array.shape

    durations = np.zeros((cells, trials, max_num_events))
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
    """This routine takes an event_array (time x cells) and corresponding trace array
    and returns the average amplitudes of events in each cell.

    :param: event_array - 2 or 3d numpy event array (time x cells, or time x cells x trials)
    :param: time_range - optional list of 2 numbers limiting the time range to count events
    :returns: 2d masked numpy array of event average amplitudes. size is cells x largest number of events.
              masked entries are account for variable number of events
    """
    event_array = np.atleast_3d(event_array)
    trace_array= np.atleast_3d(trace_array)

    max_num_events = getCounts(event_array).max()
    time, cells, trials = event_array.shape

    amps = np.zeros((cells, trials, max_num_events))
    amps[:] = np.nan

    for cell in range(cells):
        for trial in range(trials):
            event_ids = np.unique(event_array[:,cell,trial])[1:]
            for i, event_id in enumerate(event_ids):
                amps[cell, trial, i] = np.argwhere(event_array[:,cell,trial] == event_id).mean()
    amps = np.ma.array(amps, mask=np.isnan(amps))
    amps = np.squeeze(amps)

    return np.ma.masked_array(amps, np.isnan(amps))


def getWeightedEvents(event_array, trace_array):
    """This routine takes an event array and corresponding trace array and
    replaces the event labels with the average amplitude of the event.

    :param: event_array - 2 or 3d numpy event array (time x cells, or time x cells x trials)
    :param: trace_array - 2 or 3d numpy event array (time x cells, or time x cells x trials)
    :returns: 2d numpy array same shape and size of event_array, zero where there
              weren't events, and the average event amplitude for the event otherwise.
    """
    weighted_events = np.zeros_like(event_array, dtype=float)
    
    for i in np.unique(event_array)[1:]:
        weighted_events[event_array==i] = trace_array[event_array==i].mean()
    return weighted_events

def fitGaussianMixture1D(data, n, force_sort='means'):
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
    # expects single or multiple trials of dF/F
    traces = np.atleast_3d(traces) # time x cells x trials
    time, cells, trials = traces.shape
    gmmBaselines = np.zeros((time, trials)) # one baseline estimation for each trial

    for trial in range(trials):
        for frame in range(time):
            means, stds, weights, bic, aic, model = fitGaussianMixture1D(traces[frame,:,trial], 2)
            gmmBaselines[frame, trial] = means.min()

    return gmmBaselines

def findEvents(traces, stds, threshold=2.5, falling_threshold=None, baselines=None, boxWidth=3, minimum_length=2):
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

    # smooth traces and baselines
    if boxWidth is not 0:
        traces_smoothed = nd.convolve1d(traces, np.array([1]*boxWidth)/float(boxWidth), axis=0)
        baselines_smoothed = nd.convolve1d(full_baselines, np.array([1]*boxWidth)/float(boxWidth), axis=0)

    # detect events
    if falling_threshold is None:  # simply greater than the threshold
        for trial in range(trials):
            for cell in range(cells):
                events[:,cell,trial] = traces_smoothed[:,cell,trial] > baselines_smoothed[:,cell,trial] + (stds[cell, trial] * threshold)
        events = mahotas.label(events, np.array([1,1])[:,np.newaxis,np.newaxis])[0]

    # filter on size (length)
    for single_event in range(1, events.max()+1):
        if (events == single_event).sum() <= minimum_length:
            events[events == single_event] = 0
    events = mahotas.label(events>0, np.array([1,1])[:,np.newaxis,np.newaxis])[0]

    return np.squeeze(events) # if we just have one trial, then return a 2d array.

def findEventsGMM(traces, stds, threshold=2.5, falling_threshold=None, boxWidth=3, minimum_length=2):
    if traces.ndim == 2:
        traces = np.atleast_3d(traces) # time x cells x trials
        stds = np.atleast_2d(stds).T # cells x trials
    baselines = getGMMBaselines(traces) # time x trials (one population baseline trace for all cells)
    return findEvents(traces, stds, threshold, falling_threshold, baselines, boxWidth, minimum_length)

def findEventsBackground(traces, stds, threshold=2.5, falling_threshold=None, boxWidth=3, minimum_length=2):
    if traces.ndim == 2:
        traces = np.atleast_3d(traces) # time x cells x trials
        stds = np.atleast_2d(stds).T # cells x trials
    baselines = traces[:,0,:].copy() # time x trials (one population baseline trace for all cells)
    return findEvents(traces, stds, threshold, falling_threshold, baselines, boxWidth, minimum_length)
    
