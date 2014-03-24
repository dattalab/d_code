"""These routines are for doing typical extracellular (cell-attached) analyses.

The main data structures we use here are XSG dictionaries.  These contain keys
by default that have a collection of meta-data, and one key for each of the three
ephus programs (ephys, acquierer and stimulator).

We have been inspired by the spike_sort package, but re-implemented routines to better
fit the XSG data structure.  In particular, we have 'detectSpikes' and 'extractSpikes', 
as well as routines to calculate spike rate histograms and densities, and plotting a 
spike raster.





"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from itertools import repeat

import copy

__all__ = ['plot_raster', 'make_STH', 'make_spike_density', 'detectSpikes', 'extract_spikes']

def detectSpikes(orig_xsg, thresh=None, edge='falling', channel='chan0', filter_trace=False):
    # extract spike times and add a field called 'spikeTimes'

    # needs to take a merged or un-merged XSG and add a field called 'spike_times'
    # if unmerged, spike_times is a single np.ndarray of spike times (in samples), otherwise
    # it is a list of such np.ndarrays.
    
    # returns a new xsg with the added key.

    assert(edge in ['falling', 'rising'], "Edge must be 'falling' or 'rising'!")
    xsg = copy.deepcopy(orig_xsg)

    # internal function to be used with a map
    def detect(params): 
        trace, thresh, filter_trace = params

        if filter_trace:
            #trace = filterthetrace(trace)
            pass

        # thresh is now a single value or an explicit wave the same size and shape as trace
        # let's just make it explicit
        if type(thresh) is not np.ndarray:
            thresh = np.ones_like(trace) * thresh
        
        if edge == 'rising':
            i, = np.where((trace[:-1] < thresh[:-1]) & (trace[1:] > thresh[1:]))
        if edge == 'falling':
            i, = np.where((trace[:-1] > thresh[:-1]) & (trace[1:] < thresh[1:]))
        return i
                
    if 'merged' in xsg.keys():
        # important type expectation here --- could be list of floats or a list of expicit ndarrays
        if type(thresh) is not list:  
            thresh = repeat(thresh)
        if type(filter_trace) is not list:
            filter_trace = repeat(filter_trace)

        xsg['spikeTimes'] = map(detect, zip(np.rollaxis(xsg['ephys'][channel], 1, 0), thresh, filter_trace))
    else:
        xsg['spikeTimes'] = detect((xsg['ephys'][channel], thresh, filter_trace)) # wrapping here to make it compatible with the zip for a single trial

    return xsg


def extract_spikes():
    pass

def plot_raster(spike_times, win=None, n_trials=None, ax=None, height=1.):
    """Creates raster plots of spike trains:
         spike_times - a list of spike time dictionaries, or a single dictionary
         ntrials - number of trials to plot (if None plot all)
    """
    if isinstance(spike_times, dict):
        spike_times = [spike_times]
    if n_trials is None:
        n_trials=len(spike_times)
    if win is None:
        #use the last spike time, rounded up to the next ms
        last_spike_times = []
        for s in spike_times:
            try:
                last_spike_times.append(s['data'][-1])
            except:
                pass
        win = [0, np.ceil(max(last_spike_times))]
    if ax is None:
        ax = plt.gca()

    for trial in range(n_trials):
        try:
            plt.vlines(spike_times[trial]['data'], trial, trial+height)
        except:
            pass

    plt.xlim(win)
    plt.ylim((0,n_trials))
    plt.xlabel('time (ms)')
    plt.ylabel('trials')

def make_STH(spike_times, bin_size=0.25, win=None, n_trials=None, rate=False, **kwargs):
    """Parameters:
        spike_times - a list of spike time dictionaries, or a single dictionary
        bin_size - float, in ms
        win - tuple of trial length in ms, eg: (0, 30000)
        ntrials - number of trials to plot (if None plot all)

    Returns:
        STH - bins x # of spikes
        bins - bin values, in ms, use for plotting
    """
    if isinstance(spike_times, dict):
        spike_times = [spike_times]
    if n_trials is None:
        n_trials=len(spike_times)
    
    bins = np.arange(win[0],win[1],bin_size)
    
    STH = np.empty((bins.shape[0], n_trials))
    for trial in range(n_trials):
        trial_sth, bins = np.histogram(spike_times[trial]['data'], bins)
        
        if rate:
            trial_sth = trial_sth * 1. / bin_size * 1000.
        
        STH[:-1,trial] = trial_sth
    return STH, bins

def make_spike_density(sth, sigma=1):
    edges = np.arange(-3*sigma, 3*sigma, 0.001)
    kernel = stats.norm.pdf(edges, 0, sigma)
    kernel *= 0.001
    
    center = np.ceil(edges.shape[0]/2)
    center = int(center)
    
    spike_density = np.empty_like(sth)
    for i, trial in enumerate(np.rollaxis(sth, 1)):
        
        conv_spikes = np.convolve(trial.copy(), kernel)
        spike_density[:,i] = conv_spikes[center:-center+1]
        
    return spike_density
