"""These routines are for doing typical extracellular (cell-attached) analyses.

The main data structures we use here are XSG dictionaries.  These contain keys
by default that have a collection of meta-data, and one key for each of the three
ephus programs (ephys, acquierer and stimulator).

We also use some routines from the spike_sort package.  This package produces two 
different types of dictionaries for keeping track of raw spike data and spike times.
The routines in this module make XSG files compatible with spike sort routines.


"""


#import spike_sort as ss
#import spike_analysis as sa
from spike_sort.core.extract import detect_spikes, extract_spikes

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import copy

__all__ = ['traceToSpikeData', 'plot_raster', 'make_STH', 'make_spike_density', 'detect_spikes', 'extract_spikes', 'addDataFieldToXSG']

def traceToSpikeData(trace, FS=10000):
    """Simple routine to take a 1d array and turn it into a 'spike data'
    dictionary, appropriate for use with other routines in spike_sort and 
    within this module.

    :param: - trace - a 1d numpy array
    :param: - FS - optional, sample rate, by default 10k
    :returns: - a minimal spike_data dictionary
    """
    return {'data':np.atleast_2d(trace), 'FS':FS, 'n_contacts':1}

def addDataFieldToXSG(xsg, channel='chan0'):
    """This routine adds the key 'data' to an xsg, which is required
    for spike_sort routines.  It is also important to note that the
    order of the axes is reversed for spike sort - it is trials by
    samples instead of a typical samples x trials.

    We check here if the xsg is merged or not and use that information
    accordingly in building the data field.

    Note that it makes sense to call this AFTER merging XSGs!!

    :param: - xsg - a single or merged XSG dictionary 
    :param: - channel - a string, indicating the ephys channel to use for the
              data, defaults to 'chan0' 
    :returns: - the xsg dictionary with an added 'data' field
    """

    if 'merged' in xsg.keys():
        xsg['data'] = xsg['ephys'][channel].T
    else:
        xsg['data'] = np.expand_dims(xsg['ephys'][channel], 0)

    return xsg

def detect_spikes():
    # extract spike times and add a field called 'spike_times'
    pass

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
    """  
    Parameters:
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
