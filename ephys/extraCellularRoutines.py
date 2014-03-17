import spike_sort as ss
import spike_analysis as sa
from spike_sort.core.extract import detect_spikes, extract_spikes

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import copy

__all__ = ['traceToSpikeData', 'plot_raster', 'make_STH', 'make_spike_density', 'detect_spikes', 'extract_spikes', 'addDataFieldToXSG', 'detectSpikes', 'mergeXSGs']

def traceToSpikeData(trace, FS=10000):
    return {'data':np.atleast_2d(trace), 'FS':FS, 'n_contacts':1}

def addDataFieldToXSG(xsg, channel='chan0'):
    xsg['data'] = np.atleast_2d(xsg['ephys'][channel])
    return xsg

def mergeXSGs(xsg1, xsg2):
    """this is only valid on repetitions of the same sort of acquisition-
    that is, xsgs that had the same combo of acquirer, ephys and stimulator settings

    it can be used with reduce
    """
    xsg1 = copy.deepcopy(xsg1)
    xsg2 = copy.deepcopy(xsg2)

    # calculate which keys will be merged via numpy concat
    # and which by appending lists
    non_numpy_keys = xsg1.keys()
    numpy_keys = []
    for prog in ['acquirer', 'ephys', 'stimulator']:
        if xsg1[prog] is not None:
            non_numpy_keys.remove(prog)
            numpy_keys.append(prog)

    # ensure that 'merged' key is in there, and only once
    try:
        non_numpy_keys.remove('merged')
        non_numpy_keys.append('merged')
    except ValueError:
        non_numpy_keys.append('merged')

    # we need to distinguish between 'merged' and 'unmerged' xsgs
    # xsgs have the 'merged' key set to True, otherwise the key/val doesn't exist

    # the major thing here is that unmerged XSGs need 
    # to have everything wrapped in lists for merging.

    # three possibilities:
    #    both unmerged
    #    one merged, one not
    #    both merged

    # since this is going to typically be used from a reduce call,
    # we will often have the first possibility, and the second.

    for x in [xsg1, xsg2]:
        if 'merged' not in x.keys():
            x['merged'] = True
            for key in non_numpy_keys:
                x[key] = [x[key]]

    merged_xsg = {}
    for key in non_numpy_keys:
        merged_xsg[key] = xsg1[key] + xsg2[key]
    for key in numpy_keys:
        merged_xsg[key] = xsg1[key]

    return merged_xsg

def detectSpikes():
    # extract spike times and add a field called 'spike_times'
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
