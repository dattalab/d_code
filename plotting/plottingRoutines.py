import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from mpl_toolkits.axes_grid1 import ImageGrid


__all__ = ['plot_subset_of_array', 'plot_mean_and_sem', 'plot_array', 'imshow_array', 'plot_avg_and_comps']

def plot_subset_of_array(npArray, keep_slices, axis=1):
    plt.figure()
    
    mask = np.ones_like(npArray, dtype=bool)
    for s in keep_slices:
        # need to make a tuple of Ellipses and an int that is the current trace number
        slice_obj = []
        for a in range(npArray.ndim):
            if a is axis:
                slice_obj.append(s)
            else:
                slice_obj.append(Ellipsis)
        mask[tuple(slice_obj)] = False

    plot(ma.masked_array(npArray, mask))

def plot_mean_and_sem(npArray, axis=1):
    mean = npArray.mean(axis=axis)
    sem_plus = mean + stats.sem(npArray, axis=axis)
    sem_minus = mean - stats.sem(npArray, axis=axis)
    
    plt.figure()
    plt.fill_between(np.arange(mean.shape[0]), sem_plus, sem_minus, alpha=0.5)
    plt.plot(mean)

def plot_array(npArray, axis=1, xlim=None, ylim=None):

    plt.figure()

    num_plots = npArray.shape[axis]
    side = np.ceil(np.sqrt(num_plots))
    for current_plot in range(1, num_plots+1):

        plt.subplot(side, side, current_plot)
    
        # need to make a tuple of Ellipses and an int that is the current plot number
        slice_obj = []
        for a in range(npArray.ndim):
            if a is axis:
                slice_obj.append(current_plot-1)
            else:
                slice_obj.append(Ellipsis)

        plt.plot(npArray[tuple(slice_obj)])

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        else:
            plt.ylim([npArray.min()*0.9,npArray.max()*1.1])

def imshow_array(npArray, axis=2, vmax=None, vmin=None):

    plt.figure()

    num_plots = npArray.shape[axis]
    side = np.ceil(np.sqrt(num_plots))
    for current_plot in range(1, num_plots+1):

        subplot(side, side, current_plot)
    
        # need to make a tuple of Ellipses and an int that is the current plot number
        slice_obj = []
        for a in range(npArray.ndim):
            if a is axis:
                slice_obj.append(current_plot-1)
            else:
                slice_obj.append(Ellipsis)

        if vmax is None:
            vmax = npArray.max()*1.1
        if vmin is None
           vmine = npArray.min()*0.9

        imshow(npArray[tuple(slice_obj)], vmax=vmax, vmin=vmin)


def plot_avg_and_comps(npArray, axis=1):
    plt.figure()
    plt.plot(npArray, alpha=0.25, lw=1)
    plt.plot(npArray.mean(axis=axis), lw=2, color='black')
