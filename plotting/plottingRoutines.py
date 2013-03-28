import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from mpl_toolkits.axes_grid1 import ImageGrid


__all__ = ['plotSubsetOfTraceArray', 'plot_mean_and_sem', 'plot_array', 'imshow_array']

def plotSubsetOfTraceArray(npArray, subList):
    plt.figure()
    for i in subList:
        plt.plot(npArray[:,i])

def plot_mean_and_sem(array, axis=1):
    mean = array.mean(axis=axis)
    sem_plus = mean + stats.sem(array, axis=axis)
    sem_minus = mean - stats.sem(array, axis=axis)
    
    plt.figure()
    plt.fill_between(np.arange(mean.shape[0]), sem_plus, sem_minus, alpha=0.5)
    plt.plot(mean)

def plot_array(array, axis=1, xlim=None, ylim=None):

    plt.figure()

    num_plots = array.shape[axis]
    side = np.ceil(np.sqrt(num_plots))
    for current_plot in range(1, num_plots+1):

        subplot(side, side, current_plot)
    
        # need to make a tuple of Ellipses and an int that is the current plot number
        slice_obj = []
        for a in range(array.ndim):
            if a is axis:
                slice_obj.append(current_plot-1)
            else:
                slice_obj.append(Ellipsis)

        plot(array[tuple(slice_obj)])

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

def imshow_array(array, axis=2, vmax=None, vmin=None):

    plt.figure()

    num_plots = array.shape[axis]
    side = np.ceil(np.sqrt(num_plots))
    for current_plot in range(1, num_plots+1):

        subplot(side, side, current_plot)
    
        # need to make a tuple of Ellipses and an int that is the current plot number
        slice_obj = []
        for a in range(array.ndim):
            if a is axis:
                slice_obj.append(current_plot-1)
            else:
                slice_obj.append(Ellipsis)

        imshow(array[tuple(slice_obj)], vmax=vmax, vmin=vmin)

