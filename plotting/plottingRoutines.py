import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

__all__ = ['plotSubsetOfTraceArray', 'plot_mean_and_sem']

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

