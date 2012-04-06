import matplotlib.pyplot as plt
import numpy as np

__all__ = ['plotSubsetOfTraceArray']

def plotSubsetOfTraceArray(npArray, subList, avg=None):
    plt.figure()
    for i in subList:
        if avg is not None:
            plt.plot(npArray[:,i]-avg)
        else:
            plt.plot(npArray[:,i])
    # display(plt.gcf())
    # plt.close(plt.gcf())

# the following are convience functions for embedding figures inline
# if you don't start ipython qtconsole mode with the --pylab=inline option

def imshow(matrix, **kwargs):
    figure()
    if kwargs is not None:
        plt.imshow(matrix, **kwargs)
    else:
        plt.imshow(matrix)
    display(gcf())
    close()
    
def plot(array, **kwargs):
    figure()
    if kwargs is not None:
        plt.plot(array, **kwargs)
    else:
        plt.plot(array)
    display(gcf())
    close()
    
def hist(array, **kwargs):
    figure()
    if kwargs is not None:
        plt.hist(array, **kwargs)
    else:
        plt.hist(array)
    display(gcf())
    close()
