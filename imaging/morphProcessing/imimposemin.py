import numpy as np
import pymorph
import mahotas

def imimposemin(image, mask, connectivity):
    fm = image.copy()
    fm[mask] = -922337203685477580
    fm[np.logical_not(mask)] = 922337203685477580

    if 'float' in image.dtype.name:
        range = image.max() - image.min()
        if range is 0:
            h = 0.1
        else:
            h = range * 0.001
    else:
        h = 1

    fp1 = image + h

    g = np.minimum(fp1, fm)

    # may need to complement and uncomplement
    j = infrec(fm, g)
    return j

def infrec(f, g, Bc=None):
    if Bc is None: Bc = pymorph.secross()
    n = f.size
    return fast_conditional_dilate(f, g, Bc, n);

def fast_conditional_dilate(f, g, Bc=None, n=1):
    if Bc is None: Bc = pymorph.secross()
    f = pymorph.intersec(f,g)
    for i in xrange(n):
        prev = f
        f = pymorph.intersec(mahotas.dilate(f, Bc), g)
        if pymorph.isequal(f, prev): break
    return f
