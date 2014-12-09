# ipython_load_modules.py
# written by Andrew Giessel, 2014
# place in the 'startup' directory of your ipython profile directory (typically ~/.ipython/profile_default/)

# System imports

import os
import sys
import pprint
import pdb

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats
import scipy.signal as sig
import scipy.ndimage as nd

import pymorph 
import mahotas
# import seaborn as sns # optional...

# custom imports
# loop across dirs in ~/python_modules and add all of them to the python module path
print '\n\n------------Parsing ipython_load_modules.py------------\n'

# CHANGE TO REFLECT THE FULL PATH OF WHERE YOU KEEP YOUR LIBRARIES
# add the top level directory of any of your repos.
# Code below will recurse through all directories, ignoring .git info
# and not diving into package directories.

PYTHON_MODULES_DIR_LIST = [os.path.join(os.path.expandvars('$HOME'),'d_code')]


for python_module_dir in PYTHON_MODULES_DIR_LIST:
    print '---> Adding %s to the python path...' % python_module_dir
    sys.path.append(python_module_dir)
    for root, dirs, files in os.walk(python_module_dir):
        for d in dirs:
            # clearly, ignore the .git directory
            if '.git' in root or '.git' in d:
                pass
            # if directory contains "__init__.py" it is in a package and we don't/shouldn't
            # load the directory directly into the path, instead, use the package structure
            # as it is
            elif not os.path.exists(os.path.join(root, d, '__init__.py')):
                fulldir=os.path.join(root,d)
                print '---> Adding %s to the python path...' % fulldir
                sys.path.append(fulldir)


# if you included the path to the d_code repo above, these will do the right thing

try:
    import imaging.io as io
    import imaging.segmentation as seg
    import imaging.morphProcessing as mp
    import imaging.alignment as alignment
    import traces as tm
    import plotting as plotting
    import events

    import acq.scanimage as scim
except:
    print 'Error in d_code repo, or not properly added to system path!'

# other custom stuff

npr = np.random.random
plt.ion()

# default colors for plotting based on a custom .matplotlibrc

rainbow = ['#348ABD', # blue
           '#7A68A6', # purple 
           '#A60628', # red
           '#467821', # green
           '#CF4457', # pink
           '#188487', # turquoise
           '#E24A33', # orange 
           '#EDB16F', # orange
           '#7C65C9', # purple
           '#7BA7D7', # blue
           '#88BDC1', # turquoise
           '#E56060', # red
           '#EB8B68', # pink
           '#B2CF8D', # green
           '#CCCF7D', # yellowish
           '#7587DD', # blue
           '#9AC9A6'] # green

# code to make a transparent cmap (transparent at the minimum) for overlays
# from matplotlib.cm import jet
# import copy
# jet_alpha = jet(256)
# jet_alpha._lut[0,3] = 0

