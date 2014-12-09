# d_code

Welcome to the d_code repository.  This repo contains code to analyze
imaging and electrophysiology data, especially that collected from ScanImage 
and Ehpus.

For questions regarding this codebase, please open an issue rather than sending any email!

## Requirements and Installation

This package requires many core elements of the Python scientific stack,
including Numpy and Scipy.  The simplest way to get these is to install a
Python distribution such as [Anaconda](http://continuum.io/downloads)
(recommended) or [Enthought](https://store.enthought.com/downloads/). All of
this code works on a Python 2.7.x codebase.  Porting to Python 3 wouldn't be a
bad idea, but right now it runs on 2.7.

More explicitly, this code depends on the following non-standard library
modules: (all in Anaconda/Enthought):

- Numpy
- Scipy (interpolate, stats, ndimage, signal)
- Matplotlib
- Scikit-learn
- IPython
- Pyside
- Sphinx (for documentation)

In addition to what comes with either of these distributions, there are a couple
of external modules required for this package:

- pymongo 
- mahotas
- pymorph
- image_registration
- tifffile 

These can all be installed with `pip install <PACKAGE NAME>`.  If pip is not
installed, you can first install it with `easy_install pip`, and then proceed to
install these packages.

The tifffile package was written by Christoph Gohlke
<http://www.lfd.uci.edu/~gohlke/> for fast reading and writing of tiff files.
It can be compiled in place with the following command:   `python
tiffiflesetup.py build_ext --inplace`.  image_registration was forked from
https://github.com/keflavich/image_registration, and I wrote a couple of
convenience routines there.

Presuming that you are a member of the dattalab organization with access to this repo
(which you are, if you can read this), you should be able to install this code 
by standard git cloning techniques, that is, run the following at a terminal prompt:

    git clone git@github.com:dattalab/d_code.git

This will create a clone of the repo in the current directory.  After this, you should 
be able to add the top level directory to your python path and then import anything 
as you see fit, e.g.: `import imaging.io as io`.  See also the IPython startup 
file below for a way to do this automatically.

Alternatively, can feel free to fork the repo, and work from your own copy.

## Overview and Summary of Packages

There are 5 main packages:

- acq: Data import utilities for Ephus and Scanimage
- ephys: Routines for analyzing electrophsyiology (currently extracellular only)
  data
- events: Routines for detecting and analyzing 'event' arrays- 2d label arrays 
  which signify discrete events.
- imaging: Routines for I/O, alignment, morphological processing and segmentation of images.
- plotting: Some convenience routines for typical sorts of plotting and 
  visualization.
- traces: Routines for manipulating time series data (baselining, normalizing, 
  filtering, level finding and more)

## IPython startup file

`ipython_load_modules.py` is a file you can place in ~/.ipython/<PROFILE_NAME>/startup
that will run whenever a new ipython kernel is generated.  It is a pure python
script that includes a lot of standard system imports (sys, os, numpy), then
recursively adds a directory to the Python path, then imports a number of the
packages from this repo.

The way to use this file is to edit it to change the PYTHON_MODULES_DIR_LIST 
variable to reflect the list of folders you'd like to automatically add to your
Python path.  These need to be full system paths.  The program is recursive, but
doesn't include .git folders or dive into packages.  Thus, if you keep all of 
your code in one directory, adding that directory should be sufficient.
