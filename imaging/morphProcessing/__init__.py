"""
Some core morphological processing routines and the interactive image processing GUI.

========================================================================================
Core morphological processing routines (:mod:`imaging_analysis.core.morphProcessing`)
========================================================================================

"""
from morphProcessingRoutines import *
try:
    from ImageProcGUI import *
except ImportError:
    print 'Error importing ImageProcGUI. (Likely no gui mode or no appropriate matplotlib backend available).'
