"""
================================================================
Core Alignment routines (:mod:`imaging_analysis.core.alignment`)
================================================================

This package really just uses the image_registration module, which in turn is cloned
from https://github.com/keflavich/image_registration.  The key functions are
'register_images', 'register_series', and 'register_series_parallel'.

"""

from image_registration import *
