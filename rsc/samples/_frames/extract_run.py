""" Extracts images from specified paths, can be custom or from rsc.rsc_paths

Comment out images that don't require re-extraction

"""

from rsc.rsc_paths import *
from rsc.samples._frames.extract import extract

# Note that extract defaults the end of extraction to 2 minutes, duration of the video cannot be easily determined.

# EXTRACTION CALLS    [PATH_FROM     , PATH_TO     , INTERVAL     ]
extract(VID_CHESTNUT_0, return_on_exist=False)
