"""
Raw sensor integration module.

Tools for parsing, inspecting, and manipulating Simrad echosounder .raw files.
"""

from . import raw_parser
from . import metadata
from . import ping_parser
from . import bottom_detection
from . import visualization
from . import gps_parser
from . import preprocessing
from . import bottom_labelling

try:
    from . import zarr_dataset
    from . import data_loader
except ImportError:
    pass

__all__ = ['raw_parser', 'metadata', 'ping_parser', 'bottom_detection',
           'bottom_labelling', 'visualization', 'gps_parser', 'preprocessing',
           'zarr_dataset', 'data_loader']
