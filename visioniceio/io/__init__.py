"""I/O subpackage -- readers and writers grouped by data type.

Modules
-------
_helpers     Shared byte-reading primitives, DLTG parser, ``read_data()``.
metadata     Plain-text, binary DLTG, and PTH0 metadata readers.
spike        New-format ``.spike`` reader.
waveform     New-format ``.swave`` reader.
stim         New-format ``.stim`` reader.
analog       New-format ``.analog`` reader.
behaviour    Old ``.bhv`` (DLTG) and new ``.behave`` readers.
sorting      ``.ssort`` read / write.
zarr_io      Zarr store reload.
"""

from ._helpers import read_data
from .analog import read_analog_new
from .behaviour import read_behave_new, read_bhv
from .metadata import read_info_new, read_metadata, read_metadata_ifo
from .sorting import read_ssort, write_ssort
from .spike import read_spike_new
from .stim import read_stim_new
from .waveform import read_swave_new
from .zarr_io import load_from_zarr

__all__ = [
    # Generic old-format reader
    "read_data",
    # Metadata
    "read_metadata",
    "read_metadata_ifo",
    "read_info_new",
    # Spike
    "read_spike_new",
    # Waveform
    "read_swave_new",
    # Stimulus
    "read_stim_new",
    # Analog / LFP
    "read_analog_new",
    # Behaviour
    "read_bhv",
    "read_behave_new",
    # Sorting
    "read_ssort",
    "write_ssort",
    # Zarr
    "load_from_zarr",
]
