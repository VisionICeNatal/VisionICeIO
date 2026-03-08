"""VisionICeIO -- I/O for the Vision Lab (Natal) LabView data."""

from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("visioniceio")

from .core_io import (
    read_metadata,
    read_metadata_ifo,
    read_info_new,
    read_bhv,
    read_data,
    load_from_zarr,
    # New format readers
    read_spike_new,
    read_swave_new,
    read_stim_new,
    read_behave_new,
    read_analog_new,
    # Sorting results
    read_ssort,
    write_ssort,
)
from .experiment import Experiment

__all__ = [
    "read_metadata",
    "read_metadata_ifo",
    "read_info_new",
    "read_bhv",
    "read_data",
    "load_from_zarr",
    "read_spike_new",
    "read_swave_new",
    "read_stim_new",
    "read_behave_new",
    "read_analog_new",
    "read_ssort",
    "write_ssort",
    "Experiment",
]
