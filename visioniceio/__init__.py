"""VisionICeIO -- I/O for the Vision Lab (Natal) LabView data."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("visioniceio")
except PackageNotFoundError:
    # Importing from a source checkout that hasn't been `pip install -e`'d.
    # Without this fallback, the bare `import visioniceio` would crash with
    # PackageNotFoundError before the user ever gets to call any function.
    __version__ = "0.0.0-dev"

from .experiment import Experiment
from .io import (
    load_from_zarr,
    read_analog_new,
    read_behave_new,
    read_bhv,
    read_data,
    read_info_new,
    read_metadata,
    read_metadata_ifo,
    read_spike_new,
    read_ssort,
    read_stim_new,
    read_swave_new,
    write_ssort,
)

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
