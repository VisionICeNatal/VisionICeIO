"""Backward-compatibility shim.

All I/O functions have moved to the ``visioniceio.io`` subpackage,
grouped by data type.  This module re-exports them so that existing
``from visioniceio.core_io import X`` statements continue to work.
"""

from .io._helpers import (  # noqa: F401
    NEW_TO_OLD_EXT,
    _parse_metadata_value,
    _read_dltg_header,
    _read_dltg_string_datasets,
    _read_exact,
    _read_lv_string,
    dtype_map,
    read_data,
)
from .io.analog import read_analog_new  # noqa: F401
from .io.behaviour import read_behave_new, read_bhv  # noqa: F401
from .io.metadata import (  # noqa: F401
    read_info_new,
    read_metadata,
    read_metadata_ifo,
)
from .io.sorting import read_ssort, write_ssort  # noqa: F401
from .io.spike import read_spike_new  # noqa: F401
from .io.stim import read_stim_new  # noqa: F401
from .io.waveform import read_swave_new  # noqa: F401
from .io.zarr_io import load_from_zarr  # noqa: F401
