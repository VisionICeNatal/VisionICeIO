"""Stimulus label I/O -- new-format ``.stim`` reader.

Old-format ``.stm`` files are read via the generic
``read_data(filepath, 'int32', 1)`` function from ``io._helpers``.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from ._helpers import _read_exact


def read_stim_new(filepath: str | Path) -> np.ndarray:
    """Read a new-format ``.stim`` stimulus-label file.

    Structure::

        [n_trials : uint32_BE] [n_trials x uint32_BE stimulus_ordinals]

    Args:
        filepath: Path to the ``.stim`` file.

    Returns:
        1-D int32 array of per-trial stimulus ordinals (1-based).
    """
    fsize = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        n_trials = struct.unpack('>I', _read_exact(f, 4))[0]
        nbytes = n_trials * 4
        if nbytes > fsize - f.tell():
            raise EOFError(
                f"Stim file claims {n_trials} trials ({nbytes} bytes) "
                f"but only {fsize - 4} payload bytes available"
            )
        raw = _read_exact(f, nbytes)
    return np.frombuffer(raw, dtype='>u4').astype(np.int32)
