"""Stimulus label I/O -- new-format ``.stim`` reader.

Old-format ``.stm`` files are read via the generic
``read_data(filepath, 'int32', 1)`` function from ``io._helpers``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._helpers import _read_u4_count_prefixed_as_i4


def read_stim_new(filepath: str | Path) -> np.ndarray:
    """Read a new-format ``.stim`` stimulus-label file.

    Structure::

        [n_trials : uint32_BE] [n_trials x uint32_BE stimulus_ordinals]

    Args:
        filepath: Path to the ``.stim`` file.

    Returns:
        1-D int32 array of per-trial stimulus ordinals (1-based).
    """
    return _read_u4_count_prefixed_as_i4(filepath, label="Stim file")
