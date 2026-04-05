"""Spike timestamp I/O -- new-format ``.spike`` reader.

Old-format ``.spi`` files are read via the generic
``read_data(filepath, 'uint32', 1)`` function from ``io._helpers``.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from ._helpers import _read_exact


def read_spike_new(filepath: str | Path) -> list[np.ndarray]:
    """Read a new-format ``.spike`` file (headerless, big-endian uint32).

    Structure per channel-trial record (trial-major, channel-minor order)::

        [count : uint32_BE] [count x uint32_BE spike_sample_indices]

    Args:
        filepath: Path to the ``.spike`` file.

    Returns:
        List of 1-D uint32 arrays, each containing spike-time sample
        indices at the spike sampling rate.
    """
    data: list[np.ndarray] = []
    fsize = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        while f.tell() < fsize:
            raw_hdr = f.read(4)
            if len(raw_hdr) < 4:
                break
            count = struct.unpack('>I', raw_hdr)[0]
            nbytes = count * 4
            if nbytes > fsize - f.tell():
                raise EOFError(
                    f"Record claims {count} spike timestamps ({nbytes} bytes) "
                    f"but only {fsize - f.tell()} bytes remain"
                )
            raw = _read_exact(f, nbytes)
            data.append(np.frombuffer(raw, dtype='>u4').astype(np.uint32))
    return data
