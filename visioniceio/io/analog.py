"""Analog / LFP I/O -- new-format ``.analog`` reader.

Old-format ``.ana`` files are read via the generic
``read_data(filepath, 'int16', 1)`` function from ``io._helpers``.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from ._helpers import _read_exact


def read_analog_new(filepath: str | Path) -> list[np.ndarray]:
    """Read a new-format ``.analog`` LFP file (headerless, big-endian int16).

    Structure per channel-trial record::

        [count : uint32_BE] [count x int16_BE samples]

    Args:
        filepath: Path to the ``.analog`` file.

    Returns:
        List of 1-D int16 arrays (one per channel-trial record).
    """
    data: list[np.ndarray] = []
    fsize = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        while f.tell() < fsize:
            raw_hdr = f.read(4)
            if len(raw_hdr) < 4:
                break
            count = struct.unpack('>I', raw_hdr)[0]
            nbytes = count * 2
            if nbytes > fsize - f.tell():
                raise EOFError(
                    f"Record claims {count} analog samples ({nbytes} bytes) "
                    f"but only {fsize - f.tell()} bytes remain"
                )
            raw = _read_exact(f, nbytes)
            data.append(np.frombuffer(raw, dtype='>i2').astype(np.int16))
    return data
