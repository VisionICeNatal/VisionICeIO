"""Waveform snippet I/O -- new-format ``.swave`` reader.

Old-format ``.swa`` files are read via the generic
``read_data(filepath, 'int16', 2)`` function from ``io._helpers``.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from ._helpers import _read_exact


def read_swave_new(filepath: str | Path) -> tuple[list[np.ndarray], int]:
    """Read a new-format ``.swave`` waveform file (headerless).

    Structure per channel-trial record::

        [count : uint32_BE] [wf_pts : uint32_BE]
        [count x wf_pts x int16_BE waveform samples]

    Args:
        filepath: Path to the ``.swave`` file.

    Returns:
        Tuple *(data, wf_pts)* where *data* is a list of 2-D int16 arrays
        ``(n_spikes, wf_pts)`` and *wf_pts* is the number of waveform
        sample points per spike snippet.
    """
    data: list[np.ndarray] = []
    wf_pts = 0
    fsize = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        while f.tell() < fsize:
            raw_hdr = f.read(8)
            if len(raw_hdr) < 8:
                break
            count, pts = struct.unpack(">II", raw_hdr)
            if not wf_pts:
                wf_pts = pts
            nbytes = count * pts * 2
            if nbytes > fsize - f.tell():
                raise EOFError(
                    f"Record claims {count}x{pts} waveform samples "
                    f"({nbytes} bytes) but only {fsize - f.tell()} bytes remain"
                )
            raw = _read_exact(f, nbytes)
            data.append(np.frombuffer(raw, dtype=">i2").reshape(count, pts).astype(np.int16))
    return data, wf_pts
