"""Behaviour I/O -- old-format ``.bhv`` (DLTG) and new-format ``.behave``.

Both behaviour file variants are grouped here: the old DLTG container
format and the new headerless format.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from ._helpers import (
    _parse_metadata_value,
    _read_dltg_header,
    _read_exact,
    _read_u4_count_prefixed_as_i4,
)

# ---------------------------------------------------------------------------
# Old format: .bhv (DLTG container)
# ---------------------------------------------------------------------------


def read_bhv(filepath: str | Path) -> dict:
    """Read a ``.bhv`` (behaviour) DLTG file.

    The ``.bhv`` file stores per-trial behavioural data (e.g. eye
    position, reward timing, button presses) in the DLTG container.
    Datasets may be either numeric arrays (int16/int32/float) or
    LabView strings.

    This function attempts a two-pass strategy:

    1. **String datasets** -- Decoded and collected under a ``"strings"``
       key as a list.
    2. **Numeric fallback** -- If string decoding fails for a dataset,
       the raw bytes are returned under ``"raw_blocks"`` as a list of
       ``bytes`` objects for manual inspection.

    Additionally, key-value lines (``key: value``) found inside string
    datasets are extracted into the top-level dict, just like metadata.

    Args:
        filepath: Path to the ``.bhv`` file.

    Returns:
        Dictionary with parsed key-value metadata, a ``"strings"``
        list, and/or a ``"raw_blocks"`` list.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If the file does not have a valid DLTG header.
    """
    filepath = str(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    result: dict = {}
    string_datasets: list[str] = []
    raw_blocks: list[bytes] = []

    with open(filepath, "rb") as f:
        ndim, offsets, descriptor = _read_dltg_header(f)
        result["_descriptor"] = descriptor
        result["_n_datasets"] = ndim

        for off in offsets:
            f.seek(int(off))
            # Try string interpretation first (int32 length prefix)
            str_len_raw = f.read(4)
            if len(str_len_raw) < 4:
                continue
            str_len = struct.unpack(">i", str_len_raw)[0]

            # Sanity: if str_len is unreasonable, treat as numeric block
            f.seek(int(off))
            if 0 < str_len < 100_000:
                _read_exact(f, 4)  # skip length prefix
                raw_bytes = _read_exact(f, str_len)
                try:
                    text = raw_bytes.decode("ascii").strip()
                    string_datasets.append(text)
                    # Extract key:value pairs if present
                    for line in text.splitlines():
                        line = line.strip()
                        if ":" in line:
                            key, val = map(str.strip, line.split(":", 1))
                            result[key] = _parse_metadata_value(val)
                    continue
                except (UnicodeDecodeError, ValueError):
                    pass

            # Fallback: read remaining bytes until next offset or EOF
            f.seek(int(off))
            sorted_offsets = np.sort(offsets)
            idx = np.searchsorted(sorted_offsets, off + 1)
            if idx < len(sorted_offsets):
                next_off = int(sorted_offsets[idx])
                raw_blocks.append(f.read(next_off - int(off)))
            else:
                raw_blocks.append(f.read(4096))

    if string_datasets:
        result["strings"] = string_datasets
    if raw_blocks:
        result["raw_blocks"] = raw_blocks

    return result


# ---------------------------------------------------------------------------
# New format: .behave (headerless)
# ---------------------------------------------------------------------------


def read_behave_new(filepath: str | Path) -> np.ndarray:
    """Read a new-format ``.behave`` behaviour file.

    Structure::

        [n_trials : uint32_BE] [n_trials x uint32_BE behaviour_codes]

    Args:
        filepath: Path to the ``.behave`` file.

    Returns:
        1-D int32 array of per-trial behaviour codes.
    """
    return _read_u4_count_prefixed_as_i4(filepath, label="Behave file")
