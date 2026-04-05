"""Shared low-level helpers for all I/O modules.

Contains byte-reading primitives, DLTG container parsing, metadata value
parsing, and the generic DLTG data reader.  Every type-specific module
(spike, waveform, etc.) imports from here -- this avoids cross-imports
between type modules.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Dtype lookup
# ---------------------------------------------------------------------------

dtype_map = {
    'uint32':  ('>u4', 4),
    'int32':   ('>i4', 4),
    'uint16':  ('>u2', 2),
    'int16':   ('>i2', 2),
    'float32': ('>f4', 4),
    'float64': ('>f8', 8),
}


# ---------------------------------------------------------------------------
# Extension mapping: new -> old
# ---------------------------------------------------------------------------

NEW_TO_OLD_EXT = {
    '.swave': '.swa',
    '.spike': '.spi',
    '.stim': '.stm',
    '.analog': '.ana',
    '.behave': '.bhv',
    '.info': '.ifo',
}


# ---------------------------------------------------------------------------
# Byte-level helpers
# ---------------------------------------------------------------------------

def _read_exact(f, n: int) -> bytes:
    """Read exactly *n* bytes from *f*, raising on short reads."""
    buf = f.read(n)
    if len(buf) < n:
        raise EOFError(
            f"Unexpected end of file: wanted {n} bytes, got {len(buf)}"
        )
    return buf


def _read_lv_string(data: bytes, pos: int) -> tuple[str, int]:
    """Read a single LabView string (uint32_BE length + chars).

    Returns:
        Tuple of (decoded_string, new_position).

    Raises:
        EOFError: If the buffer is too short for the length prefix
            or the declared string content.
    """
    if pos + 4 > len(data):
        raise EOFError(
            f"LabView string: need 4-byte length prefix at pos {pos}, "
            f"but buffer has only {len(data)} bytes"
        )
    slen = struct.unpack_from('>I', data, pos)[0]
    pos += 4
    if pos + slen > len(data):
        raise EOFError(
            f"LabView string: declared length {slen} at pos {pos - 4}, "
            f"but only {len(data) - pos} bytes remain"
        )
    s = data[pos:pos + slen].decode('ascii', errors='replace')
    return s, pos + slen


# ---------------------------------------------------------------------------
# DLTG container parsing
# ---------------------------------------------------------------------------

def _read_dltg_header(f):
    """Read a DLTG file header and return (ndim, offset_table, descriptor).

    Args:
        f: Open file handle positioned at byte 0.

    Returns:
        Tuple of (ndim, offsets, descriptor) where *offsets* is an
        ``np.ndarray`` of uint32 byte offsets to each dataset.

    Raises:
        ValueError: If the file does not start with ``DTLG``.
    """
    magic = _read_exact(f, 4)
    if magic != b'DTLG':
        raise ValueError(
            f"Expected DTLG header, got {magic!r}"
        )
    _version = _read_exact(f, 4)
    ndim = struct.unpack('>I', _read_exact(f, 4))[0]
    p = struct.unpack('>I', _read_exact(f, 4))[0]
    ld = struct.unpack('>h', _read_exact(f, 2))[0]
    descriptor = _read_exact(f, ld).decode('ascii') if ld > 0 else ''

    # Load offset table -- 128 entries per block; entry 127 chains to
    # the next table when ndim > 127.
    f.seek(p)
    if ndim <= 128:
        raw = _read_exact(f, 128 * 4)
        offs = np.frombuffer(raw, dtype='>u4')[:ndim].copy()
    else:
        offs = []
        remaining = ndim
        while remaining > 0:
            raw = _read_exact(f, 128 * 4)
            block = np.frombuffer(raw, dtype='>u4')
            if remaining <= 127:
                offs.extend(block[:remaining].tolist())
                remaining = 0
            else:
                # First 127 entries are data offsets; entry 127 is
                # the chain pointer to the next offset table.
                offs.extend(block[:127].tolist())
                remaining -= 127
                next_table = int(block[127])
                f.seek(next_table)
        offs = offs[:ndim]

    return ndim, np.array(offs, dtype=np.uint32), descriptor


# ---------------------------------------------------------------------------
# Generic DLTG data reader
# ---------------------------------------------------------------------------

def read_data(filename, dtype, nd):
    """Read a DLTG binary file into a list of NumPy arrays.

    Args:
        filename (str): Path to the file to be read.
        dtype (str): Data type key.  Supported: ``'int16'``, ``'int32'``,
            ``'uint16'``, ``'uint32'``, ``'float32'``, ``'float64'``.
        nd (int): Number of dimensions of one single trial data set.

    Returns:
        list[numpy.ndarray]: List of numpy arrays, each containing one
        dataset.

    Note:
        A dataset is here defined as the appropriate data for one trial.
        For an electrode trace it is a 1-D object (the one electrode).
        For waveforms it is a 2-D object (index of the spike, signal of spike).
    """
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported datatype {dtype}")
    np_dtype, datasize = dtype_map[dtype]

    with open(filename, 'rb') as f:
        ndim, offset, _descriptor = _read_dltg_header(f)

        # read each dataset
        fsize = f.seek(0, 2)  # get file size
        data = []
        for off in offset:
            f.seek(int(off))
            # read dimension sizes (C-order)
            dims = struct.unpack(
                '>' + 'i' * nd, _read_exact(f, 4 * nd)
            )
            count = int(np.prod(dims))
            nbytes = count * datasize
            if nbytes > fsize:
                raise ValueError(
                    f"Dataset at offset {off} claims {count} elements "
                    f"({nbytes} bytes) but file is only {fsize} bytes"
                )
            raw = _read_exact(f, nbytes)
            arr = np.frombuffer(raw, dtype=np_dtype, count=count).astype(
                dtype=dtype
            )
            arr = arr.reshape(dims)
            data.append(arr)

    return data


def _read_dltg_string_datasets(filepath: str | Path) -> list[str]:
    """Read a DLTG file whose datasets are LabView strings.

    Each dataset block is expected to contain a 4-byte big-endian int32
    giving the string length, followed by that many ASCII bytes.

    Args:
        filepath: Path to the DLTG file.

    Returns:
        List of decoded strings, one per dataset.
    """
    with open(filepath, 'rb') as f:
        ndim, offsets, _descriptor = _read_dltg_header(f)
        strings: list[str] = []
        for off in offsets:
            f.seek(int(off))
            # LabView stores strings as (int32 length, bytes)
            str_len = struct.unpack('>i', _read_exact(f, 4))[0]
            raw_bytes = _read_exact(f, str_len)
            try:
                strings.append(raw_bytes.decode('ascii').strip())
            except UnicodeDecodeError:
                strings.append(raw_bytes.decode('latin-1').strip())
    return strings


# ---------------------------------------------------------------------------
# Metadata value parsing
# ---------------------------------------------------------------------------

def _parse_metadata_value(val: str):
    """Parse a single metadata value string into a Python type.

    Applies the same heuristics as ``read_metadata``: booleans, int,
    float (with comma-as-decimal support), comma-separated lists, or
    fallback to string.

    Args:
        val: The raw value string.

    Returns:
        Parsed Python object (bool, int, float, list, or str).
    """
    lower = val.lower()
    if lower in ('yes', 'true'):
        return True
    if lower in ('no', 'false'):
        return False

    if ',' in val:
        items = [item.strip() for item in val.split(',')]
        # European decimal notation: exactly two parts, no whitespace
        # around the comma, and the second part is pure digits.
        # E.g. "250,00" -> 250.0, "-3,50" -> -3.5.
        # List separators always use ", " (with space), so "1, 2" is
        # a two-element list, not a decimal.
        if (
            len(items) == 2
            and items[1].isdigit()
            and ', ' not in val
        ):
            try:
                return float(items[0] + '.' + items[1])
            except ValueError:
                pass
        # Comma-separated list of values
        try:
            return [int(item) for item in items]
        except ValueError:
            try:
                return [float(item) for item in items]
            except ValueError:
                return items

    try:
        return int(val)
    except ValueError:
        try:
            return float(val.replace(',', '.'))
        except ValueError:
            return val
