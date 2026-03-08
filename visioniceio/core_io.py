"""Core IO functions for VisionICe data files.

These functions provide the basic reading capabilities for the binary files.
They are used by the Experiment class, in which also the loading structure is
defined.
"""

from __future__ import annotations

import os
import struct
import warnings
from pathlib import Path

import numpy as np


dtype_map = {
    'uint32':  ('>u4', 4),
    'int32':   ('>i4', 4),
    'uint16':  ('>u2', 2),
    'int16':   ('>i2', 2),
    'float32': ('>f4', 4),
    'float64': ('>f8', 8),
}


# ---------------------------------------------------------------------------
# Helpers
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
    """
    slen = struct.unpack_from('>I', data, pos)[0]
    pos += 4
    s = data[pos:pos + slen].decode('ascii', errors='replace')
    return s, pos + slen


# ---------------------------------------------------------------------------
# Plain-text metadata
# ---------------------------------------------------------------------------

def read_metadata(filepath: str | Path) -> dict:
    """Parse metadata from a plain-text ``key: value`` file.

    Reads a text file (typically ``*-ifo.txt``) and extracts metadata as
    key-value pairs.  For the binary ``.ifo`` DLTG variant, use
    ``read_metadata_ifo`` instead.

    Args:
        filepath: Path to the metadata text file.

    Returns:
        Dictionary of parsed metadata.  Returns an empty dict if the
        file cannot be decoded as text.
    """
    result: dict = {}
    try:
        with open(str(filepath), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, val = map(str.strip, line.split(':', 1))
                result[key] = _parse_metadata_value(val)
    except UnicodeDecodeError:
        # File is binary, not text — return empty dict so callers can
        # fall through to a different parser.
        return {}
    return result


# ---------------------------------------------------------------------------
# DLTG container helpers
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

    # Load offset table — 128 entries per block; entry 127 chains to
    # the next table when ndim > 127.
    f.seek(p)
    if ndim <= 128:
        raw = f.read(128 * 4)
        offs = np.frombuffer(raw, dtype='>u4')[:ndim].copy()
    else:
        offs = []
        remaining = ndim
        while remaining > 0:
            raw = f.read(128 * 4)
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
        data = []
        for off in offset:
            f.seek(int(off))
            # read dimension sizes (C-order)
            dims = struct.unpack('>' + 'i' * nd, f.read(4 * nd))
            count = int(np.prod(dims))
            raw = f.read(count * datasize)
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
            str_len = struct.unpack('>i', f.read(4))[0]
            raw_bytes = f.read(str_len)
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


# ---------------------------------------------------------------------------
# Binary metadata readers
# ---------------------------------------------------------------------------

def read_metadata_ifo(filepath: str | Path) -> dict:
    """Read metadata from a binary ``.ifo`` DLTG file.

    The ``.ifo`` file wraps its metadata in a DLTG container.  The
    single dataset is a LabView variant-record binary that contains
    LV strings (record name, project name, drive letters), two PTH0
    path records, and then fixed-layout numeric metadata.

    If the DLTG container cannot be parsed, falls back to
    ``read_metadata`` (plain-text parser).

    Args:
        filepath: Path to the ``.ifo`` file.

    Returns:
        Dictionary of parsed metadata key-value pairs.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    filepath = str(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # --- Try reading the DLTG container ---
    try:
        with open(filepath, 'rb') as f:
            _ndim, offsets, _desc = _read_dltg_header(f)
            f.seek(int(offsets[0]))
            data = f.read()
    except (ValueError, struct.error, UnicodeDecodeError):
        return read_metadata(filepath)

    # --- Parse the binary dataset ---
    # Layout: 4 LV strings, 8 zero bytes, 2 PTH0 records, numeric metadata
    try:
        pos = 0
        record_name, pos = _read_lv_string(data, pos)
        project_name, pos = _read_lv_string(data, pos)
        # Two drive-letter strings
        _drive1, pos = _read_lv_string(data, pos)
        _drive2, pos = _read_lv_string(data, pos)

        # 8 bytes zero-padding
        pos += 8

        # Skip two PTH0 path records (scan for second, then skip it)
        pth0_1 = data.find(b'PTH0', pos)
        if pth0_1 < 0:
            return {}
        pth0_2 = data.find(b'PTH0', pth0_1 + 4)
        if pth0_2 < 0:
            return {}
        rec2_size = struct.unpack_from('>I', data, pth0_2 + 4)[0]
        pos = pth0_2 + 8 + rec2_size

        # --- Numeric metadata ---
        # Layout: NofTrials(u32), MaxTrialLength(f32), ??(u32), ??(f32),
        #   ??(u32), ??(u32), SpikeSamplingFreq(u32), ??(u32),
        #   AnalogSamplingFreq(u32), ??(u32),
        #   NofSpikeChannels(u32), NofSpikewaveformChannels(u32),
        #   NofAnalogChannels(u32), NofEventChannels(u32),
        #   ??(u32), ??(u32), ??(u32), NofPointsSpikewaveform(u32)
        fmt = '>IfIfIIIIII'
        fields = struct.unpack_from(fmt, data, pos)
        pos += struct.calcsize(fmt)
        (
            n_trials, max_trial_length,
            _unk1, _unk2,
            _unk3, _unk4,
            spike_sampling_freq,
            _unk5,
            analog_sampling_freq,
            _unk6,
        ) = fields

        fmt2 = '>IIIIIIII'
        fields2 = struct.unpack_from(fmt2, data, pos)
        pos += struct.calcsize(fmt2)
        (
            n_spike_ch, n_waveform_ch, n_analog_ch, n_event_ch,
            _unk7, _unk8, _unk9,
            n_points_waveform,
        ) = fields2

        return {
            'RecordName': record_name,
            'ProjectName': project_name,
            'NofTrials': n_trials,
            'MaxTrialLength': int(max_trial_length),
            'SpikeSamplingFrequency': spike_sampling_freq,
            'SpikewaveformSamplingFrequency': spike_sampling_freq,
            'AnalogSamplingFrequency': analog_sampling_freq,
            'NofSpikeChannels': n_spike_ch,
            'NofSpikewaveformChannels': n_waveform_ch,
            'NofAnalogChannels': n_analog_ch,
            'NofEventChannels': n_event_ch,
            'NofPointsSpikewaveform': n_points_waveform,
        }
    except (struct.error, UnicodeDecodeError, IndexError):
        # If binary parsing fails, try plain-text fallback
        return read_metadata(filepath)


def read_info_new(filepath: str | Path) -> dict:
    """Read a new-format ``.info`` metadata file (PTH0 header).

    The ``.info`` file is a LabView variant-record binary.  It stores
    the same experiment metadata as ``.ifo`` / ``-ifo.txt`` but in a
    fixed-layout binary structure starting with two ``PTH0`` path
    records, followed by numeric metadata fields and channel-label
    arrays.

    Binary layout (all multi-byte values are big-endian)::

        [PTH0 record 1 — variable length, may include extra fields]
        [PTH0 record 2 — magic(4) + size(4) + size bytes of data]
        [4 bytes zero-padding]
        [LabView string: uint32_BE(len) + chars]     <- project name
        [32 bytes reserved]
        [uint32 NofTrials]
        [uint32 MaxTrialLength]
        [uint32 AnalogSamplingFrequency]
        [float32 _gain1]
        [uint32 _reserved]
        [uint32 SpikeSamplingFrequency]
        [float32 _gain2]
        [uint32 _reserved]
        [uint32 NofPointsSpikewaveform]
        [uint32 NofSpikeChannels]
        [uint32 NofSpikewaveformChannels]
        [NofSpikeChannels x LabView string: channel labels]
        [per-trial data ...]

    Args:
        filepath: Path to the ``.info`` file.

    Returns:
        Dictionary of parsed metadata, using the same key names as
        ``read_metadata`` / ``read_metadata_ifo``.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If the file does not begin with ``PTH0``.
    """
    filepath = str(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 4 or data[:4] != b'PTH0':
        raise ValueError(
            f"Expected PTH0 header, got {data[:4]!r}"
        )

    # --- Skip two PTH0 path records ---
    # PTH0 records may contain extra trailing fields (record name,
    # padding) beyond what the size field declares.  To robustly skip
    # record 1, we scan for the second PTH0 marker.  Record 2 uses
    # the size field (which IS reliable for the last header record)
    # plus 4 bytes of zero-padding.
    second_pth0 = data.find(b'PTH0', 4)
    if second_pth0 < 0:
        raise ValueError("Could not find second PTH0 record")

    record2_size = struct.unpack_from('>I', data, second_pth0 + 4)[0]
    # Record 2 layout: magic(4) + size_field(4) + data(record2_size)
    pos = second_pth0 + 8 + record2_size
    # Skip 4-byte zero-padding after record 2
    pos += 4

    # --- Project name (LabView string) ---
    proj_name, pos = _read_lv_string(data, pos)

    # --- Reserved / zero-padded region (32 bytes) ---
    pos += 32

    # --- Numeric metadata fields ---
    meta_fields = struct.unpack_from('>IIIfIIfI', data, pos)
    pos += struct.calcsize('>IIIfIIfI')
    (
        n_trials,
        max_trial_length,
        analog_sampling_freq,
        _gain1,
        _reserved1,
        spike_sampling_freq,
        _gain2,
        _reserved2,
    ) = meta_fields

    n_points_waveform = struct.unpack_from('>I', data, pos)[0]
    pos += 4
    n_spike_channels = struct.unpack_from('>I', data, pos)[0]
    pos += 4
    n_waveform_channels = struct.unpack_from('>I', data, pos)[0]
    pos += 4

    # --- Spike channel labels (LabView string array, no count prefix) ---
    spike_channels: list[str] = []
    for _ in range(n_spike_channels):
        ch_name, pos = _read_lv_string(data, pos)
        spike_channels.append(ch_name)

    # Convert channel labels to integer list if possible
    try:
        spike_channel_ints = [int(ch) for ch in spike_channels]
    except ValueError:
        spike_channel_ints = spike_channels

    # --- Build result dict with same keys as read_metadata ---
    result: dict = {
        'RecordName': '',
        'ProjectName': proj_name,
        'NofTrials': n_trials,
        'MaxTrialLength': max_trial_length,
        'SpikeSamplingFrequency': spike_sampling_freq,
        'SpikewaveformSamplingFrequency': spike_sampling_freq,
        'AnalogSamplingFrequency': analog_sampling_freq,
        'NofSpikeChannels': n_spike_channels,
        'NofSpikewaveformChannels': n_waveform_channels,
        'NofAnalogChannels': n_spike_channels,
        'NofEventChannels': 0,
        'NofPointsSpikewaveform': n_points_waveform,
        'SpikeChannels': spike_channel_ints,
        'SpikeWaveformChannels': spike_channel_ints,
        'AnalogChannels': spike_channel_ints,
    }

    return result


# ---------------------------------------------------------------------------
# BHV reader (DLTG-based behaviour file)
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

    with open(filepath, 'rb') as f:
        ndim, offsets, descriptor = _read_dltg_header(f)
        result['_descriptor'] = descriptor
        result['_n_datasets'] = ndim

        for off in offsets:
            f.seek(int(off))
            # Try string interpretation first (int32 length prefix)
            str_len_raw = f.read(4)
            if len(str_len_raw) < 4:
                continue
            str_len = struct.unpack('>i', str_len_raw)[0]

            # Sanity: if str_len is unreasonable, treat as numeric block
            f.seek(int(off))
            if 0 < str_len < 100_000:
                f.read(4)  # skip length prefix
                raw_bytes = f.read(str_len)
                try:
                    text = raw_bytes.decode('ascii').strip()
                    string_datasets.append(text)
                    # Extract key:value pairs if present
                    for line in text.splitlines():
                        line = line.strip()
                        if ':' in line:
                            key, val = map(str.strip, line.split(':', 1))
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
        result['strings'] = string_datasets
    if raw_blocks:
        result['raw_blocks'] = raw_blocks

    return result


# ---------------------------------------------------------------------------
# New binary format readers (headerless LabView files)
# ---------------------------------------------------------------------------

# Extension mapping: new extension -> old extension
NEW_TO_OLD_EXT = {
    '.swave': '.swa',
    '.spike': '.spi',
    '.stim': '.stm',
    '.analog': '.ana',
    '.behave': '.bhv',
    '.info': '.ifo',
}


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
            raw = _read_exact(f, count * 4)
            data.append(np.frombuffer(raw, dtype='>u4').astype(np.uint32))
    return data


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
    with open(filepath, 'rb') as f:
        while f.tell() < fsize:
            raw_hdr = f.read(8)
            if len(raw_hdr) < 8:
                break
            count, pts = struct.unpack('>II', raw_hdr)
            if not wf_pts:
                wf_pts = pts
            raw = _read_exact(f, count * pts * 2)
            data.append(
                np.frombuffer(raw, dtype='>i2')
                .reshape(count, pts)
                .astype(np.int16)
            )
    return data, wf_pts


def read_stim_new(filepath: str | Path) -> np.ndarray:
    """Read a new-format ``.stim`` stimulus-label file.

    Structure::

        [n_trials : uint32_BE] [n_trials x uint32_BE stimulus_ordinals]

    Args:
        filepath: Path to the ``.stim`` file.

    Returns:
        1-D int32 array of per-trial stimulus ordinals (1-based).
    """
    with open(filepath, 'rb') as f:
        n_trials = struct.unpack('>I', _read_exact(f, 4))[0]
        raw = _read_exact(f, n_trials * 4)
    return np.frombuffer(raw, dtype='>u4').astype(np.int32)


def read_behave_new(filepath: str | Path) -> np.ndarray:
    """Read a new-format ``.behave`` behaviour file.

    Structure::

        [n_trials : uint32_BE] [n_trials x uint32_BE behaviour_codes]

    Args:
        filepath: Path to the ``.behave`` file.

    Returns:
        1-D int32 array of per-trial behaviour codes.
    """
    with open(filepath, 'rb') as f:
        n_trials = struct.unpack('>I', _read_exact(f, 4))[0]
        raw = _read_exact(f, n_trials * 4)
    return np.frombuffer(raw, dtype='>u4').astype(np.int32)


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
            raw = _read_exact(f, count * 2)
            data.append(np.frombuffer(raw, dtype='>i2').astype(np.int16))
    return data


# ---------------------------------------------------------------------------
# Spike-sorting results (.ssort)
# ---------------------------------------------------------------------------

def read_ssort(filepath: str | Path) -> list[dict]:
    """Read a ``.ssort`` spike-sorting results file.

    Structure per channel-trial record::

        [n_entries : uint32_BE] [n_fields : uint32_BE]
        [n_entries x n_fields x float32_BE]

    The first entry is a header row:
    ``[channel_idx, n_spikes, trial_idx, stim_condition, 0, ...]``.
    Remaining entries are spike rows:
    ``[cluster_label, spike_time_idx, amplitude, slope, feature1, ...]``.

    Args:
        filepath: Path to the ``.ssort`` file.

    Returns:
        list[dict]: One dict per channel-trial record with keys
        ``channel_idx`` (int), ``n_spikes`` (int), ``trial_idx`` (int),
        ``stim_condition`` (int), ``labels`` (ndarray of int32),
        ``spike_indices`` (ndarray of float32), and ``features``
        (ndarray of float32).
    """
    records: list[dict] = []
    fsize = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        while f.tell() < fsize:
            raw_hdr = f.read(8)
            if len(raw_hdr) < 8:
                break
            n_entries, n_fields = struct.unpack('>II', raw_hdr)
            raw = _read_exact(f, n_entries * n_fields * 4)
            block = np.frombuffer(raw, dtype='>f4').reshape(
                n_entries, n_fields
            )

            header_row = block[0]
            channel_idx = int(header_row[0])
            n_spikes = int(header_row[1])
            trial_idx = int(header_row[2]) if n_fields > 2 else 0
            stim_condition = int(header_row[3]) if n_fields > 3 else 0

            if n_spikes > 0:
                spike_rows = block[1:n_spikes + 1]
                labels = spike_rows[:, 0].astype(np.int32)
                spike_indices = spike_rows[:, 1].copy()
                features = (
                    spike_rows[:, 2:].copy()
                    if n_fields > 2
                    else np.empty((n_spikes, 0), dtype=np.float32)
                )
            else:
                labels = np.empty(0, dtype=np.int32)
                spike_indices = np.empty(0, dtype=np.float32)
                features = np.empty(
                    (0, max(0, n_fields - 2)), dtype=np.float32
                )

            records.append({
                'channel_idx': channel_idx,
                'n_spikes': n_spikes,
                'trial_idx': trial_idx,
                'stim_condition': stim_condition,
                'labels': labels,
                'spike_indices': spike_indices,
                'features': features,
            })
    return records


def write_ssort(
    filepath: str | Path,
    labels_per_record: list[np.ndarray],
    spike_indices_per_record: list[np.ndarray],
    features_per_record: list[np.ndarray] | None = None,
    n_fields: int = 10,
    channel_indices: list | np.ndarray | None = None,
    trial_indices: list | np.ndarray | None = None,
    stim_conditions: list | np.ndarray | None = None,
) -> None:
    """Write a ``.ssort`` spike-sorting results file.

    Produces a binary file matching the LabView ``.ssort`` format.
    Records are ordered trial-major, channel-minor (trial 0 channels
    0..N-1, then trial 1 channels 0..N-1, etc.).

    Args:
        filepath: Output file path.
        labels_per_record: List of 1-D int arrays (one per channel-trial).
            Each array contains the cluster label for every spike.
        spike_indices_per_record: List of 1-D float/int arrays of
            spike-time sample indices at the spike sampling rate.
        features_per_record: Optional list of 2-D float arrays
            ``(n_spikes, n_feat)`` with per-spike waveform features.
            If ``None``, feature columns are zero-filled.
        n_fields: Number of float32 columns per row (default 10).
        channel_indices: Optional per-record channel indices for the
            header row.  Defaults to the flat record index.
        trial_indices: Optional per-record trial indices for the
            header row.  Defaults to 0.
        stim_conditions: Optional per-record stimulus-condition codes
            for the header row.  Defaults to 0.
    """
    n_records = len(labels_per_record)
    n_feat = n_fields - 2  # columns: 0=label, 1=time_idx, 2..=features

    with open(filepath, 'wb') as f:
        for rec_idx in range(n_records):
            lab = np.asarray(
                labels_per_record[rec_idx], dtype=np.float32
            )
            idx = np.asarray(
                spike_indices_per_record[rec_idx], dtype=np.float32
            )
            n_spikes = len(lab)

            # Header row
            header = np.zeros(n_fields, dtype=np.float32)
            header[0] = float(
                channel_indices[rec_idx]
                if channel_indices is not None
                else rec_idx
            )
            header[1] = float(n_spikes)
            if trial_indices is not None and n_fields > 2:
                header[2] = float(trial_indices[rec_idx])
            if stim_conditions is not None and n_fields > 3:
                header[3] = float(stim_conditions[rec_idx])

            n_entries = 1 + n_spikes
            f.write(struct.pack('>II', n_entries, n_fields))
            f.write(header.astype('>f4').tobytes())

            if n_spikes > 0:
                rows = np.zeros(
                    (n_spikes, n_fields), dtype=np.float32
                )
                rows[:, 0] = lab
                rows[:, 1] = idx
                if features_per_record is not None:
                    feat = features_per_record[rec_idx]
                    if (
                        feat is not None
                        and feat.ndim == 2
                        and feat.shape[1] > 0
                    ):
                        cols = min(feat.shape[1], n_feat)
                        rows[:, 2:2 + cols] = np.asarray(
                            feat[:, :cols], dtype=np.float32
                        )
                f.write(rows.astype('>f4').tobytes())


# ---------------------------------------------------------------------------
# Zarr store loader
# ---------------------------------------------------------------------------

def load_from_zarr(
    zarr_path: str | Path,
    electrode: int | None = None,
):
    """Load a previously-saved zarr store back into an xarray structure.

    This allows re-loading data that was saved by
    ``Experiment.load_from_dir(..., save_as='zarr')``.

    Args:
        zarr_path: Path to the ``.zarr`` store directory.
        electrode: If specified, select a single electrode from the
            dataset.  Otherwise the full ``xr.Dataset`` is returned.

    Returns:
        ``xr.Dataset`` (full) or electrode-sliced ``xr.Dataset`` when
        *electrode* is given.

    Raises:
        FileNotFoundError: If the zarr store does not exist.
        ImportError: If xarray or zarr are not installed.
    """
    import xarray as xr

    zarr_path = str(zarr_path)
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    ds = xr.open_zarr(zarr_path)

    if electrode is not None:
        return ds.sel(electrodes=electrode)

    return ds
