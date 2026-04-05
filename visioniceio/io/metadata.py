"""Metadata readers -- plain-text, binary DLTG, and new PTH0 format.

All three metadata file variants are grouped here so that the full
metadata resolution chain (`.info` -> `.ifo` -> `-ifo.txt`) can be
understood from a single module.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

from ._helpers import (
    _parse_metadata_value,
    _read_dltg_header,
    _read_lv_string,
)

# ---------------------------------------------------------------------------
# Plain-text metadata (-ifo.txt)
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
        with open(str(filepath), encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, val = map(str.strip, line.split(':', 1))
                result[key] = _parse_metadata_value(val)
    except UnicodeDecodeError:
        # File is binary, not text -- return empty dict so callers can
        # fall through to a different parser.
        return {}
    return result


# ---------------------------------------------------------------------------
# Binary DLTG metadata (.ifo)
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
        if pth0_2 + 8 > len(data):
            return read_metadata(filepath)
        rec2_size = struct.unpack_from('>I', data, pth0_2 + 4)[0]
        pos = pth0_2 + 8 + rec2_size

        # --- Numeric metadata ---
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


# ---------------------------------------------------------------------------
# New-format PTH0 metadata (.info)
# ---------------------------------------------------------------------------

def read_info_new(filepath: str | Path) -> dict:
    """Read a new-format ``.info`` metadata file (PTH0 header).

    The ``.info`` file is a LabView variant-record binary.  It stores
    the same experiment metadata as ``.ifo`` / ``-ifo.txt`` but in a
    fixed-layout binary structure starting with two ``PTH0`` path
    records, followed by numeric metadata fields and channel-label
    arrays.

    Binary layout (all multi-byte values are big-endian)::

        [PTH0 record 1 -- variable length, may include extra fields]
        [PTH0 record 2 -- magic(4) + size(4) + size bytes of data]
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
    second_pth0 = data.find(b'PTH0', 4)
    if second_pth0 < 0:
        raise ValueError("Could not find second PTH0 record")

    if second_pth0 + 8 > len(data):
        raise EOFError(
            f"Second PTH0 record at offset {second_pth0} truncated: "
            f"need 8 bytes for header but only {len(data) - second_pth0} remain"
        )
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
    meta_fmt = '>IIIfIIfI'
    meta_size = struct.calcsize(meta_fmt)
    if pos + meta_size > len(data):
        raise EOFError(
            f"Numeric metadata at pos {pos}: need {meta_size} bytes "
            f"but only {len(data) - pos} remain"
        )
    meta_fields = struct.unpack_from(meta_fmt, data, pos)
    pos += meta_size
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

    if pos + 12 > len(data):
        raise EOFError(
            f"Channel counts at pos {pos}: need 12 bytes "
            f"but only {len(data) - pos} remain"
        )
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
