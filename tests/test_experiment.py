"""Tests for visioniceio.Experiment — load_spike_indices and friends."""

from __future__ import annotations

import os
import struct
import tempfile

import numpy as np
import pytest

from visioniceio import Experiment


def _write_spike_file(dirpath: str, name: str, records: list[np.ndarray]):
    """Write a synthetic .spike file (new format)."""
    path = os.path.join(dirpath, name + ".spike")
    with open(path, "wb") as f:
        for arr in records:
            f.write(struct.pack(">I", len(arr)))
            f.write(arr.astype(">u4").tobytes())


def _write_info_file(
    dirpath: str,
    name: str,
    n_trials: int,
    n_spike_ch: int,
    n_waveform_ch: int,
    spike_fs: int,
    analog_fs: int,
    n_pts_wf: int,
    max_trial_len: int,
):
    """Write a minimal .info metadata file (new PTH0 format)."""
    # Build a minimal PTH0 file with two PTH0 records, project name,
    # and numeric metadata matching read_info_new expectations.
    buf = bytearray()

    # PTH0 record 1 (empty path)
    buf += b"PTH0"
    buf += struct.pack(">I", 0)  # size = 0

    # PTH0 record 2 (empty path)
    buf += b"PTH0"
    buf += struct.pack(">I", 0)  # size = 0

    # 4-byte zero padding after record 2
    buf += b"\x00" * 4

    # Project name (LV string: uint32_BE len + chars)
    proj = b"test"
    buf += struct.pack(">I", len(proj)) + proj

    # 32 bytes reserved
    buf += b"\x00" * 32

    # Numeric metadata: >IIIfIIfI
    buf += struct.pack(
        ">IIIfIIfI",
        n_trials,
        max_trial_len,
        analog_fs,
        1.0,  # gain1
        0,  # reserved
        spike_fs,
        1.0,  # gain2
        0,  # reserved
    )

    # n_points_waveform, n_spike_channels, n_waveform_channels
    buf += struct.pack(">III", n_pts_wf, n_spike_ch, n_waveform_ch)

    # Channel labels (LV strings)
    for ch in range(n_spike_ch):
        label = str(ch + 1).encode("ascii")
        buf += struct.pack(">I", len(label)) + label

    path = os.path.join(dirpath, name + ".info")
    with open(path, "wb") as f:
        f.write(bytes(buf))


class TestLoadSpikeIndices:
    def test_loads_new_format(self):
        n_trials = 2
        n_ch = 2
        # trial-major, channel-minor: t0ch0, t0ch1, t1ch0, t1ch1
        records = [
            np.array([100, 200, 300], dtype=np.uint32),
            np.array([400], dtype=np.uint32),
            np.array([500, 600], dtype=np.uint32),
            np.array([], dtype=np.uint32),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            name = "test_exp"
            _write_spike_file(tmpdir, name, records)
            _write_info_file(
                tmpdir, name,
                n_trials=n_trials,
                n_spike_ch=n_ch,
                n_waveform_ch=n_ch,
                spike_fs=32000,
                analog_fs=1000,
                n_pts_wf=38,
                max_trial_len=2500,
            )
            # Also need stub .swave, .stim, .analog files — but
            # load_spike_indices only needs metadata + spike file.
            # We set path/name directly instead of calling load_from_dir.
            exp = Experiment()
            exp.path = tmpdir
            exp.name = name
            exp.metadata = exp._load_metadata()
            exp.nelectrodes = exp.metadata["NofSpikeChannels"]
            exp.ntrials = exp.metadata["NofTrials"]

            result = exp.load_spike_indices()

        assert len(result) == 4
        np.testing.assert_array_equal(result[0], [100, 200, 300])
        np.testing.assert_array_equal(result[1], [400])
        np.testing.assert_array_equal(result[2], [500, 600])
        assert len(result[3]) == 0
        # Check dtype is int32
        assert result[0].dtype == np.int32

    def test_raises_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = Experiment()
            exp.path = tmpdir
            exp.name = "nonexistent"
            with pytest.raises(FileNotFoundError):
                exp.load_spike_indices()
