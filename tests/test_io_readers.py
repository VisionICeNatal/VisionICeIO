"""Tests for new-format readers and ssort roundtrip."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from visioniceio.io.analog import read_analog_new
from visioniceio.io.behaviour import read_behave_new
from visioniceio.io.sorting import read_ssort, write_ssort
from visioniceio.io.spike import read_spike_new
from visioniceio.io.stim import read_stim_new
from visioniceio.io.waveform import read_swave_new


def _tmpfile(data: bytes, suffix: str = ".bin") -> Path:
    """Write bytes to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(data)
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# read_spike_new
# ---------------------------------------------------------------------------


class TestReadSpikeNew:
    def test_single_record(self):
        timestamps = np.array([100, 200, 300], dtype=np.uint32)
        buf = struct.pack(">I", 3) + timestamps.astype(">u4").tobytes()
        path = _tmpfile(buf, ".spike")
        result = read_spike_new(path)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], timestamps)

    def test_empty_record(self):
        buf = struct.pack(">I", 0)
        path = _tmpfile(buf, ".spike")
        result = read_spike_new(path)
        assert len(result) == 1
        assert len(result[0]) == 0

    def test_corrupt_count_raises(self):
        # Claim 1 million timestamps but file is tiny
        buf = struct.pack(">I", 1_000_000)
        path = _tmpfile(buf, ".spike")
        with pytest.raises(EOFError):
            read_spike_new(path)


# ---------------------------------------------------------------------------
# read_swave_new
# ---------------------------------------------------------------------------


class TestReadSwaveNew:
    def test_single_record(self):
        count, pts = 2, 3
        samples = np.arange(count * pts, dtype=np.int16)
        buf = struct.pack(">II", count, pts) + samples.astype(">i2").tobytes()
        path = _tmpfile(buf, ".swave")
        data, wf_pts = read_swave_new(path)
        assert wf_pts == pts
        assert len(data) == 1
        assert data[0].shape == (count, pts)


# ---------------------------------------------------------------------------
# read_stim_new
# ---------------------------------------------------------------------------


class TestReadStimNew:
    def test_basic(self):
        labels = np.array([1, 2, 3, 1], dtype=np.uint32)
        buf = struct.pack(">I", 4) + labels.astype(">u4").tobytes()
        path = _tmpfile(buf, ".stim")
        result = read_stim_new(path)
        np.testing.assert_array_equal(result, labels.astype(np.int32))

    def test_corrupt_count_raises(self):
        buf = struct.pack(">I", 999_999)
        path = _tmpfile(buf, ".stim")
        with pytest.raises(EOFError):
            read_stim_new(path)


# ---------------------------------------------------------------------------
# read_analog_new
# ---------------------------------------------------------------------------


class TestReadAnalogNew:
    def test_single_record(self):
        samples = np.array([10, -20, 30], dtype=np.int16)
        buf = struct.pack(">I", 3) + samples.astype(">i2").tobytes()
        path = _tmpfile(buf, ".analog")
        result = read_analog_new(path)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], samples)


# ---------------------------------------------------------------------------
# read_behave_new
# ---------------------------------------------------------------------------


class TestReadBehaveNew:
    def test_basic(self):
        codes = np.array([0, 1, 0, 2], dtype=np.uint32)
        buf = struct.pack(">I", 4) + codes.astype(">u4").tobytes()
        path = _tmpfile(buf, ".behave")
        result = read_behave_new(path)
        np.testing.assert_array_equal(result, codes.astype(np.int32))


# ---------------------------------------------------------------------------
# write_ssort / read_ssort roundtrip
# ---------------------------------------------------------------------------


class TestSsortRoundtrip:
    def test_roundtrip(self):
        labels = [np.array([0, 1, 2]), np.array([1, 1])]
        indices = [np.array([10.0, 20.0, 30.0]), np.array([5.0, 15.0])]

        path = _tmpfile(b"", ".ssort")
        write_ssort(path, labels, indices, n_fields=5)
        records = read_ssort(path)

        assert len(records) == 2
        assert records[0]["n_spikes"] == 3
        assert records[1]["n_spikes"] == 2
        np.testing.assert_array_equal(records[0]["labels"], [0, 1, 2])
        np.testing.assert_array_almost_equal(
            records[0]["spike_indices"], [10.0, 20.0, 30.0]
        )

    def test_empty_record(self):
        labels = [np.array([], dtype=np.int32)]
        indices = [np.array([], dtype=np.float32)]

        path = _tmpfile(b"", ".ssort")
        write_ssort(path, labels, indices, n_fields=5)
        records = read_ssort(path)

        assert len(records) == 1
        assert records[0]["n_spikes"] == 0
