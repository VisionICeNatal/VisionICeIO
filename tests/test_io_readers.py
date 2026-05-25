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
    """Variant A (header + spike rows) read/write roundtrips."""

    def test_roundtrip(self):
        labels = [np.array([0, 1, 2]), np.array([1, 1])]
        indices = [np.array([10.0, 20.0, 30.0]), np.array([5.0, 15.0])]

        path = _tmpfile(b"", ".ssort")
        write_ssort(path, labels, indices, n_fields=5)
        records = read_ssort(path)

        assert len(records) == 2
        assert records[0]["variant"] == "v10"
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
        assert records[0]["labels"].shape == (0,)

    def test_v10_named_columns_roundtrip(self):
        """Roundtrip including amp/p2p/width preserves all values."""
        labels = [np.array([1, 2, 3], dtype=np.int32)]
        indices = [np.array([100.0, 200.0, 300.0], dtype=np.float32)]
        amp_max = [np.array([200.0, 210.0, 220.0], dtype=np.float32)]
        amp_min = [np.array([-100.0, -110.0, -120.0], dtype=np.float32)]
        p2p = [np.array([300.0, 320.0, 340.0], dtype=np.float32)]
        width = [np.array([8.0, 10.0, 12.0], dtype=np.float32)]

        path = _tmpfile(b"", ".ssort")
        write_ssort(
            path, labels, indices, n_fields=10,
            amp_max_per_record=amp_max,
            amp_min_per_record=amp_min,
            peak_to_peak_per_record=p2p,
            width_per_record=width,
        )
        records = read_ssort(path)
        assert records[0]["variant"] == "v10"
        np.testing.assert_array_equal(records[0]["labels"], [1, 2, 3])
        np.testing.assert_array_almost_equal(
            records[0]["amp_max"], [200.0, 210.0, 220.0]
        )
        np.testing.assert_array_almost_equal(
            records[0]["amp_min"], [-100.0, -110.0, -120.0]
        )
        np.testing.assert_array_almost_equal(
            records[0]["peak_to_peak"], [300.0, 320.0, 340.0]
        )
        np.testing.assert_array_almost_equal(
            records[0]["width"], [8.0, 10.0, 12.0]
        )


class TestSsortV16:
    """Variant B (no header, n_fields=16) read/write roundtrips."""

    def test_v16_roundtrip(self):
        labels = [
            np.array([3, 5, 5], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
        ]
        indices = [
            np.array([1234.0, 5678.0, 9012.0], dtype=np.float32),
            np.array([100.0, 200.0], dtype=np.float32),
        ]
        amp_max = [
            np.array([196, 206, 208], dtype=np.float32),
            np.array([180, 220], dtype=np.float32),
        ]
        amp_min = [
            np.array([-104, -97, -104], dtype=np.float32),
            np.array([-80, -120], dtype=np.float32),
        ]
        p2p = [
            np.array([300, 303, 312], dtype=np.float32),
            np.array([260, 340], dtype=np.float32),
        ]
        width = [
            np.array([8, 17, 21], dtype=np.float32),
            np.array([10, 15], dtype=np.float32),
        ]
        # 6 feature columns for v16 (cols 10..15)
        features = [
            np.array([
                [266.9, -8.5, -103.6, 125.8, 4.4, 23.3],
                [270.8, -16.5, 42.0, -4.2, 41.1, -75.1],
                [270.5, -8.5, -103.6, 125.8, 4.4, 23.3],
            ], dtype=np.float32),
            np.array([
                [267.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [268.0, -1.0, -2.0, -3.0, -4.0, -5.0],
            ], dtype=np.float32),
        ]

        path = _tmpfile(b"", ".ssort")
        write_ssort(
            path, labels, indices,
            features_per_record=features,
            n_fields=16,  # selects Variant B
            channel_indices=[0, 1],
            trial_indices=[0, 0],
            stim_conditions=[7, 7],
            amp_max_per_record=amp_max,
            amp_min_per_record=amp_min,
            peak_to_peak_per_record=p2p,
            width_per_record=width,
        )
        records = read_ssort(path)

        assert len(records) == 2
        assert all(r["variant"] == "v16" for r in records)
        # Metadata
        assert records[0]["channel_idx"] == 0
        assert records[0]["trial_idx"] == 0
        assert records[0]["stim_condition"] == 7
        assert records[0]["n_spikes"] == 3
        # Per-spike fields
        np.testing.assert_array_equal(records[0]["labels"], [3, 5, 5])
        np.testing.assert_array_almost_equal(
            records[0]["spike_indices"], [1234.0, 5678.0, 9012.0]
        )
        np.testing.assert_array_almost_equal(
            records[0]["amp_max"], [196, 206, 208]
        )
        np.testing.assert_array_almost_equal(
            records[0]["amp_min"], [-104, -97, -104]
        )
        np.testing.assert_array_almost_equal(
            records[0]["peak_to_peak"], [300, 303, 312]
        )
        np.testing.assert_array_almost_equal(
            records[0]["width"], [8, 17, 21]
        )
        np.testing.assert_array_almost_equal(
            records[0]["features"], features[0]
        )

    def test_v16_empty_record_among_real_data(self):
        """Empty records (n_entries=0) must round-trip cleanly within a v16
        file that also has non-empty records (so the variant is detectable)."""
        labels = [
            np.array([1, 2], dtype=np.int32),
            np.array([], dtype=np.int32),   # empty
            np.array([3], dtype=np.int32),
        ]
        indices = [
            np.array([100.0, 200.0], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([500.0], dtype=np.float32),
        ]
        path = _tmpfile(b"", ".ssort")
        write_ssort(path, labels, indices, n_fields=16)
        records = read_ssort(path)
        assert len(records) == 3
        assert all(r["variant"] == "v16" for r in records)
        assert records[0]["n_spikes"] == 2
        assert records[1]["n_spikes"] == 0
        assert records[2]["n_spikes"] == 1
        # Empty record's per-spike arrays have width == 6 v16 feature cols
        assert records[1]["labels"].shape == (0,)
        assert records[1]["features"].shape == (0, 6)


class TestSsortVariantDetection:
    """Auto-detection of v10 vs v16 from the first non-empty record."""

    def test_detects_v16_after_empty_records(self):
        """Leading empty records must not confuse variant detection."""
        # 2 empty records, then 1 non-empty v16 record
        labels = [
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ]
        indices = [
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([111.0, 222.0], dtype=np.float32),
        ]
        path = _tmpfile(b"", ".ssort")
        write_ssort(path, labels, indices, n_fields=16)
        records = read_ssort(path)
        assert len(records) == 3
        assert records[0]["variant"] == "v16"
        assert records[1]["variant"] == "v16"
        assert records[2]["variant"] == "v16"
        assert records[2]["n_spikes"] == 2

    def test_detects_v10_default(self):
        """An all-empty file defaults to v10."""
        labels = [np.array([], dtype=np.int32)]
        indices = [np.array([], dtype=np.float32)]
        path = _tmpfile(b"", ".ssort")
        write_ssort(path, labels, indices, n_fields=10)
        records = read_ssort(path)
        assert records[0]["variant"] == "v10"

    def test_v16_byte_layout(self):
        """Hand-built v16 bytes parse with the expected per-row decoding."""
        import struct as _s
        # One record, 1 spike: channel=2, cluster=7, trial=3, sample=12345,
        # stim=4, reserved=0, amp_max=200, amp_min=-80, p2p=280, width=10,
        # extra=265.5, pca1..5=10,20,30,40,50
        row = np.array(
            [2, 7, 3, 12345, 4, 0, 200, -80, 280, 10,
             265.5, 10, 20, 30, 40, 50], dtype='>f4',
        )
        buf = _s.pack(">II", 1, 16) + row.tobytes()
        path = _tmpfile(buf, ".ssort")
        records = read_ssort(path)
        assert len(records) == 1
        r = records[0]
        assert r["variant"] == "v16"
        assert r["channel_idx"] == 2
        assert r["trial_idx"] == 3
        assert r["stim_condition"] == 4
        assert r["labels"].tolist() == [7]
        assert r["spike_indices"].tolist() == [12345.0]
        assert r["amp_max"].tolist() == [200.0]
        assert r["amp_min"].tolist() == [-80.0]
        assert r["peak_to_peak"].tolist() == [280.0]
        assert r["width"].tolist() == [10.0]
        np.testing.assert_array_almost_equal(
            r["features"][0], [265.5, 10, 20, 30, 40, 50]
        )
