"""Tests for new-format readers and ssort roundtrip."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from visioniceio.io._helpers import read_data
from visioniceio.io.analog import read_analog_new
from visioniceio.io.behaviour import read_behave_new
from visioniceio.io.metadata import read_info_new, read_metadata_ifo
from visioniceio.io.sorting import read_ssort, write_ssort
from visioniceio.io.spike import read_spike_new
from visioniceio.io.stim import read_stim_new
from visioniceio.io.waveform import read_swave_new
from visioniceio.io.zarr_io import _check_zarr_version_compat, load_from_zarr


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
# .swa <-> .swave cross-format equivalence
# ---------------------------------------------------------------------------


def _build_swa_bytes(records: list[np.ndarray]) -> bytes:
    """Build a minimal DLTG ``.swa`` file holding 2-D int16 waveform records.

    Each record is an ``(n_spikes, wf_pts) int16`` array.  Layout matches
    the real lab files: 18-byte header (no descriptor), 128-entry main
    offset table starting at byte 18, then records back-to-back.  Each
    record body is ``[i32 n_spikes][i32 wf_pts][n_spikes*wf_pts*i2 BE]``.
    """
    ndim = len(records)
    assert ndim <= 128, "test helper only supports direct-mode .swa"
    header_size = 18  # magic(4)+version(4)+ndim(4)+p(4)+ld(2), no descriptor
    table_start = header_size
    records_start = table_start + 128 * 4

    record_offsets: list[int] = []
    cur = records_start
    record_bodies: list[bytes] = []
    for arr in records:
        n, w = arr.shape
        body = struct.pack(">ii", n, w) + arr.astype(">i2").tobytes()
        record_bodies.append(body)
        record_offsets.append(cur)
        cur += len(body)

    buf = bytearray()
    buf += b"DTLG"
    buf += b"\x00\x00\x00\x01"
    buf += struct.pack(">I", ndim)
    buf += struct.pack(">I", table_start)
    buf += struct.pack(">h", 0)
    main = [0] * 128
    for i, off in enumerate(record_offsets):
        main[i] = off
    for v in main:
        buf += struct.pack(">I", v)
    for body in record_bodies:
        buf += body
    return bytes(buf)


def _build_swave_bytes(records: list[np.ndarray], wf_pts: int) -> bytes:
    """Build a minimal headerless ``.swave`` file.

    Per-record layout: ``[u32 n_spikes][u32 wf_pts][n_spikes*wf_pts*i2 BE]``,
    records back-to-back.  ``wf_pts`` is the same on every record (matching
    real lab files).
    """
    buf = bytearray()
    for arr in records:
        n = arr.shape[0]
        buf += struct.pack(">II", n, wf_pts)
        buf += arr.astype(">i2").tobytes()
    return bytes(buf)


class TestSwaSwaveEquivalence:
    """Old-format ``.swa`` (DLTG) and new-format ``.swave`` (headerless)
    must yield byte-identical NumPy arrays when written from the same source.

    Real-data validation: the paired dataset ``c5607a07_n`` (lab archive)
    contains both files for the same experiment — 7680 records, 1.47M
    spikes — and the two readers were verified to produce 0 value
    mismatches end-to-end.  This synthetic test locks the same property
    into CI.
    """

    def _make_records(self, shapes: list[tuple[int, int]]) -> list[np.ndarray]:
        """Build deterministic int16 waveforms with the given shapes.

        Uses a per-record fixed-offset arange so the test stays
        reproducible without depending on an RNG fixture.
        """
        recs: list[np.ndarray] = []
        offset = 0
        for n, w in shapes:
            count = n * w
            arr = (np.arange(count, dtype=np.int32) + offset) % 1024
            arr = arr.astype(np.int16) - 512  # mix in negative values
            recs.append(arr.reshape(n, w))
            offset += count
        return recs

    def test_mixed_record_sizes_byte_identical(self):
        wf_pts = 38  # matches real lab data
        shapes = [(3, wf_pts), (1, wf_pts), (5, wf_pts), (0, wf_pts), (2, wf_pts)]
        records = self._make_records(shapes)

        swa_path = _tmpfile(_build_swa_bytes(records), ".swa")
        swave_path = _tmpfile(_build_swave_bytes(records, wf_pts), ".swave")

        swa = read_data(str(swa_path), "int16", 2)
        swave, swave_wf_pts = read_swave_new(str(swave_path))

        assert swave_wf_pts == wf_pts
        assert len(swa) == len(swave) == len(records)
        for i, (a, b, orig) in enumerate(zip(swa, swave, records)):
            assert a.shape == b.shape == orig.shape, (
                f"shape mismatch at record {i}: .swa={a.shape}, .swave={b.shape}, orig={orig.shape}"
            )
            np.testing.assert_array_equal(
                a, b, err_msg=f"swa vs swave value mismatch at record {i}"
            )
            np.testing.assert_array_equal(
                a, orig, err_msg=f"swa vs source value mismatch at record {i}"
            )

    def test_single_spike_records(self):
        """All-singleton records (n_spikes=1) — common in low-activity trials."""
        wf_pts = 38
        records = self._make_records([(1, wf_pts)] * 10)
        swa = read_data(str(_tmpfile(_build_swa_bytes(records), ".swa")), "int16", 2)
        swave, _ = read_swave_new(str(_tmpfile(_build_swave_bytes(records, wf_pts), ".swave")))
        for a, b in zip(swa, swave):
            np.testing.assert_array_equal(a, b)


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
        np.testing.assert_array_almost_equal(records[0]["spike_indices"], [10.0, 20.0, 30.0])

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
            path,
            labels,
            indices,
            n_fields=10,
            amp_max_per_record=amp_max,
            amp_min_per_record=amp_min,
            peak_to_peak_per_record=p2p,
            width_per_record=width,
        )
        records = read_ssort(path)
        assert records[0]["variant"] == "v10"
        np.testing.assert_array_equal(records[0]["labels"], [1, 2, 3])
        np.testing.assert_array_almost_equal(records[0]["amp_max"], [200.0, 210.0, 220.0])
        np.testing.assert_array_almost_equal(records[0]["amp_min"], [-100.0, -110.0, -120.0])
        np.testing.assert_array_almost_equal(records[0]["peak_to_peak"], [300.0, 320.0, 340.0])
        np.testing.assert_array_almost_equal(records[0]["width"], [8.0, 10.0, 12.0])


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
            np.array(
                [
                    [266.9, -8.5, -103.6, 125.8, 4.4, 23.3],
                    [270.8, -16.5, 42.0, -4.2, 41.1, -75.1],
                    [270.5, -8.5, -103.6, 125.8, 4.4, 23.3],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [267.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [268.0, -1.0, -2.0, -3.0, -4.0, -5.0],
                ],
                dtype=np.float32,
            ),
        ]

        path = _tmpfile(b"", ".ssort")
        write_ssort(
            path,
            labels,
            indices,
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
        np.testing.assert_array_almost_equal(records[0]["spike_indices"], [1234.0, 5678.0, 9012.0])
        np.testing.assert_array_almost_equal(records[0]["amp_max"], [196, 206, 208])
        np.testing.assert_array_almost_equal(records[0]["amp_min"], [-104, -97, -104])
        np.testing.assert_array_almost_equal(records[0]["peak_to_peak"], [300, 303, 312])
        np.testing.assert_array_almost_equal(records[0]["width"], [8, 17, 21])
        np.testing.assert_array_almost_equal(records[0]["features"], features[0])

    def test_v16_empty_record_among_real_data(self):
        """Empty records (n_entries=0) must round-trip cleanly within a v16
        file that also has non-empty records (so the variant is detectable)."""
        labels = [
            np.array([1, 2], dtype=np.int32),
            np.array([], dtype=np.int32),  # empty
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
            [2, 7, 3, 12345, 4, 0, 200, -80, 280, 10, 265.5, 10, 20, 30, 40, 50],
            dtype=">f4",
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
        np.testing.assert_array_almost_equal(r["features"][0], [265.5, 10, 20, 30, 40, 50])


class TestSsortV10EmptyOnDisk:
    """The on-disk layout for empty channel-trial records in Variant A.

    Real lab v10 files always use ``n_entries=1`` (header-only) for empty
    channel-trials, never the 8-byte ``[0, 0]`` sentinel.  These tests
    pin the writer to that convention, and the reader to accepting both.
    """

    def test_v10_empty_record_writes_header_only(self):
        """Empty v10 record on disk: 8-byte header + 1 row of n_fields floats."""
        labels = [np.array([], dtype=np.int32)]
        indices = [np.array([], dtype=np.float32)]
        path = _tmpfile(b"", ".ssort")
        write_ssort(
            path,
            labels,
            indices,
            n_fields=10,
            channel_indices=[3],
            trial_indices=[7],
            stim_conditions=[5],
        )
        raw = Path(path).read_bytes()
        # 8 bytes header (n_entries=1, n_fields=10) + 40 bytes header row
        assert len(raw) == 48
        n_entries, n_fields = struct.unpack(">II", raw[:8])
        assert n_entries == 1
        assert n_fields == 10
        header = np.frombuffer(raw[8:48], dtype=">f4")
        # channel=3, n_spikes=0, trial=7, stim=5, then six zeros
        assert header[0] == 3.0
        assert header[1] == 0.0
        assert header[2] == 7.0
        assert header[3] == 5.0
        assert np.all(header[4:] == 0.0)

    def test_v10_empty_record_reads_back_with_metadata(self):
        """The reader recovers channel/trial/stim from a header-only empty
        record (which the [0,0] sentinel form cannot carry)."""
        labels = [np.array([], dtype=np.int32)]
        indices = [np.array([], dtype=np.float32)]
        path = _tmpfile(b"", ".ssort")
        write_ssort(
            path,
            labels,
            indices,
            n_fields=10,
            channel_indices=[3],
            trial_indices=[7],
            stim_conditions=[5],
        )
        records = read_ssort(path)
        assert len(records) == 1
        r = records[0]
        assert r["variant"] == "v10"
        assert r["n_spikes"] == 0
        assert r["channel_idx"] == 3
        assert r["trial_idx"] == 7
        assert r["stim_condition"] == 5
        assert r["labels"].shape == (0,)

    def test_v10_reader_still_accepts_zero_zero_sentinel(self):
        """Back-compat: a manually crafted [0, 0] sentinel must still parse.

        This guards against breaking any v10 file that was written by a
        prior version of write_ssort that used the sentinel form."""
        # One [0, 0] sentinel record, then one normal 1-spike record.
        sentinel = struct.pack(">II", 0, 0)
        row = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=">f4")
        header = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=">f4")
        nonempty = struct.pack(">II", 2, 10) + header.tobytes() + row.tobytes()
        path = _tmpfile(sentinel + nonempty, ".ssort")
        records = read_ssort(path)
        assert len(records) == 2
        assert records[0]["variant"] == "v10"
        assert records[0]["n_spikes"] == 0
        assert records[1]["variant"] == "v10"
        assert records[1]["n_spikes"] == 1

    def test_v10_full_roundtrip_byte_identical_for_mixed_empty(self):
        """Write a mix of empty + non-empty v10 records, read back, write
        again -> the second-pass file must be byte-identical to the first."""
        labels = [
            np.array([], dtype=np.int32),
            np.array([1, 2, 3], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([4], dtype=np.int32),
        ]
        indices = [
            np.array([], dtype=np.float32),
            np.array([10.0, 20.0, 30.0], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([55.0], dtype=np.float32),
        ]
        amp_max = [
            np.array([], dtype=np.float32),
            np.array([100.0, 110.0, 120.0], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([130.0], dtype=np.float32),
        ]
        path1 = _tmpfile(b"", ".ssort")
        path2 = _tmpfile(b"", ".ssort")
        common_kwargs = dict(
            n_fields=10,
            channel_indices=[0, 1, 2, 3],
            trial_indices=[0, 0, 1, 1],
            stim_conditions=[5, 5, 6, 6],
            amp_max_per_record=amp_max,
        )
        write_ssort(path1, labels, indices, **common_kwargs)
        records = read_ssort(path1)
        # Re-write from the parsed records and compare byte-for-byte
        write_ssort(
            path2,
            [r["labels"] for r in records],
            [r["spike_indices"] for r in records],
            n_fields=10,
            channel_indices=[r["channel_idx"] for r in records],
            trial_indices=[r["trial_idx"] for r in records],
            stim_conditions=[r["stim_condition"] for r in records],
            amp_max_per_record=[r["amp_max"] for r in records],
        )
        assert Path(path1).read_bytes() == Path(path2).read_bytes()


# ---------------------------------------------------------------------------
# load_from_zarr — version mismatch guard
# ---------------------------------------------------------------------------


class TestLoadFromZarrVersionGuard:
    """``load_from_zarr`` surfaces a clear error for a v3 store opened with
    zarr v2 (otherwise xarray re-raises as a misleading FileNotFoundError
    against a path that clearly exists)."""

    def _make_v3_marker_store(self, tmp_path: Path) -> Path:
        """Create a directory that looks like a zarr v3 store (top-level
        ``zarr.json``, no ``.zmetadata``).  We don't need a real store —
        the version check runs before xarray touches it."""
        store = tmp_path / "fake_v3.zarr"
        store.mkdir()
        (store / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}')
        return store

    def _make_v2_marker_store(self, tmp_path: Path) -> Path:
        """Directory with a ``.zmetadata`` marker (looks like consolidated
        zarr v2)."""
        store = tmp_path / "fake_v2.zarr"
        store.mkdir()
        (store / ".zmetadata").write_text('{"zarr_consolidated_format": 1, "metadata": {}}')
        return store

    def test_v3_store_under_v2_raises_clear_error(self, tmp_path, monkeypatch):
        """When zarr is v2 and the store has ``zarr.json`` (no ``.zmetadata``),
        we raise ValueError with a useful message — not the misleading
        FileNotFoundError xarray would surface."""
        import zarr

        monkeypatch.setattr(zarr, "__version__", "2.18.3")
        store = self._make_v3_marker_store(tmp_path)
        with pytest.raises(ValueError, match=r"v3 format.*installed zarr is v2"):
            _check_zarr_version_compat(str(store))

    def test_v3_store_under_v3_no_error(self, tmp_path, monkeypatch):
        """Under zarr v3, the same store must not trigger the version check."""
        import zarr

        monkeypatch.setattr(zarr, "__version__", "3.0.0")
        store = self._make_v3_marker_store(tmp_path)
        # Must not raise
        _check_zarr_version_compat(str(store))

    def test_v2_store_under_v2_no_error(self, tmp_path, monkeypatch):
        """A v2 store under zarr v2 must not trigger the version check."""
        import zarr

        monkeypatch.setattr(zarr, "__version__", "2.18.3")
        store = self._make_v2_marker_store(tmp_path)
        _check_zarr_version_compat(str(store))

    def test_v2_marker_alongside_v3_marker_no_error(self, tmp_path, monkeypatch):
        """If both markers are present (unusual but valid), defer to xarray."""
        import zarr

        monkeypatch.setattr(zarr, "__version__", "2.18.3")
        store = tmp_path / "mixed.zarr"
        store.mkdir()
        (store / "zarr.json").write_text("{}")
        (store / ".zmetadata").write_text("{}")
        _check_zarr_version_compat(str(store))

    def test_load_from_zarr_missing_path_raises_filenotfound(self, tmp_path):
        """Pre-existing behaviour: a non-existent path raises FileNotFoundError."""
        missing = tmp_path / "nope.zarr"
        with pytest.raises(FileNotFoundError, match="Zarr store not found"):
            load_from_zarr(str(missing))

    def test_load_from_zarr_surface_error_path(self, tmp_path, monkeypatch):
        """End-to-end: load_from_zarr on a v3-only store with zarr v2
        installed gives the clear ValueError, not FileNotFoundError."""
        import zarr

        monkeypatch.setattr(zarr, "__version__", "2.18.3")
        store = self._make_v3_marker_store(tmp_path)
        with pytest.raises(ValueError, match=r"v3 format.*installed zarr is v2"):
            load_from_zarr(str(store))


# ---------------------------------------------------------------------------
# read_swave_new — constant wf_pts across records
# ---------------------------------------------------------------------------


class TestSwaveMixedPtsRejected:
    """``read_swave_new`` must reject files whose records disagree on the
    snippet length (``wf_pts``), since downstream consumers
    (``Experiment._pad_waveforms``) assume a single global value."""

    def test_two_records_with_different_pts_raise(self):
        # Record 0: 1 spike x 4 pts ; record 1: 1 spike x 5 pts
        buf = b""
        # Record 0
        buf += struct.pack(">II", 1, 4)
        buf += np.array([[1, 2, 3, 4]], dtype=">i2").tobytes()
        # Record 1 (different pts!)
        buf += struct.pack(">II", 1, 5)
        buf += np.array([[1, 2, 3, 4, 5]], dtype=">i2").tobytes()

        path = _tmpfile(buf, ".swave")
        with pytest.raises(ValueError, match=r"inconsistent snippet length"):
            read_swave_new(str(path))

    def test_uniform_pts_still_works(self):
        """The sanity check must not break the common case (every record
        has the same pts)."""
        buf = b""
        for n_sp in (2, 0, 1):
            buf += struct.pack(">II", n_sp, 3)
            if n_sp > 0:
                buf += np.zeros((n_sp, 3), dtype=">i2").tobytes()
        path = _tmpfile(buf, ".swave")
        records, wf_pts = read_swave_new(str(path))
        assert wf_pts == 3
        assert [r.shape for r in records] == [(2, 3), (0, 3), (1, 3)]


# ---------------------------------------------------------------------------
# read_data — dim / nd guards
# ---------------------------------------------------------------------------


def _make_minimal_dltg(records: list[tuple[tuple[int, ...], bytes]]) -> bytes:
    """Build a minimal DLTG container holding *records*.

    Each record is ``((dim1, dim2, ...), payload_bytes)``.  Used by tests
    that need to feed pathological inputs (e.g. negative dims) to
    ``read_data``.
    """
    n = len(records)
    # Header is 4 (DTLG) + 4 (version) + 4 (ndim) + 4 (p) + 2 (ld)
    # + 0 (descriptor) = 18 bytes
    header_size = 18
    # Offset table starts immediately after header
    offset_table_start = header_size
    # Each record needs 4*len(dims) for the dim header + len(payload) bytes
    record_offsets: list[int] = []
    cursor = offset_table_start + 128 * 4  # offset table is always 128 u32
    for dims, payload in records:
        record_offsets.append(cursor)
        cursor += 4 * len(dims) + len(payload)

    buf = bytearray()
    buf += b"DTLG"
    buf += b"\x00\x00\x00\x01"  # version
    buf += struct.pack(">I", n)  # ndim
    buf += struct.pack(">I", offset_table_start)  # p
    buf += struct.pack(">h", 0)  # ld (descriptor length)

    # Offset table: 128 u32 BE, with the first `n` filled
    table = np.zeros(128, dtype=">u4")
    for i, off in enumerate(record_offsets):
        table[i] = off
    buf += table.tobytes()

    # Records
    for dims, payload in records:
        for d in dims:
            buf += struct.pack(">i", d)
        buf += payload
    return bytes(buf)


class TestReadDataDimGuards:
    """``read_data`` must reject malformed dim headers with clean
    ``ValueError`` messages instead of producing phantom output or
    surfacing cryptic NumPy errors."""

    def test_nd_zero_rejected(self):
        # The file itself is well-formed; nd=0 is invalid at the API level.
        buf = _make_minimal_dltg([((1,), b"\x00" * 2)])
        path = _tmpfile(buf, ".bin")
        with pytest.raises(ValueError, match=r"nd must be >= 1"):
            read_data(str(path), "int16", 0)

    def test_negative_dim_rejected(self):
        # Record declares dim = -1 (signed int32 unpacking).
        payload = b""  # any payload, we won't get that far
        buf = _make_minimal_dltg([((-1,), payload)])
        path = _tmpfile(buf, ".bin")
        with pytest.raises(ValueError, match=r"negative dimensions"):
            read_data(str(path), "int16", 1)

    def test_zero_dim_accepted_for_empty_records(self):
        # Zero is a legitimate dim — old-format .swa empty channel-trials
        # are stored as shape (0, n_pts).  The guard must NOT reject this.
        buf = _make_minimal_dltg([((0, 5), b"")])
        path = _tmpfile(buf, ".bin")
        records = read_data(str(path), "int16", 2)
        assert len(records) == 1
        assert records[0].shape == (0, 5)


# ---------------------------------------------------------------------------
# read_metadata_ifo — truncated DLTG falls back to text reader
# ---------------------------------------------------------------------------


class TestReadMetadataIfoTruncated:
    """``read_metadata_ifo`` docstring promises a text-mode fallback when
    the DLTG container cannot be parsed.  A truncated DLTG header raises
    ``EOFError`` from ``_read_dltg_header``; the previous catch tuple was
    missing ``EOFError`` and let the failure escape."""

    def test_truncated_dltg_falls_back_to_text(self, tmp_path):
        # DTLG magic + version + truncated ndim (only 2 of 4 bytes)
        truncated = b"DTLG\x00\x00\x00\x01\x00\x00"
        path = tmp_path / "truncated.ifo"
        path.write_bytes(truncated)
        # Must NOT raise EOFError — must fall through to read_metadata,
        # which returns {} for binary-looking content.
        result = read_metadata_ifo(str(path))
        assert result == {}


# ---------------------------------------------------------------------------
# read_info_new — bogus record2_size raises clear ValueError
# ---------------------------------------------------------------------------


class TestReadInfoNewBogusRecord2Size:
    """``read_info_new`` must validate the declared ``record2_size``
    against the available file bytes instead of trusting it blindly and
    letting downstream readers surface a less-clear error."""

    def test_record2_size_past_eof_raises(self, tmp_path):
        # File: PTH0 record1 (size=0) + PTH0 record2 with size=0xFFFFFFFF
        # but only a few bytes of actual data afterwards.
        buf = bytearray()
        buf += b"PTH0"
        buf += struct.pack(">I", 0)  # record 1 size
        buf += b"PTH0"
        buf += struct.pack(">I", 0xFFFFFFFF)  # record 2 size — implausibly large
        buf += b"\x00" * 16  # only a handful of bytes after

        path = tmp_path / "bogus.info"
        path.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match=r"PTH0 record declares size"):
            read_info_new(str(path))
