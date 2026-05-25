"""Tests for visioniceio.io._helpers — byte-level primitives and parsers."""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from visioniceio.io._helpers import (
    _parse_metadata_value,
    _read_exact,
    _read_lv_string,
    dtype_map,
)

# ---------------------------------------------------------------------------
# _read_exact
# ---------------------------------------------------------------------------


class TestReadExact:
    def test_reads_exact_bytes(self):
        f = io.BytesIO(b"hello world")
        assert _read_exact(f, 5) == b"hello"

    def test_raises_on_short_read(self):
        f = io.BytesIO(b"hi")
        with pytest.raises(EOFError, match="wanted 10 bytes, got 2"):
            _read_exact(f, 10)

    def test_zero_bytes(self):
        f = io.BytesIO(b"")
        assert _read_exact(f, 0) == b""


# ---------------------------------------------------------------------------
# _read_lv_string
# ---------------------------------------------------------------------------


class TestReadLvString:
    def _pack(self, text: str) -> bytes:
        encoded = text.encode("ascii")
        return struct.pack(">I", len(encoded)) + encoded

    def test_simple_string(self):
        data = self._pack("hello")
        s, pos = _read_lv_string(data, 0)
        assert s == "hello"
        assert pos == 4 + 5

    def test_empty_string(self):
        data = self._pack("")
        s, pos = _read_lv_string(data, 0)
        assert s == ""
        assert pos == 4

    def test_offset(self):
        prefix = b"\x00" * 10
        data = prefix + self._pack("test")
        s, pos = _read_lv_string(data, 10)
        assert s == "test"
        assert pos == 10 + 4 + 4

    def test_truncated_length_prefix(self):
        data = b"\x00\x00"  # only 2 bytes, need 4
        with pytest.raises(EOFError, match="length prefix"):
            _read_lv_string(data, 0)

    def test_truncated_string_body(self):
        # Declare 100-byte string but only provide 3 bytes
        data = struct.pack(">I", 100) + b"abc"
        with pytest.raises(EOFError, match="declared length 100"):
            _read_lv_string(data, 0)


# ---------------------------------------------------------------------------
# _parse_metadata_value
# ---------------------------------------------------------------------------


class TestParseMetadataValue:
    def test_integer(self):
        assert _parse_metadata_value("42") == 42

    def test_float(self):
        assert _parse_metadata_value("3.14") == pytest.approx(3.14)

    def test_european_decimal(self):
        assert _parse_metadata_value("250,00") == pytest.approx(250.0)

    def test_boolean_yes(self):
        assert _parse_metadata_value("yes") is True

    def test_boolean_no(self):
        assert _parse_metadata_value("no") is False

    def test_comma_separated_ints(self):
        assert _parse_metadata_value("1, 2, 3") == [1, 2, 3]

    def test_plain_string(self):
        assert _parse_metadata_value("hello world") == "hello world"


# ---------------------------------------------------------------------------
# dtype_map
# ---------------------------------------------------------------------------


def test_dtype_map_has_standard_types():
    for key in ("uint32", "int32", "uint16", "int16", "float32", "float64"):
        assert key in dtype_map
        np_dtype, size = dtype_map[key]
        assert np.dtype(np_dtype).itemsize == size


# ---------------------------------------------------------------------------
# Imports / public API smoke test
# ---------------------------------------------------------------------------


def test_top_level_imports():
    import visioniceio

    assert hasattr(visioniceio, "Experiment")
    assert hasattr(visioniceio, "read_data")
    assert hasattr(visioniceio, "read_spike_new")
    assert hasattr(visioniceio, "read_swave_new")
    assert hasattr(visioniceio, "read_stim_new")
    assert hasattr(visioniceio, "read_analog_new")
    assert hasattr(visioniceio, "read_bhv")
    assert hasattr(visioniceio, "read_behave_new")
    assert hasattr(visioniceio, "read_ssort")
    assert hasattr(visioniceio, "write_ssort")
    assert hasattr(visioniceio, "read_metadata")
    assert hasattr(visioniceio, "read_metadata_ifo")
    assert hasattr(visioniceio, "read_info_new")
    assert hasattr(visioniceio, "load_from_zarr")


def test_backward_compat_imports():
    from visioniceio.core_io import read_data, read_spike_new

    assert callable(read_data)
    assert callable(read_spike_new)


# ---------------------------------------------------------------------------
# DLTG offset table — direct mode (ndim <= 128) and two-level mode (ndim > 128)
# ---------------------------------------------------------------------------

import tempfile  # noqa: E402

from visioniceio.io._helpers import _read_dltg_header, read_data  # noqa: E402


def _build_dltg(records: list[bytes], dtype: str = "int32") -> bytes:
    """Build a synthetic DLTG file in memory and return its bytes.

    Each entry in *records* is the raw payload for one record (just the
    bytes after the per-record dim header; the header is added by this
    function from the record's len/itemsize).  This helper handles both
    direct mode (ndim <= 128) and two-level mode (ndim > 128) automatically.

    Each record is encoded as ``[int32 dim][dim × itemsize bytes]``.
    """
    itemsize = {"int16": 2, "int32": 4, "uint32": 4, "float32": 4}[dtype]
    ndim = len(records)

    # Decide layout
    if ndim <= 128:
        n_chunks = 0  # direct mode, no sub-tables
    else:
        n_chunks = (ndim + 127) // 128
        assert n_chunks <= 128, "test helper does not support ndim > 16384"

    # Header is 4 + 4 + 4 + 4 + 2 + 0 = 18 bytes (no descriptor)
    header_size = 18

    # Layout decision:
    # - bytes 0..17: header
    # - byte 18: main offset table (always 128 × 4 = 512 bytes)
    # - byte 530: (two-level only) chunk sub-tables, each 512 bytes
    # - then records back-to-back
    p = header_size
    if ndim <= 128:
        records_start = p + 128 * 4
        sub_table_offsets: list[int] = []
    else:
        records_start = p + 128 * 4 + n_chunks * 128 * 4
        sub_table_offsets = [
            p + 128 * 4 + ci * 128 * 4 for ci in range(n_chunks)
        ]

    # Compute absolute offset of every record
    record_abs_offsets: list[int] = []
    cur = records_start
    for r in records:
        record_abs_offsets.append(cur)
        # each record stores: int32 dim + len(r) bytes payload
        cur += 4 + len(r)

    # Build the file
    buf = bytearray()
    # DLTG header
    buf += b"DTLG"
    buf += b"\x00\x00\x00\x01"           # version
    buf += struct.pack(">I", ndim)
    buf += struct.pack(">I", p)
    buf += struct.pack(">h", 0)          # ld = 0 (no descriptor)
    assert len(buf) == header_size

    # Main offset table (128 entries)
    main = [0] * 128
    if ndim <= 128:
        for i, off in enumerate(record_abs_offsets):
            main[i] = off
    else:
        for ci, sub_off in enumerate(sub_table_offsets):
            main[ci] = sub_off
    for v in main:
        buf += struct.pack(">I", v)

    # Per-chunk sub-tables (two-level only)
    if ndim > 128:
        for ci in range(n_chunks):
            sub = [0] * 128
            start = ci * 128
            stop = min(start + 128, ndim)
            for j, off in enumerate(record_abs_offsets[start:stop]):
                sub[j] = off
            for v in sub:
                buf += struct.pack(">I", v)

    # Records
    for r in records:
        n_elements = len(r) // itemsize
        buf += struct.pack(">i", n_elements)
        buf += r

    return bytes(buf)


def _record_int32(values: list[int]) -> bytes:
    """Pack a list of ints as big-endian int32 bytes (the payload of one record)."""
    return b"".join(struct.pack(">i", v) for v in values)


class TestDLTGDirectMode:
    """ndim <= 128: main table entries are direct record offsets."""

    def test_small_ndim(self):
        recs = [_record_int32([10, 20, 30]), _record_int32([40])]
        f = tempfile.NamedTemporaryFile(suffix=".dltg", delete=False)
        f.write(_build_dltg(recs, dtype="int32"))
        f.close()

        with open(f.name, "rb") as fh:
            ndim, offsets, desc = _read_dltg_header(fh)
        assert ndim == 2
        assert offsets.shape == (2,)
        assert offsets.dtype == np.uint32

        data = read_data(f.name, "int32", 1)
        assert len(data) == 2
        np.testing.assert_array_equal(data[0], [10, 20, 30])
        np.testing.assert_array_equal(data[1], [40])

    def test_ndim_128_exact_boundary(self):
        """ndim == 128 still uses direct mode (last value in main table)."""
        recs = [_record_int32([i]) for i in range(128)]
        f = tempfile.NamedTemporaryFile(suffix=".dltg", delete=False)
        f.write(_build_dltg(recs, dtype="int32"))
        f.close()

        with open(f.name, "rb") as fh:
            ndim, offsets, _ = _read_dltg_header(fh)
        assert ndim == 128
        assert len(offsets) == 128
        data = read_data(f.name, "int32", 1)
        assert len(data) == 128
        np.testing.assert_array_equal(data[127], [127])


class TestDLTGTwoLevelMode:
    """ndim > 128: main table entries point to per-chunk sub-tables."""

    def test_ndim_129_just_over_boundary(self):
        """ndim == 129 triggers two-level mode with 2 chunks (128 + 1)."""
        recs = [_record_int32([i, i + 1]) for i in range(129)]
        f = tempfile.NamedTemporaryFile(suffix=".dltg", delete=False)
        f.write(_build_dltg(recs, dtype="int32"))
        f.close()

        with open(f.name, "rb") as fh:
            ndim, offsets, _ = _read_dltg_header(fh)
        assert ndim == 129
        assert len(offsets) == 129
        # All offsets should be monotonically increasing (records packed in order)
        assert np.all(np.diff(offsets) > 0)
        data = read_data(f.name, "int32", 1)
        np.testing.assert_array_equal(data[0], [0, 1])
        np.testing.assert_array_equal(data[128], [128, 129])

    def test_ndim_300_three_chunks(self):
        """ndim = 300 = 128 + 128 + 44 → 3 sub-tables."""
        recs = [_record_int32([i] * (1 + i % 5)) for i in range(300)]
        f = tempfile.NamedTemporaryFile(suffix=".dltg", delete=False)
        f.write(_build_dltg(recs, dtype="int32"))
        f.close()

        with open(f.name, "rb") as fh:
            ndim, offsets, _ = _read_dltg_header(fh)
        assert ndim == 300
        assert len(offsets) == 300
        data = read_data(f.name, "int32", 1)
        for i in range(300):
            np.testing.assert_array_equal(data[i], [i] * (1 + i % 5))

    def test_ndim_overflow_raises(self):
        """ndim > 128*128=16384 must raise a clear ValueError."""
        import io
        # Build a header that claims ndim=20000 — we don't need real records.
        buf = bytearray()
        buf += b"DTLG"
        buf += b"\x00\x00\x00\x01"
        buf += struct.pack(">I", 20_000)   # ndim
        buf += struct.pack(">I", 18)        # p (table right after header)
        buf += struct.pack(">h", 0)         # ld
        # 128 × 4 zero bytes for the main table
        buf += b"\x00" * (128 * 4)

        with pytest.raises(ValueError, match="exceeds the two-level"):
            _read_dltg_header(io.BytesIO(bytes(buf)))


class TestReadDataValidation:
    """read_data should give clear errors on truncated / corrupt files."""

    def test_truncated_record_raises(self):
        """A record that claims more bytes than remain in the file errors."""
        # Build a tiny DLTG file with 1 record claiming 1_000_000 int32s
        recs = [_record_int32([1])]  # legitimate 1-element record
        good = _build_dltg(recs, dtype="int32")
        # Corrupt: rewrite the record's dim header to claim 1M elements.
        # In direct-mode layout, record 0 sits at byte 18 + 128*4 = 530.
        corrupted = bytearray(good)
        struct.pack_into(">i", corrupted, 530, 1_000_000)

        f = tempfile.NamedTemporaryFile(suffix=".dltg", delete=False)
        f.write(bytes(corrupted))
        f.close()

        with pytest.raises(ValueError, match="bytes remain in the file"):
            read_data(f.name, "int32", 1)
