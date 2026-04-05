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
