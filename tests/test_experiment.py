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
                tmpdir,
                name,
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


def _make_minimal_exp(ntrials=2, nelectrodes=2, max_spikes=3):
    """Build a minimal Experiment with a stub xr.Dataset for sorting tests."""
    import xarray as xr

    exp = Experiment()
    exp.ntrials = ntrials
    exp.nelectrodes = nelectrodes
    exp.max_spikes = max_spikes
    # Build a minimal Dataset with the required coords
    coords = {
        "electrodes": np.arange(nelectrodes),
        "trials": np.arange(ntrials),
        "spikes_idx": np.arange(max_spikes),
    }
    exp.data = xr.Dataset(coords=coords)
    return exp


class TestAttachSortingValidation:
    """``Experiment._attach_sorting`` must validate inputs with clear errors."""

    def test_rejects_wrong_record_count(self):
        exp = _make_minimal_exp(ntrials=2, nelectrodes=2, max_spikes=3)
        records = [{"n_spikes": 0, "labels": np.empty(0, dtype=np.int32)}] * 3  # 3 instead of 4
        with pytest.raises(ValueError, match="record count"):
            exp._attach_sorting(records)

    def test_rejects_n_spikes_exceeding_max_spikes(self):
        exp = _make_minimal_exp(ntrials=1, nelectrodes=1, max_spikes=2)
        records = [
            {
                "n_spikes": 5,  # > max_spikes=2
                "labels": np.array([1, 1, 2, 2, 3], dtype=np.int32),
            }
        ]
        with pytest.raises(ValueError, match="max_spikes"):
            exp._attach_sorting(records)


class TestImportSortingResults:
    """``Experiment.import_sorting_results`` works without n_fields arg."""

    def test_basic_import(self):
        exp = _make_minimal_exp(ntrials=2, nelectrodes=2, max_spikes=3)
        labels = [
            np.array([1, 2], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([3, 3, 4], dtype=np.int32),
            np.array([5], dtype=np.int32),
        ]
        indices = [
            np.array([10.0, 20.0], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([30.0, 40.0, 50.0], dtype=np.float32),
            np.array([60.0], dtype=np.float32),
        ]
        result = exp.import_sorting_results(labels, indices)
        assert len(result) == 4
        # cluster_labels attached to dataset with right shape
        assert "cluster_labels" in exp.data
        assert exp.data["cluster_labels"].shape == (2, 2, 3)  # ch, trial, spikes

    def test_import_with_amp_arrays(self):
        exp = _make_minimal_exp(ntrials=1, nelectrodes=1, max_spikes=2)
        labels = [np.array([1, 2], dtype=np.int32)]
        indices = [np.array([10.0, 20.0], dtype=np.float32)]
        amp_max = [np.array([200.0, 210.0], dtype=np.float32)]
        result = exp.import_sorting_results(
            labels,
            indices,
            amp_max_per_record=amp_max,
        )
        np.testing.assert_array_almost_equal(result[0]["amp_max"], [200.0, 210.0])
        # other named columns default to zeros
        np.testing.assert_array_equal(result[0]["amp_min"], [0.0, 0.0])
