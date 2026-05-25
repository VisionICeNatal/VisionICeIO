# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-05-25

### Added

- `.ssort` reader now auto-detects two binary variants observed in
  real lab files:
  - Variant A (`n_fields = 10`) — header row followed by `n_spikes`
    spike rows.
  - Variant B (`n_fields = 16`) — no header row; every row is a spike
    with channel / trial / stim stamped redundantly per row.
- `read_ssort()` now returns structured per-spike arrays
  (`labels`, `spike_indices`, `amp_max`, `amp_min`, `peak_to_peak`,
  `width`, `features`) instead of an opaque feature blob.  Each record
  also carries a `variant` key (`'v10'` or `'v16'`).
- `write_ssort()` and `Experiment.save_ssort()` gained optional
  `amp_max_per_record`, `amp_min_per_record`,
  `peak_to_peak_per_record`, `width_per_record` parameters so that
  Variant A / Variant B roundtrips preserve every column the sorter
  produces.  Pass `n_fields=16` to write Variant B.
- Cross-format equivalence test for `.swa` (old DLTG) vs. `.swave`
  (new headerless) waveform readers in
  `tests/test_io_readers.py::TestSwaSwaveEquivalence`.  Empirically
  verified on the lab's only paired dataset, `c5607a07_n`: both readers
  return byte-identical NumPy arrays across 7,680 records / 1.47 M
  spikes (≈ 55.8 M int16 samples).  The synthetic regression test
  exercises the same property in CI without requiring lab fixtures.

### Changed

- `read_stim_new()` and `read_behave_new()` now share a single
  implementation, `_read_u4_count_prefixed_as_i4()`, in
  `visioniceio.io._helpers`.  The two readers were byte-identical
  apart from one word in the error message; deduping removes the
  "fix one and forget the other" hazard.  Side benefit: the old
  ``fsize - 4`` remaining-bytes computation (which silently assumed
  the file pointer was at byte 4) is replaced with the more robust
  ``fsize - f.tell()`` pattern already used by the other new-format
  readers.
- `Experiment._read_raw('waveform')` now always returns
  ``(records, wf_pts)`` regardless of file format — the old-format
  (``.swa``) branch derives ``wf_pts`` from ``records[0].shape[1]``.
  Callers no longer need to ``isinstance(..., tuple)``-dispatch on the
  return value, and ``snippet_points`` now reflects the actual data
  shape on disk (not just the metadata field) even for old-format
  experiments.

### Fixed

- DLTG container reader (`_read_dltg_header` + `read_data`) now
  correctly handles the **two-level offset table** used when
  `ndim > 128`.  The previous implementation assumed a (non-existent)
  chain pointer at entry 127 of each 128-entry block; in practice the
  format dispatches by `ndim`:

  - `ndim ≤ 128` — main-table entries are direct record offsets.
  - `ndim > 128` — main-table entries are pointers to per-chunk
    sub-tables, each holding 128 absolute record offsets.

  Symptom of the old bug: `Experiment.load_from_dir` on any older
  experiment lacking a new-format `.analog` / `.spike` / `.stim`
  file (≈ 78% of the lab's archived recordings) crashed with
  `EOFError: Unexpected end of file: wanted 512 bytes, got 0` the
  moment it tried to read the `.ana` / `.spi` / `.swa` companion.
  Verified byte-identical to the new-format readers across
  ≈ 100,000 records / 23 lab datasets.
- `read_data()` size validation now compares `nbytes` against the
  bytes remaining from the current file position (`fsize - f.tell()`)
  instead of the absolute file size, producing a meaningful error
  message on truncated records.
- `read_data()` and `_read_dltg_header()` now raise a clear
  `ValueError` when `ndim` exceeds the two-level addressing capacity
  of 16,384.  The message names the file (when the handle was opened
  from a path) and hints that the cause is likely a newer DLTG variant
  (e.g. third-level chaining or a different table format), so the next
  step is to extend the reader rather than chase a data-corruption red
  herring.  The exact boundary value `ndim == 16,384` is accepted; only
  values strictly greater raise.  All three behaviours are pinned by
  tests in `TestDLTGTwoLevelMode`.
- `read_ssort()` no longer crashes on empty channel-trial records
  (8-byte `[0, 0]` sentinels) that are common in real Variant B files
  — the old code raised `IndexError` on the first such record.
- `read_ssort()` correctly parses Variant B files; previously it
  misinterpreted col 0 (channel index) as a cluster label and col 1
  (cluster label) as a spike count, producing nonsense data.
- `Experiment.import_sorting_results()` no longer mis-allocates the
  feature column count via `n_fields - 4`; features are now whatever
  the caller supplies (or an empty `(n_spikes, 0)` array).  The
  signature drops the redundant `n_fields` argument.
- `Experiment._attach_sorting()` validates that the sorting record
  count matches `ntrials * nelectrodes` and that no record claims more
  spikes than `max_spikes`, with clear error messages instead of
  cryptic NumPy failures.

## [0.1.0] - 2026-03-08

### Added

- `Experiment` class with `load_from_dir()` for loading a full experiment
  directory into an `xr.Dataset`.
- Auto-detection of old (DLTG) vs. new (headerless) file format per data type.
- Old-format DLTG readers: `read_data()` for `.swa`, `.spi`, `.stm`, `.ana`.
- New-format headerless readers: `read_spike_new()`, `read_swave_new()`,
  `read_stim_new()`, `read_behave_new()`, `read_analog_new()`.
- Metadata readers for all three variants:
  - `read_metadata()` for plain-text `-ifo.txt` files.
  - `read_metadata_ifo()` for binary `.ifo` DLTG files.
  - `read_info_new()` for binary `.info` PTH0 files.
- Behaviour file readers: `read_bhv()` (old `.bhv` DLTG) and
  `read_behave_new()` (new `.behave` headerless).
- Spike-sorting I/O: `read_ssort()` and `write_ssort()` for the `.ssort`
  binary format, plus convenience wrappers `Experiment.load_ssort()` and
  `Experiment.save_ssort()`.
- `Experiment.import_sorting_results()` for attaching sorting results
  directly from memory (same API as `save_ssort()`, no file I/O).
- `Experiment.load_ssort()` and `Experiment.save_ssort()` now store
  results on `self.sorting_results` and add a `cluster_labels` DataArray
  (electrodes × trials × spikes_idx) to `Experiment.data`.
- Zarr persistence via `Experiment.load_from_dir(..., save_as='zarr')` and
  `load_from_zarr()` for reloading.
- Sphinx documentation with `pydata-sphinx-theme`, covering API reference,
  developer guide, and binary format specification.
- `myst-parser` extension for including Markdown pages (`data_format.md`)
  in the Sphinx toctree.
- `make serve` target in the docs Makefile for local preview.
- Comprehensive example notebook (`examples/example_data_loading.ipynb`)
  exercising every public function with synthetic data.
- Single-source versioning: `__version__` and Sphinx `release` are both
  derived from `pyproject.toml`.

### Fixed

- DLTG header parsing uses `_read_exact()` throughout, giving clear
  `EOFError` messages on truncated files instead of cryptic `struct.error`.
- DLTG offset-table reading for files with more than 128 datasets was
  re-worked to follow what was *believed* at the time to be a chain
  pointer at entry 127 of each block.  This understanding turned out
  to be wrong; the real format uses a two-level addressing scheme and
  the actual fix is documented under `[Unreleased]`.
- Zarr encoding now auto-detects the installed zarr version and uses the
  correct API: ``"compressor"`` key with ``numcodecs.Blosc`` for zarr v2,
  ``"compressors"`` key with ``zarr.codecs.BloscCodec`` for zarr v3.
  Previously, using the v2 compressor on a zarr v3 environment raised
  ``TypeError: Expected a BytesBytesCodec``, and using the string
  ``shuffle='noshuffle'`` with ``numcodecs.Blosc`` caused a repr error.
- Minimum ``zarr`` dependency bumped from ``>=2.12`` to ``>=2.16`` to
  ensure compatibility with NumPy 2.x (``np.product`` was removed in
  NumPy 2.0; older zarr versions depend on it).
- `read_bhv()` sorts the offset array before calling `np.searchsorted`,
  preventing incorrect block-boundary lookups on unsorted offset tables.
