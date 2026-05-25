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

### Fixed

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

- DLTG offset-table chaining for files with more than 128 datasets now
  correctly follows chain pointers at entry 127 of each block.
- DLTG header parsing uses `_read_exact()` throughout, giving clear
  `EOFError` messages on truncated files instead of cryptic `struct.error`.
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
