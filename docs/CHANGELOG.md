# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-05-29

### Added

- `Experiment` class early draft

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
