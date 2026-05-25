# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-05-25

### Added

- Regression tests pinning the v10 empty-record on-disk layout
  (`TestSsortV10EmptyOnDisk` in `tests/test_io_readers.py`): writer
  emits header-only, reader still accepts the legacy `[0,0]` sentinel,
  and a round-trip of mixed empty + non-empty records is byte-identical.
- Regression tests for the `load_from_zarr` v2/v3 version guard
  (`TestLoadFromZarrVersionGuard`).
- Regression test for `Experiment.import_sorting_results` with a
  per-record array containing a `None` entry (`test_import_with_partial_none_per_record_arrays`).
- Regression tests for the second-round hardening fixes
  (`TestSwaveMixedPtsRejected`, `TestReadDataDimGuards`,
  `TestReadMetadataIfoTruncated`, `TestReadInfoNewBogusRecord2Size`,
  `TestExperimentInit`).
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

- `docs/data_format.md` no longer states that empty `.ssort` records
  are `[0, 0]` "in both variants"; the section now documents v10's
  header-only convention and v16's `[0, 0]` sentinel separately, with
  a note that the reader still accepts the legacy `[0, 0]` form for
  v10 back-compat.
- `docs/data_format.md` `features` row in the returned-record table no
  longer hard-codes `cols 6–9 (4 feats)` / `cols 10–15 (6 feats)`.
  The text now states that `features` covers the trailing columns
  after the named fields and qualifies the column ranges with the
  default `n_fields=10` for Variant A — `read_ssort()` actually
  returns `rows[:, 6:]` (or `rows[:, 10:]`) open-ended, and the old
  doc would have silently lied for a non-default `n_fields`.
- Demo-notebook dependencies (`matplotlib`, `ipywidgets`) moved from
  the orphaned `requirements.txt` into a new ``notebook`` extras in
  ``pyproject.toml``; ``requirements.txt`` removed.  Install with
  ``pip install -e ".[notebook]"``.
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

- `write_ssort()` now emits the **header-only** form (`n_entries=1`,
  48 bytes at `n_fields=10`) for empty channel-trial records in
  Variant A — matching what the lab sorter actually writes.  The
  previous implementation collapsed empty v10 records to the 8-byte
  `[0, 0]` sentinel, which preserved spike data but shrank files by
  `40 × n_empty` bytes and produced output that didn't byte-match
  lab fixtures.  Verified on the corpus: 16/16 real v10 `.ssort`
  files (including `c5607a07.ssort` and `c5102a08_o.ssort`) now
  round-trip byte-identically.  The reader still accepts the legacy
  `[0, 0]` form for back-compat with any previously-written file.
- The module docstring at the top of `visioniceio/io/sorting.py`
  no longer claims `[0, 0]` is used "in both variants" — only v16
  uses that sentinel; v10 uses header-only.  This had drifted from
  the empirical reality of every real-world `.ssort` file in the
  corpus.
- `load_from_zarr()` now raises a clear `ValueError` (mentioning the
  detected store version and the installed zarr version) when given
  a zarr v3 store while the environment has zarr v2 installed.
  Previously, xarray's open path eventually surfaced
  `FileNotFoundError: No such file or directory: '<path>'` against
  a directory that obviously existed — wrong cause, wrong remedy.
  Detection looks for a top-level `zarr.json` without a
  `.zmetadata`, the unambiguous v3 marker.
- `Experiment.import_sorting_results._per_record` no longer relies
  on a closure-captured `i` from the enclosing loop; the record
  index is now passed explicitly.  Same behaviour, less fragile.
- `read_bhv()` hoists the `np.sort(offsets)` call out of the
  per-record loop in the raw-block fallback path (was O(n²);
  unreachable for the lab corpus, but the hoist is free).
- `Experiment.__init__` no longer writes `_file_format` or
  `pad_value` attributes that were never read elsewhere — dead
  state removed.
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
- `read_metadata_ifo()` now also catches `EOFError` from the DLTG
  parse attempt before falling back to the plain-text reader.  The
  docstring promised a text fallback "if the DLTG container cannot be
  parsed", but a truncated `.ifo` raised `EOFError` from
  `_read_dltg_header` and bypassed it.
- `read_swave_new()` now raises a clear `ValueError` when waveform
  records disagree on the snippet length (`pts`) instead of returning
  a stale first-record `wf_pts` and letting
  `Experiment._pad_waveforms` fail later with NumPy's confusing
  "inhomogeneous shape" error.
- `read_info_new()` now validates the declared second-PTH0
  `record2_size` against the bytes remaining in the file, raising a
  clear `ValueError` if the size would push the parser past EOF.
  A corrupt or non-PTH0 file would previously advance the read
  cursor arbitrarily and surface a less-clear EOFError from a
  downstream LV-string read.
- `read_data()` now rejects `nd < 1` (API-level guard) and rejects
  per-record dim headers containing **negative** values (data-level
  guard).  A `dim == 0` is still accepted — legacy `.swa` empty
  channel-trials are legitimately stored as shape `(0, n_pts)`.
- `Experiment.__init__` now initialises `self.behaviour = None`
  alongside `self.sorting_results`.  The attribute was only ever set
  inside `_attach_bhv()`, so `hasattr(exp, "behaviour")` was `False`
  on the default `load_bhv=False` path.

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
