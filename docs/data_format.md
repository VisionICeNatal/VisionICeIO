# Data Format Specification

## Overview

The recording system writes multiple binary files per experiment.
`VisionICeIO` reads these files, wraps them into an xarray Dataset, and
optionally persists the result as a zarr store.

Two binary file generations are supported: the original DLTG-header format
and the newer headerless format. The `Experiment` class tries the new format
first and falls back to the old format automatically.

### Data Flow

```{mermaid}
flowchart LR
    subgraph files ["Binary Files"]
        direction TB
        F1[".spike / .spi"]
        F2[".swave / .swa"]
        F3[".stim / .stm"]
        F4[".analog / .ana"]
        F5[".info / .ifo / -ifo.txt"]
    end

    subgraph iopkg ["io subpackage"]
        direction TB
        R1["spike: read_spike_new<br/>read_data"]
        R2["waveform: read_swave_new<br/>read_data"]
        R3["stim: read_stim_new<br/>read_data"]
        R4["analog: read_analog_new<br/>read_data"]
        R5["metadata: read_info_new<br/>read_metadata_ifo<br/>read_metadata"]
    end

    EXP["Experiment<br/>load_from_dir()"]
    DS["xr.Dataset"]
    ZARR[".zarr store"]

    F1 --> R1 --> EXP
    F2 --> R2 --> EXP
    F3 --> R3 --> EXP
    F4 --> R4 --> EXP
    F5 --> R5 --> EXP
    EXP --> DS --> ZARR
```

## Old Format — DLTG Container (LabView)

All four data files share the DTLG container format:

| Offset | Size | Content |
|--------|------|---------|
| 0      | 4 B  | Magic: `DTLG` (ASCII) |
| 4      | 4 B  | Version bytes |
| 8      | 4 B  | `ndim`: number of data blocks (big-endian uint32) |
| 12     | 4 B  | `p`: byte offset to the block-offset table |
| 16     | 2 B  | `ld`: length of the descriptor string |
| 18     | ld B | Descriptor (ASCII) |

After the header, byte `p` holds a fixed-size **128 × 4-byte offset
table**.  This table is dispatched by `ndim`:

- **Direct mode** (`ndim ≤ 128`).  Entries `[0..ndim-1]` are absolute
  byte offsets to records; remaining slots are zero-filled.
- **Two-level mode** (`ndim > 128`).  Entries `[0..n_chunks-1]`
  (where `n_chunks = ceil(ndim / 128)`) are absolute byte offsets to
  per-chunk **sub-tables**.  Each sub-table is itself a 128 × u32 BE
  block of absolute record offsets, with unused trailing slots
  zero-filled.  Record `i` is found at
  `sub_tables[i // 128][i % 128]`.  Maximum supported `ndim` is
  `128 × 128 = 16384`.

Each data block (record) starts with `nd` dimension sizes
(big-endian int32, where `nd` is the per-file dimensionality given
in the table below), followed by `prod(dims) × itemsize` bytes of
raw data in row-major (C) order.

### Old format file types

| Suffix | Content | Raw dtype | Block dimensions |
|--------|---------|-----------|-----------------|
| `.swa` | Waveform snippets | int16 | (n_spikes, snippet_points) per electrode×trial |
| `.spi` | Spike sample indices | uint32 | (n_spikes,) per electrode×trial |
| `.stm` | Stimulus labels | int32 | (n_trials,) — single block |
| `.ana` | Analog / LFP traces | int16 | (lfp_points,) per electrode×trial |

The metadata file (`*-ifo.txt`) provides key-value pairs including
`SpikeSamplingFrequency`, `NofTrials`, `NofSpikeChannels`,
`NofPointsSpikewaveform`, `MaxTrialLength`, etc.

## New Format — Headerless LabView Binary

The newer LabView export omits the DLTG header. All values are big-endian.
Records are stored in **trial-major, channel-minor** order: trial 0 channel 0,
trial 0 channel 1, …, trial 0 channel N-1, trial 1 channel 0, etc.,
giving `n_trials × n_channels` records per file.

### New format file types

| New suffix | Old equivalent | Structure per record |
|------------|---------------|---------------------|
| `.spike`   | `.spi`        | `[count:u32] [count × u32 spike_indices]` |
| `.swave`   | `.swa`        | `[count:u32] [wf_pts:u32] [count × wf_pts × i16 samples]` |
| `.stim`    | `.stm`        | `[n_trials:u32] [n_trials × u32 ordinals]` — single record |
| `.behave`  | `.bhv`        | `[n_trials:u32] [n_trials × u32 codes]` — single record |
| `.analog`  | `.ana`        | `[count:u32] [count × i16 samples]` |
| `.info`    | `.ifo`        | Binary metadata (LabView variant record) |

### Format preference

`Experiment.load_from_dir()` checks for each data type's new-format file
first (e.g. `.spike` before `.spi`).  Each data type resolves
independently via `_read_raw()`, so a mix of old and new files is
supported (though unusual in practice).  Metadata resolution order:
`.info` (PTH0) → `.ifo` (DLTG) → `-ifo.txt` (plain text).

```{mermaid}
flowchart TD
    START["load_from_dir(path, name)"] --> META

    subgraph META ["Metadata Resolution"]
        M1{".info exists?"}
        M2{".ifo exists?"}
        M3{"-ifo.txt exists?"}
        M4["read_info_new"]
        M5["read_metadata_ifo"]
        M6["read_metadata"]
        M7["FileNotFoundError"]

        M1 -->|Yes| M4
        M1 -->|No| M2
        M2 -->|Yes| M5
        M2 -->|No| M3
        M3 -->|Yes| M6
        M3 -->|No| M7
    end

    META --> DATA

    subgraph DATA ["Per Data-Type Resolution"]
        D1{"new file exists?<br/>(.spike, .swave, ...)"}
        D2["new-format reader"]
        D3{"old file exists?<br/>(.spi, .swa, ...)"}
        D4["read_data (DLTG)"]
        D5["FileNotFoundError"]

        D1 -->|Yes| D2
        D1 -->|No| D3
        D3 -->|Yes| D4
        D3 -->|No| D5
    end

    DATA --> DS["Build xr.Dataset"]
    DS --> SAVE{"save_as='zarr'?"}
    SAVE -->|Yes| ZARR[".zarr store"]
    SAVE -->|No| DONE["Done"]
```

## .ssort — Spike Sorting Results

> **Note:** VisionICeIO provides read/write support for the `.ssort`
> format. The sorting computation itself is performed by an external tool,
> which passes its results to `write_ssort()` or `Experiment.save_ssort()`.
> Results can also be imported directly into an `Experiment` instance
> (without file I/O) via `Experiment.import_sorting_results()`.
> After loading or importing, cluster labels are available as
> `Experiment.data['cluster_labels']`.

The `.ssort` file stores spike sorting labels and waveform features
aligned to the unsorted spike train.  Records are **trial-major,
channel-minor** (1:1 aligned with `.spike` / `.swave`).

Two binary variants are observed in real recordings, distinguished by
the per-record `n_fields` value.  `read_ssort()` auto-detects the
variant from the first non-empty record; `write_ssort(..., n_fields=...)`
selects the variant on write.

### Common framing

Every variant uses the same record framing:

```
[n_entries : uint32_BE]  [n_fields : uint32_BE]
[n_entries × n_fields × float32_BE]
```

**Empty channel-trial records** (no spikes detected on that
channel-trial) use different on-disk representations in the two
variants, matching what the lab sorter writes:

- **Variant A (`n_fields = 10`)**: header-only — `n_entries = 1`, the
  header row stamps `channel_idx`/`trial_idx`/`stim_condition` so
  positional metadata survives on disk (`n_spikes` in the header is
  `0`).  At `n_fields = 10` this is `8 + 40 = 48` bytes.
- **Variant B (`n_fields = 16`)**: sentinel — `n_entries = 0,
  n_fields = 0`, no payload (8 bytes total).  Channel/trial/stim
  metadata is unrecoverable from an empty v16 record on disk;
  callers re-derive it from the record's flat index.

For back-compat, `read_ssort()` also accepts the `[0, 0]` sentinel
form for Variant A (used by any file written by a pre-2026-05 release
of `write_ssort`).

### Variant A — `n_fields = 10` (header row + spike rows)

`n_entries = 1 (header) + n_spikes`.

**Header row** (entry 0):

| Column | Meaning |
|--------|---------|
| 0      | Channel index |
| 1      | Spike count for this channel-trial |
| 2      | Trial index |
| 3      | Stimulus condition |
| 4–9    | Reserved (0) |

**Spike rows** (entries 1 … n_spikes):

| Column | Meaning |
|--------|---------|
| 0      | Cluster label (int, stored as float32) |
| 1      | Spike sample index (at spike sampling rate) |
| 2      | Amplitude max |
| 3      | Amplitude min |
| 4      | Peak-to-peak (= col 2 − col 3) |
| 5      | Width |
| 6      | Extra scalar feature (slope or similar) |
| 7–9    | PCA1, PCA2, PCA3 |

### Variant B — `n_fields = 16` (no header, redundant per-row metadata)

`n_entries = n_spikes`.  There is **no header row** — every row is a
spike.  Channel / trial / stimulus are stamped redundantly on every
row (constant within a record).

| Column | Meaning |
|--------|---------|
| 0      | Channel index (constant within record) |
| 1      | Cluster label |
| 2      | Trial index (constant within record) |
| 3      | Spike sample index |
| 4      | Stimulus condition (constant within record) |
| 5      | Reserved (always 0) |
| 6      | Amplitude max |
| 7      | Amplitude min |
| 8      | Peak-to-peak (= col 6 − col 7) |
| 9      | Width |
| 10     | Extra scalar feature |
| 11–15  | PCA1..PCA5 |

### Returned record structure (both variants)

`read_ssort()` returns a list of dicts, one per channel-trial, with
the same keys regardless of variant:

| Key | Type | Description |
|-----|------|-------------|
| `channel_idx`     | int           | 0-based channel index |
| `n_spikes`        | int           | spike count (0 for empty records) |
| `trial_idx`       | int           | 0-based trial index |
| `stim_condition`  | int           | stim condition code |
| `variant`         | str           | `'v10'` or `'v16'` |
| `labels`          | ndarray int32 | cluster labels, shape `(n_spikes,)` |
| `spike_indices`   | ndarray float32 | sample indices, shape `(n_spikes,)` |
| `amp_max`         | ndarray float32 | shape `(n_spikes,)` |
| `amp_min`         | ndarray float32 | shape `(n_spikes,)` |
| `peak_to_peak`    | ndarray float32 | shape `(n_spikes,)` |
| `width`           | ndarray float32 | shape `(n_spikes,)` |
| `features`        | ndarray float32 | shape `(n_spikes, n_feat)`; trailing columns after the named fields — v10: `cols 6..n_fields-1` (4 feats at default `n_fields=10`); v16: `cols 10..15` (6 feats; `n_fields` is always 16 for v16) |

### Reading and writing

- `read_ssort(filepath)` → auto-detects variant and returns a list of
  the dicts described above.
- `write_ssort(filepath, labels, spike_indices, ..., n_fields=10)` →
  writes Variant A by default.  Pass `n_fields=16` for Variant B.
  Accepts optional `amp_max_per_record`, `amp_min_per_record`,
  `peak_to_peak_per_record`, `width_per_record`, `features_per_record`,
  `channel_indices`, `trial_indices`, `stim_conditions`.
- `Experiment.load_ssort()` / `Experiment.save_ssort()` provide
  convenience wrappers that resolve paths relative to the experiment
  directory and validate the record count / max-spikes invariants.
- `Experiment.import_sorting_results()` accepts the same arrays as
  `save_ssort()` but stores them on the instance without writing to
  disk.

## XArray Structure

```{mermaid}
graph TD
    DS["<b>xr.Dataset</b><br>Experiment.data"]

    DS --> W["<b>waveforms</b><br><i>electrodes × trials × spikes_idx × snippet_time</i><br>float32"]
    DS --> ST["<b>spike_times</b><br><i>electrodes × trials × spikes_idx</i><br>float32"]
    DS --> NS["<b>n_spikes</b><br><i>electrodes × trials</i><br>int32"]
    DS --> SL["<b>stim_label</b><br><i>trials</i><br>int32"]
    DS --> LFP["<b>lfp</b><br><i>electrodes × trials × lfp_time</i><br>int16"]
    DS --> CL["<b>cluster_labels</b> (optional)<br><i>electrodes × trials × spikes_idx</i><br>float32"]
    DS --> ATT["<b>attrs</b><br>metadata dict"]

    style DS fill:#4a90d9,stroke:#2c5f8a,color:#fff
    style W fill:#f5f5f5,stroke:#999
    style ST fill:#f5f5f5,stroke:#999
    style NS fill:#f5f5f5,stroke:#999
    style SL fill:#f5f5f5,stroke:#999
    style LFP fill:#f5f5f5,stroke:#999
    style CL fill:#fff3cd,stroke:#856404
    style ATT fill:#e8e8e8,stroke:#999
```

After loading, `Experiment.data` is an `xr.Dataset` with:

```
Dimensions:
  electrodes:  (n_electrodes,)      e.g. 64
  trials:      (n_trials,)          e.g. 240
  spikes_idx:  (max_spikes,)        padded to the maximum spike count
  snippet_time:(snippet_points,)    e.g. 48 sample points
  lfp_time:    (lfp_points,)        e.g. 50000 sample points

Data variables:
  waveforms       (electrodes, trials, spikes_idx, snippet_time)  float32
  spike_times     (electrodes, trials, spikes_idx)                float32
  n_spikes        (electrodes, trials)                            int32
  stim_label      (trials,)                                       int32
  lfp             (electrodes, trials, lfp_time)                  int16
  cluster_labels  (electrodes, trials, spikes_idx)                float32  # after load_ssort()

Coordinates:
  electrodes:   0 .. n_electrodes-1
  trials:       0 .. n_trials-1
  spikes_idx:   0 .. max_spikes-1   (simple enumeration index)
  snippet_time: 0/fs .. (snippet_points-1)/fs  (seconds)
  lfp_time:     0/fs_lfp .. (lfp_points-1)/fs_lfp  (seconds)
```

### NaN Padding

Spike counts vary between (electrode, trial) pairs.  The `spikes_idx`
dimension is sized to the global maximum across the entire experiment.
Trials with fewer spikes are filled with `NaN` in both `waveforms` and
`spike_times`.  This requires float storage (hence float32, not int16).

### Spike time conversion

Raw spike values are sample indices (uint32).  During loading they are
divided by `SpikeSamplingFrequency` to produce times in seconds (float32).
These are trial-relative (0 = trial start).

### Waveform dtype

Raw `.swa` values are int16.  They are cast to float32 during loading.
This preserves all 16-bit precision (float32 has ~7 decimal digits of
mantissa vs ~5 for int16) while allowing NaN padding.

## Zarr Persistence

`Experiment.load_from_dir(..., save_as='zarr')` writes the dataset to a
zarr store using Blosc (zstd, level 1) compression.  The store can later
be reopened with `load_from_zarr(path)` → `xr.Dataset`, avoiding the
need to re-parse binary files.
