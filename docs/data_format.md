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

After the header, a table of 128 × 4-byte offsets points to individual data
blocks. For files with more than 128 blocks, offsets are chained recursively.

Each data block starts with dimension sizes (big-endian int32, one per
dimension), followed by the raw data in row-major (C) order.

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

The `.ssort` file stores spike sorting labels and waveform features aligned
to the unsorted spike train. Like the data files, records are
**trial-major, channel-minor**.

### Binary layout

Each record:

```
[n_entries : uint32_BE]  [n_fields : uint32_BE]
[n_entries × n_fields × float32_BE]
```

`n_entries = 1 (header) + n_spikes`. `n_fields` is typically 10.

**Header row** (first entry):

| Column | Meaning |
|--------|---------|
| 0      | Channel index (0-based within the trial) |
| 1      | Actual spike count for this channel-trial |
| 2      | Trial index |
| 3      | Stimulus condition code |
| 4–9    | Reserved (0) |

**Spike rows** (entries 1 … n_spikes):

| Column | Meaning |
|--------|---------|
| 0      | Cluster label (int, stored as float32) |
| 1      | Spike time sample index (at spike sampling rate) |
| 2      | Amplitude |
| 3      | Slope |
| 4–6    | PCA components (PCA1, PCA2, PCA3) |
| 7–9    | Reserved; zero if unused |

### Reading and writing

- `read_ssort(filepath)` → list of dicts with keys `channel_idx`,
  `n_spikes`, `trial_idx`, `stim_condition`, `labels`, `spike_indices`,
  `features`.
- `write_ssort(filepath, labels, spike_indices, features, n_fields, ...)` writes
  the matching binary format.  Accepts optional `channel_indices`,
  `trial_indices`, and `stim_conditions` arrays.
- `Experiment.load_ssort()` / `Experiment.save_ssort()` provide convenience
  wrappers that resolve paths relative to the experiment directory.
- `Experiment.import_sorting_results()` accepts the same arrays as
  `save_ssort()` but stores them on the instance without writing to disk.

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
