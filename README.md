# VisionICeIO

I/O utilities for the [ICe Vision Lab](https://github.com/VisionICeNatal)
(Natal, Brazil) LabView recording system.

VisionICeIO reads the binary files produced by the lab's LabView acquisition
software, wraps all data channels into a single
[xarray](https://docs.xarray.dev/) `Dataset`, and optionally persists the
result as a compressed [Zarr](https://zarr.dev/) store for fast reloading.

## Features

- **Two file generations** -- reads both old DLTG-header files and new
  headerless LabView binaries; auto-detects which format is present.
- **Spike data** -- spike timestamps, waveform snippets, spike counts.
- **Continuous data** -- LFP / analog traces, stimulus labels.
- **Metadata** -- plain-text (`-ifo.txt`), binary DLTG (`.ifo`), and
  PTH0 (`.info`) metadata files.
- **Behaviour** -- old `.bhv` and new `.behave` files.
- **Spike sorting I/O** -- read / write `.ssort` files; attach cluster
  labels to the dataset from file or directly from memory.
- **xarray integration** -- all channels in one `xr.Dataset` with named
  dimensions and physical coordinates (seconds).
- **Zarr persistence** -- save as a compressed Zarr store; reload later
  without re-parsing the raw binaries.

## Installation

```bash
pip install visioniceio
```

For development (editable install with extras):

```bash
git clone https://github.com/VisionICeNatal/VisionICeIO.git
cd VisionICeIO
pip install -e ".[dev,test,docs]"
```

To run the example notebooks (`examples/real_data_exploration.ipynb`),
also add the `notebook` extra:

```bash
pip install -e ".[dev,test,docs,notebook]"
```

### Requirements

Python >= 3.10, plus `numpy`, `xarray`, `zarr`, and `numcodecs`.
Both Zarr v2 and v3 are supported.

## Quick Start

### Load an experiment

```python
from visioniceio import Experiment

exp = Experiment()
exp.load_from_dir(path="/path/to/data", name="c5607a07")

ds = exp.data          # xr.Dataset
ds
```

```
<xarray.Dataset>
Dimensions:    (electrodes: 64, trials: 240, spikes_idx: 312,
                snippet_time: 48, lfp_time: 50000)
Data variables:
    waveforms   (electrodes, trials, spikes_idx, snippet_time)  float32
    spike_times (electrodes, trials, spikes_idx)                 float32
    n_spikes    (electrodes, trials)                              int32
    stim_label  (trials)                                          int32
    lfp         (electrodes, trials, lfp_time)                    int16
```

### Work with the data

```python
# Waveforms for electrode 0, trial 5
wf = ds.waveforms.sel(electrodes=0, trials=5)

# All spike times for one electrode
st = ds.spike_times.sel(electrodes=0)

# LFP filtered by stimulus condition
lfp_cond1 = ds.lfp.sel(trials=ds.stim_label == 1)
```

### Reload from Zarr

When `save_as='zarr'` (the default), the dataset is persisted as a
compressed Zarr store next to the raw files:

```python
from visioniceio import load_from_zarr

ds = load_from_zarr("/path/to/data/c5607a07.zarr")
```

### Spike sorting results

`.ssort` files exist in two variants in the lab's recordings (10-column
"header + spike rows" and 16-column "no-header" layouts).
`Experiment.load_ssort()` auto-detects the variant; `save_ssort()`
selects it via the ``n_fields`` argument (10 or 16).  See the
[data format spec](https://VisionICeNatal.github.io/VisionICeIO/data_format.html#ssort-spike-sorting-results)
for the column tables.

```python
# Load from a .ssort file (variant auto-detected;
# stores on exp.data['cluster_labels'])
records = exp.load_ssort()

# -- or -- import directly from memory (no file needed)
records = exp.import_sorting_results(
    labels_per_record=labels,
    spike_indices_per_record=spike_idx,
)

# -- or -- write to disk, then attach
filepath = exp.save_ssort(
    labels_per_record=labels,
    spike_indices_per_record=spike_idx,
    # n_fields=16 to write the no-header variant
)

# Cluster labels are now part of the dataset
exp.data['cluster_labels']  # (electrodes, trials, spikes_idx) float32
```

## Binary Formats

Two file generations are supported:

| Format | Extensions | Header |
|--------|-----------|--------|
| Old (DLTG) | `.swa`, `.spi`, `.stm`, `.ana`, `.ifo`, `.bhv` | 4-byte `DTLG` magic + offset table |
| New (headerless) | `.swave`, `.spike`, `.stim`, `.analog`, `.behave`, `.info` | None (sequential records) |
| Sorting | `.ssort` | Per-record count + field table |

`Experiment.load_from_dir()` checks for each data type's new-format file
first (e.g. `.spike` before `.spi`). Each data type resolves independently,
so a mix of old and new files is supported.

See the [data format specification](https://VisionICeNatal.github.io/VisionICeIO/data_format.html)
for the full binary layout.

## API Quick Reference

### High-level (`Experiment`)

| Method | Description |
|--------|-------------|
| `Experiment.load_from_dir()` | Load all channels into `xr.Dataset` (optionally save as Zarr) |
| `Experiment.load_ssort()` | Read `.ssort` file, attach to dataset |
| `Experiment.save_ssort()` | Write `.ssort` file, attach to dataset |
| `Experiment.import_sorting_results()` | Attach sorting results from memory (no file I/O) |

### Low-level readers (`core_io`)

| Function | Description |
|----------|-------------|
| `read_spike_new()` | New-format `.spike` file |
| `read_swave_new()` | New-format `.swave` file |
| `read_stim_new()` | New-format `.stim` file |
| `read_behave_new()` | New-format `.behave` file |
| `read_analog_new()` | New-format `.analog` file |
| `read_data()` | Old DLTG container (`.swa`, `.spi`, `.stm`, `.ana`) |
| `read_metadata()` | Plain-text `-ifo.txt` |
| `read_metadata_ifo()` | Binary DLTG `.ifo` |
| `read_info_new()` | Binary PTH0 `.info` |
| `read_bhv()` | Old DLTG `.bhv` behaviour file |
| `read_ssort()` / `write_ssort()` | Low-level `.ssort` I/O |
| `load_from_zarr()` | Reopen a saved Zarr store |

## Examples

Two Jupyter notebooks are included under `examples/`:

- **`example_data_loading.ipynb`** -- exercises every public function with
  synthetic data (no real data files needed).
- **`real_data_exploration.ipynb`** -- template for exploring a real
  experiment directory.

## Documentation

Full API reference, developer guide, and binary format specification:

[https://VisionICeNatal.github.io/VisionICeIO/](https://VisionICeNatal.github.io/VisionICeIO/)

## License

AGPL-3.0-only
