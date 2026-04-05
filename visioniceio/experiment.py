"""Experiment class for VisionICeIO.

Reads the metadata and the binary files from one experiment.

Wraps the data into an xarray Dataset.  Optionally stores as Zarr.
"""

import os

import numpy as np
import xarray as xr

from .io import (
    read_analog_new,
    read_behave_new,
    read_bhv,
    read_data,
    read_info_new,
    read_metadata,
    read_metadata_ifo,
    read_spike_new,
    read_ssort,
    read_stim_new,
    read_swave_new,
    write_ssort,
)


def _zarr_encoding(data_vars):
    """Build a zarr encoding dict compatible with both zarr v2 and v3.

    For zarr v2 (``zarr < 3``), uses ``numcodecs.Blosc`` and the
    ``"compressor"`` (singular) encoding key.  For zarr v3 (``zarr >= 3``),
    uses ``zarr.codecs.BloscCodec`` and the ``"compressors"`` (plural) key
    with a list value.

    Args:
        data_vars: Iterable of variable names.

    Returns:
        dict: Encoding dict suitable for ``xr.Dataset.to_zarr(encoding=...)``.
    """
    import zarr

    major = int(zarr.__version__.split(".")[0])

    if major >= 3:
        from zarr.codecs import BloscCodec

        codec = [BloscCodec(cname="zstd", clevel=1, shuffle="noshuffle")]
        key = "compressors"
    else:
        from numcodecs import Blosc

        codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.NOSHUFFLE)
        key = "compressor"

    return {var: {key: codec} for var in data_vars}


class Experiment:
    """One experiment directory in the current workflow structure.

    Supports both old DLTG files (``.swa``, ``.spi``, ``.stm``, ``.ana``)
    and new headerless LabView files (``.swave``, ``.spike``, ``.stim``,
    ``.analog``).  When both formats are present, the **new** format is
    preferred.
    """

    # Dispatch table: data_type -> (new_ext, new_reader, old_ext, old_dtype, old_ndim)
    _READERS = {
        'spike':    ('.spike',  read_spike_new,  '.spi', 'uint32', 1),
        'waveform': ('.swave',  read_swave_new,  '.swa', 'int16', 2),
        'stim':     ('.stim',   read_stim_new,   '.stm', 'int32', 1),
        'analog':   ('.analog', read_analog_new,  '.ana', 'int16', 1),
    }

    def __init__(self):
        self.path = None
        self.name = None
        self.data = None
        self.metadata = None
        self.pad_value = np.nan
        self._file_format = None  # 'new' or 'old', set during load
        self.sorting_results = None  # list[dict] | None, populated by load_ssort()

    # ------------------------------------------------------------------
    # File resolution helpers
    # ------------------------------------------------------------------

    def _resolve_file(self, new_ext: str, old_ext: str) -> tuple[str, str]:
        """Return (filepath, format) preferring new over old.

        Returns:
            Tuple of (absolute_path, 'new'|'old').

        Raises:
            FileNotFoundError: If neither file exists.
        """
        new_path = os.path.join(self.path, self.name + new_ext)
        old_path = os.path.join(self.path, self.name + old_ext)
        if os.path.exists(new_path):
            return new_path, 'new'
        if os.path.exists(old_path):
            return old_path, 'old'
        raise FileNotFoundError(
            f"Neither {new_path} nor {old_path} found."
        )

    def _read_raw(self, data_type: str) -> list | tuple | np.ndarray:
        """Read raw data for *data_type*, auto-detecting file format.

        Uses the new-format reader if the new file exists, otherwise
        falls back to the old DLTG reader.

        Args:
            data_type: One of ``'spike'``, ``'waveform'``, ``'stim'``,
                ``'analog'``.

        Returns:
            The raw data as returned by the appropriate reader.
        """
        new_ext, new_reader, old_ext, old_dtype, old_ndim = (
            self._READERS[data_type]
        )
        new_path = os.path.join(self.path, self.name + new_ext)
        old_path = os.path.join(self.path, self.name + old_ext)

        if os.path.exists(new_path):
            return new_reader(new_path)
        if os.path.exists(old_path):
            return read_data(old_path, old_dtype, old_ndim)
        raise FileNotFoundError(
            f"Neither {new_path} nor {old_path} found."
        )

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    def _load_metadata(self) -> dict:
        """Load metadata from available metadata files.

        Priority order:

        1. ``.info`` (new-format PTH0 binary)
        2. ``.ifo``  (old-format DLTG binary)
        3. ``-ifo.txt`` (plain text)

        Returns:
            Parsed metadata dictionary.

        Raises:
            FileNotFoundError: If no readable metadata file is found.
        """
        info_new = os.path.join(self.path, self.name + '.info')
        ifo_bin = os.path.join(self.path, self.name + '.ifo')
        ifo_txt = os.path.join(self.path, self.name + '-ifo.txt')

        # Try new-format .info (PTH0 container)
        if os.path.exists(info_new):
            try:
                meta = read_info_new(info_new)
                if meta:
                    return meta
            except (ValueError, OSError):
                pass  # fall through

        # Try binary .ifo (DLTG container), accept only if non-empty
        if os.path.exists(ifo_bin):
            meta = read_metadata_ifo(ifo_bin)
            if meta:
                return meta

        # Fall through to plain-text -ifo.txt
        if os.path.exists(ifo_txt):
            return read_metadata(ifo_txt)

        raise FileNotFoundError(
            f"No metadata file found. Looked for:\n"
            f"  {info_new}\n  {ifo_bin}\n  {ifo_txt}"
        )

    # ------------------------------------------------------------------
    # Behaviour loading
    # ------------------------------------------------------------------

    def _load_bhv(self) -> dict | np.ndarray | None:
        """Load behaviour data, preferring .behave over .bhv.

        Returns:
            For new format: 1-D int32 array of behaviour codes.
            For old format: dict from ``read_bhv``.
            ``None`` if no behaviour file exists.
        """
        behave_path = os.path.join(self.path, self.name + '.behave')
        if os.path.exists(behave_path):
            return read_behave_new(behave_path)
        bhv_path = os.path.join(self.path, self.name + '.bhv')
        if os.path.exists(bhv_path):
            return read_bhv(bhv_path)
        return None

    def _attach_bhv(self) -> None:
        """Load and attach behaviour data to the dataset."""
        bhv_data = self._load_bhv()
        if bhv_data is None:
            self.behaviour = None
            return

        self.behaviour = bhv_data
        if isinstance(bhv_data, np.ndarray):
            # New format: simple int array
            self.data.attrs['bhv_codes'] = bhv_data.tolist()
        elif isinstance(bhv_data, dict):
            for k, v in bhv_data.items():
                if k.startswith('_'):
                    continue
                if isinstance(v, (bool, int, float, str)):
                    self.data.attrs[f"bhv_{k}"] = v
            if 'strings' in bhv_data:
                self.data.attrs['bhv_strings'] = '\n'.join(
                    bhv_data['strings']
                )

    # ------------------------------------------------------------------
    # Padding / reshaping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_spike_times(
        spike_data: list[np.ndarray],
        max_spikes: int,
        sample_rate: float,
    ) -> np.ndarray:
        """Convert spike sample indices to seconds and pad to *max_spikes*.

        Returns:
            2-D float32 array ``(n_records, max_spikes)``.
        """
        return np.array(
            [
                np.pad(
                    s.astype(np.float32) / sample_rate,
                    (0, max_spikes - s.shape[0]),
                    'constant',
                    constant_values=np.nan,
                )
                for s in spike_data
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _pad_waveforms(
        wave_data: list[np.ndarray],
        max_spikes: int,
    ) -> np.ndarray:
        """Pad waveform arrays to *max_spikes* along axis 0.

        Returns:
            3-D float32 array ``(n_records, max_spikes, wf_pts)``.
        """
        return np.array(
            [
                np.pad(
                    da.astype(np.float32),
                    ((0, max_spikes - da.shape[0]), (0, 0)),
                    'constant',
                    constant_values=np.nan,
                )
                for da in wave_data
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _to_electrode_major(
        arr: np.ndarray,
        ntrials: int,
        nelectrodes: int,
        *extra_dims: int,
    ) -> np.ndarray:
        """Reshape from flat (trial-major) to (electrodes, trials, ...).

        Input *arr* has its first axis as ``ntrials * nelectrodes``
        records in trial-major, channel-minor order.  This reshapes
        to ``(ntrials, nelectrodes, *extra_dims)`` then transposes
        axes 0 and 1 to produce ``(nelectrodes, ntrials, *extra_dims)``.
        """
        reshaped = arr.reshape(ntrials, nelectrodes, *extra_dims)
        return np.ascontiguousarray(reshaped.swapaxes(0, 1))

    # ------------------------------------------------------------------
    # Main loading entry point
    # ------------------------------------------------------------------

    def load_from_dir(
        self,
        path=None,
        name=None,
        save_as='zarr',
        load_bhv: bool = False,
    ):
        """Load the data and metadata from the experiment directory.

        Tries new-format files (``.swave``, ``.spike``, ``.stim``,
        ``.analog``) first. Falls back to old DLTG files if the new
        ones are not found.

        Args:
            path: Path to the experiment directory.
            name: Experiment name (file prefix without suffix).
            save_as: ``'zarr'`` to persist as Zarr store, or ``None``
                to skip.
            load_bhv: If ``True``, also load the behaviour file.
        """
        self.path = path
        self.name = name
        self.metadata = self._load_metadata()
        self.sample_rate_spike = self.metadata['SpikeSamplingFrequency']
        self.sample_rate_lfp = self.metadata['AnalogSamplingFrequency']
        self.snippet_points = self.metadata['NofPointsSpikewaveform']
        self.lfp_points = self.metadata['MaxTrialLength']
        self.ntrials = self.metadata['NofTrials']
        self.nelectrodes = self.metadata['NofSpikeChannels']

        self._load_data(load_bhv)

        if save_as == 'zarr':
            encoding = _zarr_encoding(self.data.data_vars)
            output_path = os.path.join(self.path, f"{self.name}.zarr")
            self.data.to_zarr(
                output_path,
                mode="w",
                consolidated=True,
                write_empty_chunks=False,
                encoding=encoding,
            )

    # ------------------------------------------------------------------
    # Unified data loading (handles both old and new format)
    # ------------------------------------------------------------------

    def _load_data(self, load_bhv: bool) -> None:
        """Load spike, waveform, stim and analog data from either format."""

        # --- Spike times ---
        spike_data = self._read_raw('spike')
        self._n_spikes = np.array(
            [s.shape[0] for s in spike_data], dtype=np.int32
        )
        self.max_spikes = int(self._n_spikes.max())

        d_spi = self._pad_spike_times(
            spike_data, self.max_spikes, self.sample_rate_spike
        )
        d_spi = self._to_electrode_major(
            d_spi, self.ntrials, self.nelectrodes, self.max_spikes
        )

        # --- Waveforms ---
        wave_result = self._read_raw('waveform')
        if isinstance(wave_result, tuple):
            # New format: (data_list, wf_pts)
            wave_data, wf_pts = wave_result
            if wf_pts:
                self.snippet_points = wf_pts
        else:
            # Old format: list of 2-D arrays
            wave_data = wave_result

        d_wav = self._pad_waveforms(wave_data, self.max_spikes)
        d_wav = self._to_electrode_major(
            d_wav,
            self.ntrials,
            self.nelectrodes,
            self.max_spikes,
            self.snippet_points,
        )

        # --- Build core dataset ---
        coords_et = {
            'electrodes': np.arange(self.nelectrodes),
            'trials': np.arange(self.ntrials),
        }

        self.waveforms = xr.DataArray(
            data=d_wav,
            name='waveforms',
            dims=('electrodes', 'trials', 'spikes_idx', 'snippet_time'),
            coords={
                **coords_et,
                'spikes_idx': np.arange(self.max_spikes),
                'snippet_time': (
                    np.arange(self.snippet_points) / self.sample_rate_spike
                ),
            },
        )
        self.n_spikes = xr.DataArray(
            data=self._n_spikes.reshape(
                self.ntrials, self.nelectrodes
            ).T,
            name='n_spikes',
            dims=('electrodes', 'trials'),
            coords={
                'electrodes': np.arange(self.nelectrodes),
                'trials': np.arange(self.ntrials),
            },
        )
        self.spike_times = xr.DataArray(
            data=d_spi,
            name='spike_times',
            dims=('electrodes', 'trials', 'spikes_idx'),
            coords={
                **coords_et,
                'spikes_idx': np.arange(self.max_spikes),
            },
        )
        self.data = xr.Dataset(
            data_vars={
                'waveforms': self.waveforms,
                'n_spikes': self.n_spikes,
                'spike_times': self.spike_times,
            },
            attrs=self.metadata,
        )

        # --- Stimulus labels ---
        stim_result = self._read_raw('stim')
        if isinstance(stim_result, list):
            # Old format returns list of arrays; stim has one dataset
            stim_arr = np.array(stim_result[0], dtype='int32')
        else:
            stim_arr = stim_result
        self.stim_label = xr.DataArray(
            data=stim_arr,
            name='stim_label',
            dims=['trials'],
            coords={'trials': np.arange(self.ntrials)},
        )
        self.data = self.data.merge(self.stim_label.to_dataset())

        # --- Analog / LFP ---
        ana_data = self._read_raw('analog')
        d_ana = np.array(ana_data, dtype='int16').reshape(
            self.ntrials, self.nelectrodes, self.lfp_points
        )
        d_ana = np.ascontiguousarray(d_ana.swapaxes(0, 1))
        self.lfp = xr.DataArray(
            data=d_ana,
            name='lfp',
            dims=('electrodes', 'trials', 'lfp_time'),
            coords={
                **coords_et,
                'lfp_time': (
                    np.arange(self.lfp_points) / self.sample_rate_lfp
                ),
            },
        )
        self.data = self.data.merge(self.lfp.to_dataset())

        # --- Behaviour (optional) ---
        if load_bhv:
            self._attach_bhv()

    # ------------------------------------------------------------------
    # Raw spike index access
    # ------------------------------------------------------------------

    def load_spike_indices(self) -> list[np.ndarray]:
        """Load per-record spike index arrays from ``.spike`` or ``.spi`` file.

        Automatically detects the file format (new ``.spike`` vs legacy
        ``.spi``).  Records are in trial-major, channel-minor order:
        ``record_index = trial * n_electrodes + channel``.

        Returns:
            List of 1-D arrays, one per record, containing spike sample
            indices as int32.

        Raises:
            FileNotFoundError: If neither ``.spike`` nor ``.spi`` file exists.
        """
        spike_data = self._read_raw('spike')
        return [arr.astype(np.int32) for arr in spike_data]

    # ------------------------------------------------------------------
    # Spike-sorting convenience wrappers
    # ------------------------------------------------------------------

    def _attach_sorting(self, records: list[dict]) -> None:
        """Store sorting records on *self* and add ``cluster_labels``.

        Shared helper used by :meth:`load_ssort`, :meth:`save_ssort`,
        and :meth:`import_sorting_results`.

        Args:
            records: List of per-channel-trial dicts as returned by
                :func:`~visioniceio.io.sorting.read_ssort` or built by
                :meth:`import_sorting_results`.
        """
        self.sorting_results = records

        # Reshape cluster labels into (electrodes, trials, max_spikes)
        # Records are trial-major, channel-minor — same order as spike_times
        labels_flat = [
            np.pad(
                r['labels'].astype(np.float32),
                (0, self.max_spikes - r['n_spikes']),
                constant_values=np.nan,
            )
            for r in records
        ]
        labels_arr = np.array(labels_flat, dtype=np.float32)
        labels_arr = self._to_electrode_major(
            labels_arr, self.ntrials, self.nelectrodes, self.max_spikes
        )

        # Add cluster_labels to self.data
        self.data['cluster_labels'] = xr.DataArray(
            data=labels_arr,
            dims=('electrodes', 'trials', 'spikes_idx'),
            coords={
                'electrodes': self.data.electrodes,
                'trials': self.data.trials,
                'spikes_idx': self.data.spikes_idx,
            },
        )

    def load_ssort(self, filepath: str | None = None) -> list[dict]:
        """Load spike-sorting results from a ``.ssort`` file.

        Reads the binary ``.ssort`` file and stores the parsed records on
        ``self.sorting_results``.  Cluster labels are reshaped to match the
        ``spike_times`` layout and added to ``self.data`` as
        ``cluster_labels``.

        Args:
            filepath: Path to the ``.ssort`` file. If ``None``, looks
                for ``<name>.ssort`` in the experiment directory.

        Returns:
            list[dict]: Per-channel-trial records with keys
            ``channel_idx``, ``n_spikes``, ``trial_idx``,
            ``stim_condition``, ``labels``, ``spike_indices``,
            ``features``.
        """
        if filepath is None:
            filepath = os.path.join(self.path, self.name + '.ssort')
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Sorting file not found: {filepath}"
            )
        records = read_ssort(filepath)
        self._attach_sorting(records)
        return records

    def save_ssort(
        self,
        labels_per_record: list[np.ndarray],
        spike_indices_per_record: list[np.ndarray],
        features_per_record: list[np.ndarray] | None = None,
        filepath: str | None = None,
        n_fields: int = 10,
        channel_indices: list | np.ndarray | None = None,
        trial_indices: list | np.ndarray | None = None,
        stim_conditions: list | np.ndarray | None = None,
    ) -> str:
        """Write spike-sorting results to a ``.ssort`` file.

        After writing, the results are also stored on the instance
        (equivalent to calling :meth:`load_ssort` on the written file).

        Args:
            labels_per_record: List of 1-D label arrays, one per
                channel-trial record (trial-major, channel-minor order).
            spike_indices_per_record: List of 1-D arrays of spike-time
                sample indices.
            features_per_record: Optional per-spike feature arrays.
            filepath: Output path. Defaults to ``<path>/<name>.ssort``.
            n_fields: Number of columns per spike row (default 10).
            channel_indices: Optional per-record channel indices.
            trial_indices: Optional per-record trial indices.
            stim_conditions: Optional per-record stimulus-condition codes.

        Returns:
            str: The path the file was written to.
        """
        if filepath is None:
            filepath = os.path.join(self.path, self.name + '.ssort')
        write_ssort(
            filepath,
            labels_per_record,
            spike_indices_per_record,
            features_per_record,
            n_fields,
            channel_indices,
            trial_indices,
            stim_conditions,
        )
        # Re-read the written file to populate self.sorting_results
        # and self.data['cluster_labels'] (single code path via load_ssort)
        self.load_ssort(filepath)
        return filepath

    def import_sorting_results(
        self,
        labels_per_record: list[np.ndarray],
        spike_indices_per_record: list[np.ndarray],
        features_per_record: list[np.ndarray] | None = None,
        n_fields: int = 10,
        channel_indices: list | np.ndarray | None = None,
        trial_indices: list | np.ndarray | None = None,
        stim_conditions: list | np.ndarray | None = None,
    ) -> list[dict]:
        """Import sorting results directly into the Experiment (no file I/O).

        Accepts the same arrays as :meth:`save_ssort` /
        :func:`~visioniceio.io.sorting.write_ssort` but stores them on ``self`` without writing
        a ``.ssort`` file.  This is useful when sorting results are
        produced in memory and disk persistence is not (yet) needed.

        After calling this method, ``self.sorting_results`` holds the
        record list and ``self.data['cluster_labels']`` contains the
        reshaped label array.

        Args:
            labels_per_record: List of 1-D label arrays, one per
                channel-trial record (trial-major, channel-minor order).
            spike_indices_per_record: List of 1-D arrays of spike-time
                sample indices.
            features_per_record: Optional per-spike feature arrays
                ``(n_spikes, n_feat)``.  If ``None``, an empty array is
                stored for each record.
            n_fields: Number of feature columns (default 10).  Only used
                when *features_per_record* is ``None``.
            channel_indices: Optional per-record channel indices.
                Defaults to cycling ``0 .. n_electrodes-1``.
            trial_indices: Optional per-record trial indices.
                Defaults to ``floor(record_index / n_electrodes)``.
            stim_conditions: Optional per-record stimulus-condition codes.
                Defaults to 0.

        Returns:
            list[dict]: The constructed records (same structure as
            :func:`~visioniceio.io.sorting.read_ssort` output).
        """
        n_records = len(labels_per_record)

        # Resolve optional per-record metadata
        if channel_indices is None:
            channel_indices = [i % self.nelectrodes for i in range(n_records)]
        if trial_indices is None:
            trial_indices = [i // self.nelectrodes for i in range(n_records)]
        if stim_conditions is None:
            stim_conditions = [0] * n_records

        records: list[dict] = []
        for i in range(n_records):
            labels = np.asarray(labels_per_record[i])
            spike_idx = np.asarray(spike_indices_per_record[i])
            n_spikes = len(labels)

            if features_per_record is not None:
                feat = np.asarray(features_per_record[i], dtype=np.float32)
            else:
                # n_fields minus the 4 standard columns (label, index, amp, slope)
                n_feat = max(n_fields - 4, 0)
                feat = np.zeros((n_spikes, n_feat), dtype=np.float32)

            records.append({
                'channel_idx': int(channel_indices[i]),
                'n_spikes': n_spikes,
                'trial_idx': int(trial_indices[i]),
                'stim_condition': int(stim_conditions[i]),
                'labels': labels,
                'spike_indices': spike_idx,
                'features': feat,
            })

        self._attach_sorting(records)
        return records
