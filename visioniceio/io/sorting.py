"""Spike-sorting results I/O -- ``.ssort`` read and write.

The ``.ssort`` binary format stores cluster labels and waveform features
aligned to the unsorted spike train.  Records are trial-major,
channel-minor.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

from ._helpers import _read_exact


def read_ssort(filepath: str | Path) -> list[dict]:
    """Read a ``.ssort`` spike-sorting results file.

    Structure per channel-trial record::

        [n_entries : uint32_BE] [n_fields : uint32_BE]
        [n_entries x n_fields x float32_BE]

    The first entry is a header row:
    ``[channel_idx, n_spikes, trial_idx, stim_condition, 0, ...]``.
    Remaining entries are spike rows:
    ``[cluster_label, spike_time_idx, amplitude, slope, feature1, ...]``.

    Args:
        filepath: Path to the ``.ssort`` file.

    Returns:
        list[dict]: One dict per channel-trial record with keys
        ``channel_idx`` (int), ``n_spikes`` (int), ``trial_idx`` (int),
        ``stim_condition`` (int), ``labels`` (ndarray of int32),
        ``spike_indices`` (ndarray of float32), and ``features``
        (ndarray of float32).
    """
    records: list[dict] = []
    fsize = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        while f.tell() < fsize:
            raw_hdr = f.read(8)
            if len(raw_hdr) < 8:
                break
            n_entries, n_fields = struct.unpack('>II', raw_hdr)
            nbytes = n_entries * n_fields * 4
            if nbytes > fsize - f.tell():
                raise EOFError(
                    f"Record claims {n_entries}x{n_fields} entries "
                    f"({nbytes} bytes) but only {fsize - f.tell()} bytes remain"
                )
            raw = _read_exact(f, nbytes)
            block = np.frombuffer(raw, dtype='>f4').reshape(
                n_entries, n_fields
            )

            header_row = block[0]
            channel_idx = int(header_row[0])
            n_spikes = int(header_row[1])
            trial_idx = int(header_row[2]) if n_fields > 2 else 0
            stim_condition = int(header_row[3]) if n_fields > 3 else 0

            if n_spikes > 0:
                spike_rows = block[1:n_spikes + 1]
                labels = spike_rows[:, 0].astype(np.int32)
                spike_indices = spike_rows[:, 1].copy()
                features = (
                    spike_rows[:, 2:].copy()
                    if n_fields > 2
                    else np.empty((n_spikes, 0), dtype=np.float32)
                )
            else:
                labels = np.empty(0, dtype=np.int32)
                spike_indices = np.empty(0, dtype=np.float32)
                features = np.empty(
                    (0, max(0, n_fields - 2)), dtype=np.float32
                )

            records.append({
                'channel_idx': channel_idx,
                'n_spikes': n_spikes,
                'trial_idx': trial_idx,
                'stim_condition': stim_condition,
                'labels': labels,
                'spike_indices': spike_indices,
                'features': features,
            })
    return records


def write_ssort(
    filepath: str | Path,
    labels_per_record: list[np.ndarray],
    spike_indices_per_record: list[np.ndarray],
    features_per_record: list[np.ndarray] | None = None,
    n_fields: int = 10,
    channel_indices: list | np.ndarray | None = None,
    trial_indices: list | np.ndarray | None = None,
    stim_conditions: list | np.ndarray | None = None,
) -> None:
    """Write a ``.ssort`` spike-sorting results file.

    Produces a binary file matching the LabView ``.ssort`` format.
    Records are ordered trial-major, channel-minor (trial 0 channels
    0..N-1, then trial 1 channels 0..N-1, etc.).

    Args:
        filepath: Output file path.
        labels_per_record: List of 1-D int arrays (one per channel-trial).
            Each array contains the cluster label for every spike.
        spike_indices_per_record: List of 1-D float/int arrays of
            spike-time sample indices at the spike sampling rate.
        features_per_record: Optional list of 2-D float arrays
            ``(n_spikes, n_feat)`` with per-spike waveform features.
            If ``None``, feature columns are zero-filled.
        n_fields: Number of float32 columns per row (default 10).
        channel_indices: Optional per-record channel indices for the
            header row.  Defaults to the flat record index.
        trial_indices: Optional per-record trial indices for the
            header row.  Defaults to 0.
        stim_conditions: Optional per-record stimulus-condition codes
            for the header row.  Defaults to 0.
    """
    n_records = len(labels_per_record)
    n_feat = n_fields - 2  # columns: 0=label, 1=time_idx, 2..=features

    with open(filepath, 'wb') as f:
        for rec_idx in range(n_records):
            lab = np.asarray(
                labels_per_record[rec_idx], dtype=np.float32
            )
            idx = np.asarray(
                spike_indices_per_record[rec_idx], dtype=np.float32
            )
            n_spikes = len(lab)

            # Header row
            header = np.zeros(n_fields, dtype=np.float32)
            header[0] = float(
                channel_indices[rec_idx]
                if channel_indices is not None
                else rec_idx
            )
            header[1] = float(n_spikes)
            if trial_indices is not None and n_fields > 2:
                header[2] = float(trial_indices[rec_idx])
            if stim_conditions is not None and n_fields > 3:
                header[3] = float(stim_conditions[rec_idx])

            n_entries = 1 + n_spikes
            f.write(struct.pack('>II', n_entries, n_fields))
            f.write(header.astype('>f4').tobytes())

            if n_spikes > 0:
                rows = np.zeros(
                    (n_spikes, n_fields), dtype=np.float32
                )
                rows[:, 0] = lab
                rows[:, 1] = idx
                if features_per_record is not None:
                    feat = features_per_record[rec_idx]
                    if (
                        feat is not None
                        and feat.ndim == 2
                        and feat.shape[1] > 0
                    ):
                        cols = min(feat.shape[1], n_feat)
                        rows[:, 2:2 + cols] = np.asarray(
                            feat[:, :cols], dtype=np.float32
                        )
                f.write(rows.astype('>f4').tobytes())
