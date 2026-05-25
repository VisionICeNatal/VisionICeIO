"""Spike-sorting results I/O -- ``.ssort`` read and write.

Two distinct ``.ssort`` format variants are observed in the lab's real
recordings.  Both store records in trial-major, channel-minor order
(record ``i`` = trial ``i // n_ch``, channel ``i % n_ch``) -- 1:1
aligned with ``.spike`` / ``.swave``.

Each record (regardless of variant) starts with::

    [n_entries : u32_BE] [n_fields : u32_BE] [n_entries x n_fields x float32_BE]

Empty channel-trial records (no spikes detected) use different on-disk
representations in the two variants, matching what the lab sorter emits:

- **Variant A (v10)**: ``n_entries = 1`` — a header row but no spike rows.
  At ``n_fields = 10`` this is 8 + 40 = 48 bytes.  The header row stamps
  ``channel_idx`` / ``trial_idx`` / ``stim_condition`` so the positional
  metadata survives on disk; ``n_spikes`` in the header is ``0``.
- **Variant B (v16)**: ``[0 : u32_BE] [0 : u32_BE]`` (8 bytes, no payload).
  Channel / trial / stim metadata is unrecoverable from an empty v16
  record on disk; callers re-derive it from the record's flat index.

The reader transparently handles both forms (and accepts the ``[0,0]``
form for v10 as well, for backward compatibility with any pre-fix
``write_ssort`` output).

Variant A -- ``n_fields == 10`` (typical), header + spike rows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``n_entries = 1 (header) + n_spikes``.

Header row (entry 0):

    ===== =========================
    Col   Meaning
    ===== =========================
    0     channel_idx
    1     n_spikes
    2     trial_idx
    3     stim_condition
    4..   reserved (0)
    ===== =========================

Spike rows (entries ``1..n_spikes``):

    ===== =========================
    Col   Meaning
    ===== =========================
    0     cluster_label (int as float32)
    1     spike_sample_idx
    2     amp_max
    3     amp_min
    4     peak_to_peak  (= col 2 - col 3)
    5     width
    6     extra scalar feature (slope or similar)
    7-9   PCA1, PCA2, PCA3
    ===== =========================

Variant B -- ``n_fields == 16``, no header, redundant per-row metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``n_entries == n_spikes``.  Every row is one spike; channel/trial/stim
are stamped redundantly on every row.

    ===== =========================
    Col   Meaning
    ===== =========================
    0     channel_idx (constant within record)
    1     cluster_label
    2     trial_idx (constant within record)
    3     spike_sample_idx
    4     stim_condition (constant within record)
    5     reserved (always 0)
    6     amp_max
    7     amp_min
    8     peak_to_peak  (= col 6 - col 7)
    9     width
    10    extra scalar feature
    11-15 PCA1..PCA5
    ===== =========================

Variant detection
^^^^^^^^^^^^^^^^^

:func:`read_ssort` peeks at the first non-empty record's ``n_fields``.
``n_fields == 16`` -> Variant B; anything else -> Variant A.

If a file is *all* empty records (no spikes anywhere), Variant A is
assumed.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V16_NFIELDS = 16
"""Signature ``n_fields`` value identifying Variant B (no-header layout)."""

V10_DEFAULT_NFIELDS = 10
"""Default ``n_fields`` used when writing Variant A files."""


# ---------------------------------------------------------------------------
# Variant detection
# ---------------------------------------------------------------------------


def _peek_first_nonempty_n_fields(f, fsize: int) -> int | None:
    """Scan the file looking for the first non-empty record; return its
    ``n_fields``.  Restores the file pointer to byte 0 on exit."""
    f.seek(0)
    while f.tell() < fsize:
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        ne, nf = struct.unpack(">II", hdr)
        if ne > 0:
            f.seek(0)
            return nf
        # empty record: 0 payload, continue scanning
    f.seek(0)
    return None


# ---------------------------------------------------------------------------
# Public read entry point
# ---------------------------------------------------------------------------


def read_ssort(filepath: str | Path) -> list[dict]:
    """Read a ``.ssort`` spike-sorting results file.

    Auto-detects the file's format variant from the first non-empty
    record's ``n_fields`` field (16 -> Variant B, anything else ->
    Variant A) and dispatches to the appropriate parser.

    Args:
        filepath: Path to the ``.ssort`` file.

    Returns:
        list[dict]: One dict per channel-trial record (trial-major,
        channel-minor order, 1:1 aligned with ``.spike`` / ``.swave``).
        Empty channel-trials yield a record with ``n_spikes == 0``
        and empty per-spike arrays.

        Each record has the following keys:

        - ``channel_idx`` (int)
        - ``n_spikes`` (int)
        - ``trial_idx`` (int)
        - ``stim_condition`` (int)
        - ``variant`` (str): ``'v10'`` or ``'v16'``
        - ``labels`` (ndarray[int32], shape ``(n_spikes,)``):
          cluster labels
        - ``spike_indices`` (ndarray[float32], shape ``(n_spikes,)``):
          spike sample indices at the spike sampling rate
        - ``amp_max`` (ndarray[float32], shape ``(n_spikes,)``)
        - ``amp_min`` (ndarray[float32], shape ``(n_spikes,)``)
        - ``peak_to_peak`` (ndarray[float32], shape ``(n_spikes,)``)
        - ``width`` (ndarray[float32], shape ``(n_spikes,)``)
        - ``features`` (ndarray[float32], shape ``(n_spikes, n_feat)``):
          remaining columns after the named fields above.  For Variant
          A this is ``cols[6:]`` (slope + PCA); for Variant B
          ``cols[10:]`` (extra + PCA1..PCA5).

    Raises:
        EOFError: On truncated records.
        ValueError: On structural inconsistencies.
    """
    filepath = str(filepath)
    fsize = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        nf_first = _peek_first_nonempty_n_fields(f, fsize)
        if nf_first == V16_NFIELDS:
            return _read_ssort_v16(f, fsize)
        return _read_ssort_v10(f, fsize)


# ---------------------------------------------------------------------------
# Variant A reader (header + spike rows)
# ---------------------------------------------------------------------------


def _read_ssort_v10(f, fsize: int) -> list[dict]:
    """Read a Variant A (header + spike rows) ``.ssort`` file."""
    records: list[dict] = []
    while f.tell() < fsize:
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        n_entries, n_fields = struct.unpack(">II", hdr)

        if n_entries == 0:
            # Empty channel-trial: no payload, no metadata available
            records.append(_empty_record("v10", n_fields))
            continue

        if n_fields < 2:
            raise ValueError(f"Variant A .ssort requires n_fields >= 2, got {n_fields}")

        nbytes = n_entries * n_fields * 4
        if nbytes > fsize - f.tell():
            raise EOFError(
                f"Record claims {n_entries}x{n_fields} entries "
                f"({nbytes} bytes) but only {fsize - f.tell()} bytes remain"
            )
        raw = f.read(nbytes)
        block = np.frombuffer(raw, dtype=">f4").reshape(n_entries, n_fields)

        header = block[0]
        channel_idx = int(header[0])
        declared_n_spikes = int(header[1])
        trial_idx = int(header[2]) if n_fields > 2 else 0
        stim_condition = int(header[3]) if n_fields > 3 else 0

        # Trust the row count over the header's declared n_spikes when
        # they disagree (header is a denormalization and can be stale).
        actual_n_spikes = n_entries - 1
        n_spikes = (
            min(declared_n_spikes, actual_n_spikes) if declared_n_spikes >= 0 else actual_n_spikes
        )

        if n_spikes > 0:
            rows = block[1 : 1 + n_spikes]
            labels = rows[:, 0].astype(np.int32)
            spike_indices = rows[:, 1].astype(np.float32)
            amp_max = (
                rows[:, 2].astype(np.float32)
                if n_fields > 2
                else np.zeros(n_spikes, dtype=np.float32)
            )
            amp_min = (
                rows[:, 3].astype(np.float32)
                if n_fields > 3
                else np.zeros(n_spikes, dtype=np.float32)
            )
            peak_to_peak = (
                rows[:, 4].astype(np.float32)
                if n_fields > 4
                else np.zeros(n_spikes, dtype=np.float32)
            )
            width = (
                rows[:, 5].astype(np.float32)
                if n_fields > 5
                else np.zeros(n_spikes, dtype=np.float32)
            )
            features = (
                rows[:, 6:].astype(np.float32)
                if n_fields > 6
                else np.empty((n_spikes, 0), dtype=np.float32)
            )
        else:
            labels, spike_indices, amp_max, amp_min, peak_to_peak, width, features = (
                _empty_spike_arrays(
                    "v10",
                    n_fields,
                )
            )

        records.append(
            {
                "channel_idx": channel_idx,
                "n_spikes": n_spikes,
                "trial_idx": trial_idx,
                "stim_condition": stim_condition,
                "variant": "v10",
                "labels": labels,
                "spike_indices": spike_indices,
                "amp_max": amp_max,
                "amp_min": amp_min,
                "peak_to_peak": peak_to_peak,
                "width": width,
                "features": features,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Variant B reader (no header, per-row metadata)
# ---------------------------------------------------------------------------


def _read_ssort_v16(f, fsize: int) -> list[dict]:
    """Read a Variant B (no header, 16 columns) ``.ssort`` file."""
    records: list[dict] = []
    while f.tell() < fsize:
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        n_entries, n_fields = struct.unpack(">II", hdr)

        if n_entries == 0:
            records.append(_empty_record("v16", V16_NFIELDS))
            continue

        if n_fields != V16_NFIELDS:
            raise ValueError(
                f"Variant B .ssort: expected n_fields={V16_NFIELDS}, "
                f"got {n_fields} at offset {f.tell() - 8}"
            )

        nbytes = n_entries * n_fields * 4
        if nbytes > fsize - f.tell():
            raise EOFError(
                f"Record claims {n_entries}x{n_fields} entries "
                f"({nbytes} bytes) but only {fsize - f.tell()} bytes remain"
            )
        raw = f.read(nbytes)
        block = np.frombuffer(raw, dtype=">f4").reshape(n_entries, n_fields)

        n_spikes = n_entries
        channel_idx = int(block[0, 0])
        trial_idx = int(block[0, 2])
        stim_condition = int(block[0, 4])

        labels = block[:, 1].astype(np.int32)
        spike_indices = block[:, 3].astype(np.float32)
        amp_max = block[:, 6].astype(np.float32)
        amp_min = block[:, 7].astype(np.float32)
        peak_to_peak = block[:, 8].astype(np.float32)
        width = block[:, 9].astype(np.float32)
        features = block[:, 10:].astype(np.float32)

        records.append(
            {
                "channel_idx": channel_idx,
                "n_spikes": n_spikes,
                "trial_idx": trial_idx,
                "stim_condition": stim_condition,
                "variant": "v16",
                "labels": labels,
                "spike_indices": spike_indices,
                "amp_max": amp_max,
                "amp_min": amp_min,
                "peak_to_peak": peak_to_peak,
                "width": width,
                "features": features,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Empty-record helpers
# ---------------------------------------------------------------------------


def _n_features_for_variant(variant: str, n_fields: int) -> int:
    """Number of trailing 'feature' columns for a given variant/n_fields."""
    if variant == "v10":
        # Cols 0..5 are named (label, idx, amp_max, amp_min, p2p, width)
        return max(n_fields - 6, 0)
    if variant == "v16":
        # Cols 0..9 are named/metadata
        return max(n_fields - 10, 0)
    raise ValueError(f"Unknown ssort variant {variant!r}")


def _empty_spike_arrays(variant: str, n_fields: int):
    """Return (labels, spike_indices, amp_max, amp_min, p2p, width, features)
    arrays for a record with ``n_spikes == 0``."""
    n_feat = _n_features_for_variant(variant, n_fields)
    return (
        np.empty(0, dtype=np.int32),  # labels
        np.empty(0, dtype=np.float32),  # spike_indices
        np.empty(0, dtype=np.float32),  # amp_max
        np.empty(0, dtype=np.float32),  # amp_min
        np.empty(0, dtype=np.float32),  # peak_to_peak
        np.empty(0, dtype=np.float32),  # width
        np.empty((0, n_feat), dtype=np.float32),  # features
    )


def _empty_record(variant: str, n_fields: int) -> dict:
    """Build the record dict for an empty (n_spikes=0) channel-trial.

    Channel / trial / stim are not recoverable from an empty record on
    disk -- callers that need positional metadata should re-derive it
    from the record's index in the returned list.
    """
    labels, indices, amax, amin, p2p, width, feats = _empty_spike_arrays(
        variant,
        n_fields,
    )
    return {
        "channel_idx": 0,
        "n_spikes": 0,
        "trial_idx": 0,
        "stim_condition": 0,
        "variant": variant,
        "labels": labels,
        "spike_indices": indices,
        "amp_max": amax,
        "amp_min": amin,
        "peak_to_peak": p2p,
        "width": width,
        "features": feats,
    }


# ---------------------------------------------------------------------------
# Public write entry point
# ---------------------------------------------------------------------------


def write_ssort(
    filepath: str | Path,
    labels_per_record: list[np.ndarray],
    spike_indices_per_record: list[np.ndarray],
    features_per_record: list[np.ndarray] | None = None,
    n_fields: int = V10_DEFAULT_NFIELDS,
    channel_indices: list | np.ndarray | None = None,
    trial_indices: list | np.ndarray | None = None,
    stim_conditions: list | np.ndarray | None = None,
    amp_max_per_record: list[np.ndarray] | None = None,
    amp_min_per_record: list[np.ndarray] | None = None,
    peak_to_peak_per_record: list[np.ndarray] | None = None,
    width_per_record: list[np.ndarray] | None = None,
) -> None:
    """Write a ``.ssort`` spike-sorting results file.

    The variant is selected by ``n_fields``:

    - ``n_fields == 16`` -> Variant B (no header row, per-row metadata).
    - any other value     -> Variant A (header row + spike rows).

    Records are ordered trial-major, channel-minor (trial 0 channels
    ``0..N-1``, then trial 1 channels ``0..N-1``, ...).

    Args:
        filepath: Output file path.
        labels_per_record: List of 1-D label arrays (one per channel-trial).
        spike_indices_per_record: List of 1-D arrays of spike-time sample
            indices at the spike sampling rate.
        features_per_record: Optional list of 2-D float arrays
            ``(n_spikes, n_feat)``.  Written into columns *after* the
            named amplitude/width columns -- Variant A: column 6+,
            Variant B: column 10+.  Excess columns are truncated;
            missing columns are zero-filled.
        n_fields: Number of float32 columns per row.  Must be ``>= 2``
            for Variant A; must equal ``16`` for Variant B.
        channel_indices: Optional per-record channel indices.
            Defaults to ``i % max_channel`` if discernible, else the
            flat record index.
        trial_indices: Optional per-record trial indices.  Defaults to 0.
        stim_conditions: Optional per-record stimulus-condition codes.
            Defaults to 0.
        amp_max_per_record: Optional per-spike amp_max arrays (Variant
            A: column 2; Variant B: column 6).  Zero-filled if absent.
        amp_min_per_record: As above for amp_min.
        peak_to_peak_per_record: As above for peak-to-peak amplitude.
        width_per_record: As above for spike width.
    """
    if n_fields == V16_NFIELDS:
        _write_ssort_v16(
            filepath,
            labels_per_record,
            spike_indices_per_record,
            features_per_record,
            channel_indices,
            trial_indices,
            stim_conditions,
            amp_max_per_record,
            amp_min_per_record,
            peak_to_peak_per_record,
            width_per_record,
        )
    else:
        _write_ssort_v10(
            filepath,
            labels_per_record,
            spike_indices_per_record,
            features_per_record,
            n_fields,
            channel_indices,
            trial_indices,
            stim_conditions,
            amp_max_per_record,
            amp_min_per_record,
            peak_to_peak_per_record,
            width_per_record,
        )


# ---------------------------------------------------------------------------
# Variant A writer
# ---------------------------------------------------------------------------


def _write_ssort_v10(
    filepath,
    labels_per_record,
    spike_indices_per_record,
    features_per_record,
    n_fields,
    channel_indices,
    trial_indices,
    stim_conditions,
    amp_max_per_record,
    amp_min_per_record,
    peak_to_peak_per_record,
    width_per_record,
) -> None:
    if n_fields < 2:
        raise ValueError(f"Variant A .ssort requires n_fields >= 2, got {n_fields}")
    n_records = len(labels_per_record)

    with open(filepath, "wb") as f:
        for rec_idx in range(n_records):
            lab = np.asarray(labels_per_record[rec_idx], dtype=np.float32)
            idx = np.asarray(
                spike_indices_per_record[rec_idx],
                dtype=np.float32,
            )
            n_spikes = len(lab)

            # Header row: always emitted, even for empty channel-trials.
            # Real-world v10 files use this header-only representation
            # (n_entries=1) for empty records — never the [0,0] sentinel.
            # See module docstring for the full convention.
            header = np.zeros(n_fields, dtype=np.float32)
            header[0] = float(channel_indices[rec_idx] if channel_indices is not None else rec_idx)
            header[1] = float(n_spikes)
            if trial_indices is not None and n_fields > 2:
                header[2] = float(trial_indices[rec_idx])
            if stim_conditions is not None and n_fields > 3:
                header[3] = float(stim_conditions[rec_idx])

            n_entries = 1 + n_spikes
            f.write(struct.pack(">II", n_entries, n_fields))
            f.write(header.astype(">f4").tobytes())

            if n_spikes > 0:
                rows = np.zeros((n_spikes, n_fields), dtype=np.float32)
                rows[:, 0] = lab
                if n_fields > 1:
                    rows[:, 1] = idx
                _fill_named(rows, 2, amp_max_per_record, rec_idx, n_fields)
                _fill_named(rows, 3, amp_min_per_record, rec_idx, n_fields)
                _fill_named(rows, 4, peak_to_peak_per_record, rec_idx, n_fields)
                _fill_named(rows, 5, width_per_record, rec_idx, n_fields)
                # Features start at column 6
                if features_per_record is not None:
                    feat = features_per_record[rec_idx]
                    if (
                        feat is not None
                        and getattr(feat, "ndim", 0) == 2
                        and feat.shape[1] > 0
                        and n_fields > 6
                    ):
                        cols = min(feat.shape[1], n_fields - 6)
                        rows[:, 6 : 6 + cols] = np.asarray(
                            feat[:, :cols],
                            dtype=np.float32,
                        )
                f.write(rows.astype(">f4").tobytes())


# ---------------------------------------------------------------------------
# Variant B writer
# ---------------------------------------------------------------------------


def _write_ssort_v16(
    filepath,
    labels_per_record,
    spike_indices_per_record,
    features_per_record,
    channel_indices,
    trial_indices,
    stim_conditions,
    amp_max_per_record,
    amp_min_per_record,
    peak_to_peak_per_record,
    width_per_record,
) -> None:
    n_records = len(labels_per_record)
    n_fields = V16_NFIELDS

    with open(filepath, "wb") as f:
        for rec_idx in range(n_records):
            lab = np.asarray(labels_per_record[rec_idx], dtype=np.float32)
            idx = np.asarray(
                spike_indices_per_record[rec_idx],
                dtype=np.float32,
            )
            n_spikes = len(lab)

            if n_spikes == 0:
                f.write(struct.pack(">II", 0, 0))
                continue

            chan = float(channel_indices[rec_idx] if channel_indices is not None else rec_idx)
            trial = float(trial_indices[rec_idx] if trial_indices is not None else 0)
            stim = float(stim_conditions[rec_idx] if stim_conditions is not None else 0)

            rows = np.zeros((n_spikes, n_fields), dtype=np.float32)
            rows[:, 0] = chan
            rows[:, 1] = lab
            rows[:, 2] = trial
            rows[:, 3] = idx
            rows[:, 4] = stim
            # col 5 reserved (zero)
            _fill_named(rows, 6, amp_max_per_record, rec_idx, n_fields)
            _fill_named(rows, 7, amp_min_per_record, rec_idx, n_fields)
            _fill_named(rows, 8, peak_to_peak_per_record, rec_idx, n_fields)
            _fill_named(rows, 9, width_per_record, rec_idx, n_fields)
            # Features start at column 10
            if features_per_record is not None:
                feat = features_per_record[rec_idx]
                if feat is not None and getattr(feat, "ndim", 0) == 2 and feat.shape[1] > 0:
                    cols = min(feat.shape[1], n_fields - 10)
                    rows[:, 10 : 10 + cols] = np.asarray(
                        feat[:, :cols],
                        dtype=np.float32,
                    )

            f.write(struct.pack(">II", n_spikes, n_fields))
            f.write(rows.astype(">f4").tobytes())


# ---------------------------------------------------------------------------
# Write-side small helpers
# ---------------------------------------------------------------------------


def _fill_named(rows, col, per_record, rec_idx, n_fields):
    """Fill one column of ``rows`` from a per-record array if provided."""
    if per_record is None or col >= n_fields:
        return
    arr = per_record[rec_idx]
    if arr is None:
        return
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] != rows.shape[0]:
        raise ValueError(
            f"named column at col {col}: expected length {rows.shape[0]} "
            f"but got {arr.shape[0]} for record {rec_idx}"
        )
    rows[:, col] = arr
