"""Zarr persistence -- reload a previously-saved Zarr store."""

from __future__ import annotations

import os
from pathlib import Path


def _check_zarr_version_compat(zarr_path: str) -> None:
    """Raise a clear error if a v3 store is opened with zarr v2 installed.

    A zarr v3 store has a top-level ``zarr.json`` and *no* ``.zmetadata``;
    a zarr v2 store has ``.zmetadata`` (consolidated) and/or ``.zgroup``
    but no ``zarr.json``.  When zarr v2 is installed and the user points
    at a v3 store, xarray's open path eventually raises
    ``FileNotFoundError: No such file or directory: '<zarr_path>'`` —
    which is wrong (the path exists) and sends users down the wrong
    debugging trail.  We surface a clearer error before that happens.
    """
    has_v3_marker = os.path.exists(os.path.join(zarr_path, "zarr.json"))
    has_v2_marker = os.path.exists(os.path.join(zarr_path, ".zmetadata")) or os.path.exists(
        os.path.join(zarr_path, ".zgroup")
    )

    if not has_v3_marker or has_v2_marker:
        # Either it's a v2 store (any zarr can read it) or it has no
        # recognisable layout (let xarray/zarr raise their own error).
        return

    try:
        import zarr
    except ImportError:
        return  # missing zarr -> let xarray raise its own ImportError

    zarr_major = int(zarr.__version__.split(".")[0])
    if zarr_major < 3:
        raise ValueError(
            f"Zarr store {zarr_path!r} appears to be in v3 format "
            f"(found 'zarr.json', no '.zmetadata') but installed zarr "
            f"is v{zarr.__version__}.  Upgrade zarr to >=3 to read this "
            f"store, or re-write it with zarr v2."
        )


def load_from_zarr(
    zarr_path: str | Path,
    electrode: int | None = None,
):
    """Load a previously-saved zarr store back into an xarray structure.

    This allows re-loading data that was saved by
    ``Experiment.load_from_dir(..., save_as='zarr')``.

    Args:
        zarr_path: Path to the ``.zarr`` store directory.
        electrode: If specified, select a single electrode from the
            dataset.  Otherwise the full ``xr.Dataset`` is returned.

    Returns:
        ``xr.Dataset`` (full) or electrode-sliced ``xr.Dataset`` when
        *electrode* is given.

    Raises:
        FileNotFoundError: If the zarr store does not exist.
        ValueError: If the store is in zarr v3 format but the installed
            zarr is v2 (xarray would otherwise raise a misleading
            ``FileNotFoundError`` against the existing directory).
        ImportError: If xarray or zarr are not installed.
    """
    import xarray as xr

    zarr_path = str(zarr_path)
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    _check_zarr_version_compat(zarr_path)

    ds = xr.open_zarr(zarr_path)

    if electrode is not None:
        return ds.sel(electrodes=electrode)

    return ds
