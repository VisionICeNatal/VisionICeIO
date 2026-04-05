"""Zarr persistence -- reload a previously-saved Zarr store."""

from __future__ import annotations

import os
from pathlib import Path


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
        ImportError: If xarray or zarr are not installed.
    """
    import xarray as xr

    zarr_path = str(zarr_path)
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    ds = xr.open_zarr(zarr_path)

    if electrode is not None:
        return ds.sel(electrodes=electrode)

    return ds
