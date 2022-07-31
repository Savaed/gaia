"""DTO for Kepler time series stellar parameters and TCE scalar features."""

import io
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional
from zlib import DEFLATED

import numpy as np
import pandas as pd
from astropy.io import fits

from gaia.constants import get_quarter_prefixes
from gaia.data.models import TCE, KeplerData, StellarParameters, TimeSeries
from gaia.enums import Cadence
from gaia.io.core import read as mp_read


class FitsFileError(Exception):
    """Raised when any error with a FITS file occures."""


class MissingKOI(Exception):
    """Raised when a specified KOI is missing in an existing file."""


def _check_kepid(kepid: int):
    if not 0 < kepid < 1_000_000_000:
        raise ValueError(f"'kepid' must be in range 1 - 999 999 999 inclusive, but got {kepid=}")


def _read_csv_as_df(filename):
    return pd.read_csv(io.BytesIO(mp_read(filename, mode="rb")))


def get_kepler_filenames(data_dir: str, kepid: int, cadence: Cadence) -> list[str]:
    _check_kepid(kepid)
    return [
        f"{data_dir}/{kepid:09}/kplr{kepid:09}-{quarter_prefix}_{cadence.value}.fits"
        for quarter_prefix in get_quarter_prefixes(cadence)
    ]


def read_fits_as_dict(filename: str) -> dict[str, Any]:
    """Read time series from a FITS file as a dictionary with all fields mapped.

    Parameters
    ----------
    filename : str
        Location of a FITS file

    Returns
    -------
    dict[str, Any]
        Mapped time series

    Raises
    ------
    FitsFileError
        Issue with reading a file
    FileNotFoundError
    """
    try:
        with fits.open(io.BytesIO(mp_read(filename))) as hdul:
            time_series = hdul["LIGHTCURVE"]
            return {column.name: time_series.data[column.name] for column in time_series.columns}
    except FileNotFoundError:
        raise
    except Exception as ex:
        raise FitsFileError(f"Cannot read a file '{filename}'. {ex}")


def read_tce(filename: str, kepid: Optional[int] = None) -> list[TCE]:
    if kepid is not None:
        _check_kepid(kepid)

    df = _read_csv_as_df(filename)

    if kepid is not None:
        df = df.loc[df["kepid"] == kepid]

        if df.empty:
            raise MissingKOI(f"Cannot find any TCE for {kepid=}")

    return [TCE.from_dict(row) for _, row in df.iterrows()]


def read_time_series(
    kepid: int, data_dir: str, cadence: Cadence, include: Optional[dict[str, str]] = None
) -> TimeSeries:
    filenames = get_kepler_filenames(data_dir, kepid, cadence)
    time_series = defaultdict(lambda: [])

    for path in filenames:
        try:
            file_time_series = read_fits_as_dict(path)
        except FileNotFoundError:
            # No file for a specific quarter
            continue

        for k, v in file_time_series.items():
            if include is None:
                mapped_key = k
            elif k in include:
                mapped_key = include[k]
            else:
                continue

            time_series[mapped_key].append(v)

    if not time_series:
        # No file found, there are no files for a specified cadence or kepid at all
        raise FitsFileError(f"No file found for {kepid=}, {cadence=} at location={data_dir!r}")

    return TimeSeries(kepid, data=time_series)


def read_stellar_params(filename: str, kepid: Optional[int] = None) -> StellarParameters:
    if kepid is not None:
        _check_kepid(kepid)

    df = _read_csv_as_df(filename)

    if kepid is not None:
        df = df.loc[df["kepid"] == kepid]

        if df.empty:
            raise MissingKOI(f"Cannot find any stellar parameters for {kepid=}")

    return [StellarParameters.from_dict(row) for _, row in df.iterrows()]


@dataclass(frozen=True)
class FilesLocations:
    tce: str
    stellar: str
    time_series: str


def read_kepler_data(data_location: FilesLocations, kepid: int) -> KeplerData:
    raise NotImplementedError
