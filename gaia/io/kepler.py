"""DTO for Kepler time series stellar parameters and TCE scalar features."""

import io
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

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


def _check_kepid(kepid: int) -> None:  # pylint: disable=missing-function-docstring
    if not 0 < kepid < 1_000_000_000:
        raise ValueError(f"'kepid' must be in range 1 - 999 999 999 inclusive, but got {kepid=}")


def _read_csv_as_df(filename) -> pd.DataFrame:  # pylint: disable=missing-function-docstring
    return pd.read_csv(io.BytesIO(mp_read(filename, mode="rb")))


def get_kepler_filenames(data_dir: str, kepid: int, cadence: Cadence) -> list[str]:
    """Generate file paths for Kepler time series FITS files.

    Parameters
    ----------
    data_dir : str
        Source folder path
    kepid : int
        The ID of KOI target
    cadence : Cadence
        Cadence of observations

    Returns
    -------
    list[str]
        Filenames for a specific KOI and cadence located in `data_dir` folder

    Raises
    ------
    ValueError
        Parameter `kepid` is outside of the range 1-999 999 999
    """
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
        FITS file not found
    """
    try:
        with fits.open(io.BytesIO(mp_read(filename))) as hdul:
            time_series = hdul["LIGHTCURVE"]
            return {column.name: time_series.data[column.name] for column in time_series.columns}
    except FileNotFoundError:
        raise
    except Exception as ex:
        raise FitsFileError(f"Cannot read a file '{filename}'. {ex}") from ex


def read_tce(filename: str, kepid: Optional[int] = None) -> list[TCE]:
    """Read TCEs for one or all KOIs.

    Parameters
    ----------
    filename : str
        Location of a CSV file
    kepid : Optional[int], optional
        The ID of KOI target. If `None` then return TCEs for all KOIs available in the file, by
        default None

    Returns
    -------
    list[TCE]
        Transit-like events

    Raises
    ------
    MissingKOI
        No KOI found
    ValueError
        Parameter `kepid` is outside of the range 1-999 999 999
    FileNotFoundError
        No CSV file found
    KeyError:
        Missing required CSV column in a file
    """
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
    """Read time series for a specific KOI.

    Parameters
    ----------
    kepid : int
        The ID of KOI target
    data_dir : str
        Root time series folder
    cadence : Cadence
        Observation cadence
    include : Optional[dict[str, str]], optional
        A dictionary that tells which fields in the FITS file should be mapped to TimeSeries
        attributes. It must be {FITS field: TimeSeries attribute}. If `None`, all fields are
        mapped without renaming them, by default None

    Returns
    -------
    TimeSeries
        An object which contains attributes from the `include` parameter

    Raises
    ------
    FitsFileError
        An issue with reading a FITS file
    ValueError
        Parameter `kepid` is outside of the range 1-999 999 999
    """
    filenames = get_kepler_filenames(data_dir, kepid, cadence)
    time_series = defaultdict(lambda: [])

    for path in filenames:
        try:
            file_time_series = read_fits_as_dict(path)
        except FileNotFoundError:
            # No file for a specific quarter
            continue

        for ts_field, ts_data in file_time_series.items():
            if include is None:
                mapped_key = ts_field
            elif ts_field in include:
                mapped_key = include[ts_field]
            else:
                continue

            time_series[mapped_key].append(ts_data)

    if not time_series:
        # No file found, there are no files for a specified cadence or kepid at all
        raise FitsFileError(f"No file found for {kepid=}, {cadence=} at location={data_dir!r}")

    return TimeSeries(kepid, data=time_series)


def read_stellar_params(filename: str, kepid: Optional[int] = None) -> list[StellarParameters]:
    """Read stellar features for one or all KOIs.

    Parameters
    ----------
    filename : str
        Location of a CSV file
    kepid : Optional[int], optional
        The ID of KOI target. If `None` then return stellar parameters for all KOIs available in
        the file, by default None

    Returns
    -------
    list[StellarParameters]
        Target(s) stellar properties

    Raises
    ------
    MissingKOI
        No KOI found
    ValueError
        Parameter `kepid` is outside of the range 1-999 999 999
    FileNotFoundError
        No CSV file found
    KeyError:
        Missing required CSV column in a file
    """
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
