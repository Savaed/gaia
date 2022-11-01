"""
IO operations that can be run in any of the supported environments: local,
Google Cloud Platform (GCP), HDFS.
"""

from typing import Protocol, TypeVar

import numpy as np
import pandas as pd
import structlog
import tensorflow as tf
from astropy.io import fits
from tensorflow.python.framework.errors import NotFoundError  # pylint: disable=no-name-in-module

from gaia.enums import Cadence


T_co = TypeVar("T_co", covariant=True)


logger = structlog.stdlib.get_logger()

# Quarter index to filename prefix for long cadence Kepler data.
# Reference: https://archive.stsci.edu/kepler/software/get_kepler.py
_LONG_CADENCE_QUARTER_PREFIXES = (
    ["2009131105131"],
    ["2009166043257"],
    ["2009259160929"],
    ["2009350155506"],
    ["2010078095331", "2010009091648"],
    ["2010174085026"],
    ["2010265121752"],
    ["2010355172524"],
    ["2011073133259"],
    ["2011177032512"],
    ["2011271113734"],
    ["2012004120508"],
    ["2012088054726"],
    ["2012179063303"],
    ["2012277125453"],
    ["2013011073258"],
    ["2013098041711"],
    ["2013131215648"],
)

# Quarter index to filename prefix for short cadence Kepler data.
# Reference: https://archive.stsci.edu/kepler/software/get_kepler.py
_SHORT_CADENCE_QUARTER_PREFIXES = (
    ["2009131110544"],
    ["2009166044711"],
    ["2009201121230", "2009231120729", "2009259162342"],
    ["2009291181958", "2009322144938", "2009350160919"],
    ["2010009094841", "2010019161129", "2010049094358", "2010078100744"],
    ["2010111051353", "2010140023957", "2010174090439"],
    ["2010203174610", "2010234115140", "2010265121752"],
    ["2010296114515", "2010326094124", "2010355172524"],
    ["2011024051157", "2011053090032", "2011073133259"],
    ["2011116030358", "2011145075126", "2011177032512"],
    ["2011208035123", "2011240104155", "2011271113734"],
    ["2011303113607", "2011334093404", "2012004120508"],
    ["2012032013838", "2012060035710", "2012088054726"],
    ["2012121044856", "2012151031540", "2012179063303"],
    ["2012211050319", "2012242122129", "2012277125453"],
    ["2012310112549", "2012341132017", "2013011073258"],
    ["2013017113907", "2013065031647", "2013098041711"],
    ["2013121191144", "2013131215648"],
)


class FITSReadingError(Exception):
    """Raised when any error occurs while reading the FITS file."""


class MissingExtensionHDUError(Exception):
    """Raised when the specified extension HDU not found in the FITS file."""


class Reader(Protocol[T_co]):
    """Generic data reader."""

    def read(self, source: str) -> T_co:
        """Read a data from a specific source.

        Args:
            source (str): The source location

        Returns:
            T_co: Generic data
        """
        ...


class DataFrameReader:
    """CSV file reader that return a tabular data as pandas DataFrame object."""

    def __init__(self) -> None:
        self.log = logger.new()

    def read(self, source: str) -> pd.DataFrame:
        """Read a CSV file as pandas DataFrame object.

        Args:
            source (str): A source location. It can be local system or external e.g. HDFS, GCP

        Raises:
            FileNotFoundError: No file found

        Returns:
            pd.DataFrame: Tabular data
        """
        try:
            self.log.debug("Reading CSV file", path=source)
            with tf.io.gfile.GFile(source) as gf:
                return pd.read_csv(gf)
        except NotFoundError as ex:
            raise FileNotFoundError(source) from ex


class FITSTimeSeriesReader:
    """FITS file time series reader."""

    def __init__(self, hdu: str = "LIGHTCURVE") -> None:
        """Initialize a `FITSTimeSeriesReader` object.

        Args:
            hdu (str, optional): HDU time series extension name. Defaults to "LIGHTCURVE".
        """
        self.log = logger.new()
        self._hdu = hdu

    def read(self, source: str) -> dict[str, np.ndarray]:
        """Read time series data from a FITS file.

        A file may be read from any of the following locations: local, GCP, HDFS or AWS.

        Args:
            source (str): The location of the source FITS file

        Raises:
            FileNotFoundError: No file found

        Returns:
            dict[str, np.ndarray]: Time series in format of `{"fits_columns_name", values, ...}`
        """
        return self._read_as_dict(source)

    def _read_as_dict(self, source: str) -> dict[str, np.ndarray]:
        """Read a FITS file as a dict."""
        self.log.debug("Reading FITS file", path=source)
        try:
            with tf.io.gfile.GFile(source, mode="rb") as gf, fits.open(gf) as hdul:
                time_series = hdul[self._hdu]
                self.log.debug("Extension HDU read", hdu=self._hdu)
                data = {col.name: time_series.data[col.name] for col in time_series.columns}
                return data
        except NotFoundError as ex:
            raise FileNotFoundError(source) from ex
        except KeyError as ex:
            raise MissingExtensionHDUError(
                f"Extension HDU {self._hdu!r} not found in {source!r}"
            ) from ex
        except Exception as ex:
            raise FITSReadingError(f"Unable to read {source}. {ex}") from ex


def _check_kepid(kepid: int) -> None:
    """Raise `ValueError` if `kepid` is outside of the range [1-999 999 999].

    Args:
        kepid (int): The ID of Kepler Object of Interest (KOI) target star or star system
    """
    if not 0 < kepid < 1_000_000_000:
        raise ValueError(f"Specified {kepid=} is outside of the range [1-999 999 999]")


def get_quarter_prefixes(cadence: Cadence) -> list[str]:
    """Get Kepler observation quarters prefixes for a specified cadence.

    Args:
        cadence (Cadence): Observation cadence

    Returns:
        list[str]: Quarter prefixes (quarter start date)
    """
    logger.debug("Getting Kepler quarters prefixes", cadence=cadence)
    queraters = (
        _SHORT_CADENCE_QUARTER_PREFIXES
        if cadence is Cadence.SHORT
        else _LONG_CADENCE_QUARTER_PREFIXES
    )

    quarter_prefs = []
    for prefs in queraters:
        quarter_prefs.extend(prefs)

    return quarter_prefs


def get_kepler_fits_paths(
    data_dir: str, kepid: int, cadence: Cadence, quarters: tuple[str, ...] | None = None
) -> list[str]:
    """Generate file paths for Kepler time series FITS files.

    Args:
        data_dir (str): Source folder path
        kepid (int): The ID of KOI target
        cadence (Cadence): Cadence of observations
        quarters (Optional[tuple[str, ...]], optional): Prefixes of the quarters for which the time
        series should be returned. If None, then all available data will be returned. Defaults to
        None.

    Returns:
        list[str]: Filenames for a specific KOI and cadence located in `data_dir` folder optionaly
        filtered by quarter prefixes

    Raises:
        ValueError: Parameter `kepid` is outside of the range 1-999 999 999
    """
    _check_kepid(kepid)
    prefixes = get_quarter_prefixes(cadence)

    if quarters:
        logger.debug("Filter Kepler paths based on quarter prefixes", include_quarters=quarters)
        prefixes = [pref for pref in prefixes if pref in quarters]

    return [
        f"{data_dir}/{kepid:09}/kplr{kepid:09}-{quarter_prefix}_{cadence.value}.fits"
        for quarter_prefix in prefixes
    ]
