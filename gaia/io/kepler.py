from collections import defaultdict
from dataclasses import dataclass
from typing import Any, AnyStr, Optional, Union

import pandas as pd
import tensorflow as tf
from astropy.io import fits
from tensorflow.python.framework.errors import NotFoundError  # pylint: disable=no-name-in-module

from gaia.constants import get_quarter_prefixes
from gaia.data.models import TCE, KeplerData, StellarParameters, TimeSeries
from gaia.enums import Cadence, TceLabel, TceSpecificLabel


@dataclass(frozen=True)
class MissingKOI(Exception):
    """Raised when a specified KOI is missing in an existing file."""

    kepid: int


class FitsFileError(Exception):
    """Raised when any error with a FITS file occures."""


def gfile_read(filename: str, all_lines: bool = False) -> Union[list[AnyStr], AnyStr]:
    with tf.io.gfile.GFile(filename, "r") as gf:
        return gf.readlines() if all_lines else gf.read()


def gfile_write(filename: str, data: Any, append: bool = False) -> None:
    mode = "a" if append else "w"
    with tf.io.gfile.GFile(filename, mode) as gf:
        gf.write(data)


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
    check_kepid(kepid)
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
        with fits.open(tf.io.gfile.GFile(filename, mode="rb")) as hdul:
            time_series = hdul["LIGHTCURVE"]
            return {column.name: time_series.data[column.name] for column in time_series.columns}
    except NotFoundError as ex:
        raise FileNotFoundError(ex) from ex
    except Exception as ex:
        raise FitsFileError(f"Cannot read a file '{filename}'. {ex}") from ex


def pd_read_csv(filename: str) -> pd.DataFrame:
    with tf.io.gfile.GFile(filename) as gf:
        return pd.read_csv(gf)


def check_kepid(kepid: int) -> None:
    if not 0 < kepid < 1_000_000_000:
        raise ValueError(f"'kepid' must be in range 1 - 999 999 999 inclusive, but got {kepid=}")


@dataclass(frozen=True)
class FilesLocation:
    tce_filename: str
    stellar_filename: str
    cfp_filename: str
    koi_filename: str
    time_series_dir: str


class KeplerReader:
    def __init__(
        self,
        tce_filename: str,
        stellar_filename: str,
        cfp_filename: str,
        koi_filename: str,
        time_series_dir: str,
    ) -> None:
        self.tce_filename = tce_filename
        self.stellar_filename = stellar_filename
        self.cfp_filename = cfp_filename
        self.koi_filename = koi_filename
        self.time_series_dir = time_series_dir

        self._tce_df: pd.DataFrame = None
        self._koi_df: pd.DataFrame = None
        self._cfp_df: pd.DataFrame = None
        self._stellar_df: pd.DataFrame = None

    @property
    def tce_df(self) -> pd.DataFrame:
        """Threshold-Crossing Event (TCE) pandas DataFrame created from a CSV file."""
        return pd_read_csv(self.tce_filename) if self._tce_df is None else self._tce_df

    @property
    def koi_df(self) -> pd.DataFrame:
        """Kepler Object of Interest (KOI) pandas DataFrame created from a CSV file."""
        return pd_read_csv(self.koi_filename) if self._koi_df is None else self._koi_df

    @property
    def cfp_df(self) -> pd.DataFrame:
        """Certified False Positive pandas DataFrame created from a CSV file."""
        return pd_read_csv(self.cfp_filename) if self._cfp_df is None else self._cfp_df

    @property
    def stellar_df(self) -> pd.DataFrame:
        """Stellar parameters pandas DataFrame created from a CSV file."""
        return pd_read_csv(self.stellar_filename) if self._stellar_df is None else self._stellar_df

    def read_tces(self, *, kepid: int) -> list[TCE]:
        """Read TCEs for the target star.

        Parameters
        ----------
        kepid : int
            The ID of a KOI target

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
        check_kepid(kepid)
        tce_info = self.tce_df[
            [
                "kepid",
                "tce_plnt_num",
                "tce_time0bk",
                "tce_duration",
                "tce_period",
                "tce_cap_stat",
                "tce_hap_stat",
                "boot_fap",
                "tce_rb_tcount0",
                "tce_prad",
                "tcet_period",
            ]
        ]

        tce_info = tce_info.loc[tce_info["kepid"] == kepid]

        if tce_info.empty:
            raise MissingKOI(kepid)

        tces = []

        for _, tce in tce_info.iterrows():
            label, specific_label = self._get_tce_labels(kepid, tce["tce_plnt_num"])
            tce_dict = tce.to_dict()
            tce_dict["label"] = label
            tce_dict["specific_label"] = specific_label
            tces.append(TCE.from_dict(tce_dict))

        return tces

    def read_stellar_params(self, *, kepid: int) -> StellarParameters:
        """Read stellar features of the target star.

        Parameters
        ----------
        kepid : int
            The ID of a KOI target

        Returns
        -------
        StellarParameters
            Stellar properties of the target star

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
        check_kepid(kepid)
        df = self.stellar_df.loc[self.stellar_df["kepid"] == kepid]

        if df.empty:
            raise MissingKOI(kepid)

        return StellarParameters.from_dict(df.to_dict("records")[0])

    def read_time_series(
        self, *, kepid: int, cadence: Cadence, include: Optional[dict[str, str]] = None
    ) -> TimeSeries:
        """Read time series for a specific KOI.

        Parameters
        ----------
        kepid : int
            The ID of KOI target
        cadence : Cadence
            Observation cadence
        include : Optional[dict[str, str]], optional
            A dictionary that tells which fields in the FITS file should be mapped to TimeSeries
            attributes. It must be {FITS field: TimeSeries attribute}. If `None`, all fields are
            mapped without renaming them, by default None

        Returns
        -------
        TimeSeries
            An object which contains attributes from the `include` dict values

        Raises
        ------
        FitsFileError
            An issue with reading a FITS file
        ValueError
            Parameter `kepid` is outside of the range 1-999 999 999
        """
        check_kepid(kepid)
        filenames = get_kepler_filenames(self.time_series_dir, kepid, cadence)
        time_series = defaultdict(lambda: [])

        for path in filenames:
            try:
                file_ts = read_fits_as_dict(path)
            except FileNotFoundError:
                # No file for a specific quarter
                continue

            for ts_field, ts_data in file_ts.items():
                if include is None:
                    mapped_key = ts_field
                elif ts_field in include:
                    mapped_key = include[ts_field]
                else:
                    continue

                time_series[mapped_key].append(ts_data)

        if not time_series:
            # No file found, there are no files for a specified cadence or kepid at all
            raise FitsFileError(
                f"No file found for {kepid=}, {cadence=} at location={self.time_series_dir!r}"
            )

        return TimeSeries(kepid, data=time_series)

    def read_full_data(
        self, *, kepid: int, cadence: Cadence, include: Optional[dict[str, str]] = None
    ) -> KeplerData:
        time_series = self.read_time_series(kepid=kepid, cadence=cadence, include=include)
        stellar_params = self.read_stellar_params(kepid=kepid)
        tces = self.read_tces(kepid=kepid)
        return KeplerData(
            kepid=kepid, time_series=time_series, stellar_params=stellar_params, tces=tces
        )

    def _get_tce_labels(self, kepid: int, tce_num: int) -> tuple[TceLabel, TceSpecificLabel]:
        koi_tces = self._get_koi_by_id_tce_num(kepid, tce_num)

        if not koi_tces.empty and koi_tces["koi_disposition"].item() == "CONFIRMED":
            return TceLabel.PLANET_CANDIDATE, TceSpecificLabel.PLANET_CANDIDATE

        if not koi_tces.empty:
            name = koi_tces["kepoi_name"].item()
            cfp_df = self.cfp_df.loc[self.cfp_df["kepoi_name"] == name]

            if cfp_df["fpwg_disp_status"].item() == "CERTIFIED FP":
                return TceLabel.FALSE_POSITIVE, TceSpecificLabel.ASTROPHYSICAL_FALSE_POSITIVE

        return TceLabel.FALSE_POSITIVE, TceSpecificLabel.NON_TRANSIT_PHENOMENON

    def _get_koi_by_id_tce_num(self, kepid: int, tce_num: int):
        return self.koi_df.loc[
            (self.koi_df["kepid"] == kepid) & (self.koi_df["koi_tce_plnt_num"] == tce_num)
        ]
