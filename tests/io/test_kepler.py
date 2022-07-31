from collections import namedtuple
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.table import Table
from numpy.testing import assert_array_almost_equal

from gaia.data.models import TCE, StellarParameters
from gaia.enums import Cadence
from gaia.io.kepler import (
    FitsFileError,
    MissingKOI,
    get_kepler_filenames,
    read_fits_as_dict,
    read_stellar_params,
    read_tce,
    read_time_series,
)
from tests.conftest import create_df


# TODO: Use fixtures to bound test data and expected result if it's a complex object.
# Use dataclass to simplify test cases code.
# Move fixtures on the top of a file to separate test data from test logic.
# Use mocker fixture from pytest-mock to simplify code? Use patch("xxx", return_value=...)?

FitsFile = namedtuple("FitsFile", ["file", "fields", "time_series"])


@pytest.fixture(name="invalid_kepid", params=[-1, 0, 1_000_000_000])
def fixture_invalid_kepid(request):
    return request.param


@pytest.fixture(name="fits_file", params=[(["A", "B"], ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]))])
def fixture_fits_file(request):
    fields, data = request.param
    hdu1 = fits.PrimaryHDU()
    ts_data = np.stack(data, axis=1)
    time_series = Table(names=fields, data=ts_data)
    hdu2 = fits.BinTableHDU(name="LIGHTCURVE", data=time_series)
    hdu3 = fits.ImageHDU(name="APERTURE")
    return FitsFile(fits.HDUList([hdu1, hdu2, hdu3]), fields, ts_data)


@pytest.fixture(name="quarter_prefixes")
def fixture_quarter_prefixes():
    return ["2009201121230", "2009231120729", "2009259162342"]


@patch("gaia.io.kepler.mp_read")
def test_read_fits_as_dict__missing_file(mock_read: Mock):
    """Test check whether FileNotFoundError is raised when a file is missing."""
    mock_read.side_effect = FileNotFoundError()

    with pytest.raises(FileNotFoundError):
        read_fits_as_dict("test.fits")


@patch("gaia.io.kepler.mp_read")
def test_read_fits_as_dict__fits_reading_error(mock_read: Mock):
    """Test check whether FitsFileError is raised when cannot read an existing FITS file."""
    mock_read.side_effect = Exception()

    with pytest.raises(FitsFileError):
        read_fits_as_dict("test.fits")


@patch("gaia.io.kepler.mp_read")
@patch("gaia.io.kepler.io.BytesIO")
@patch("gaia.io.kepler.fits.open")
def test_read_fits_as_dist__fits_fields_mapping(
    mock_fits_open: Mock,
    mock_io: Mock,
    mock_read: Mock,
    fits_file: FitsFile,
):
    """Test check whether all read FITS file fields are mapped to dictionary keys."""
    mock_fits_open.return_value = fits_file.file
    result = read_fits_as_dict("x.fits")
    assert set(result.keys()) == set(fits_file.fields)


@patch("gaia.io.kepler.mp_read")
@patch("gaia.io.kepler.io.BytesIO")
@patch("gaia.io.kepler.fits.open")
def test_read_fits_as_dict__return_correct_dict(
    mock_fits_open: Mock, mock_io: Mock, mock_read: Mock, fits_file: FitsFile
):
    """Test check whether a correct dictionary is returned after reading a FITS file"""
    mock_fits_open.return_value = fits_file.file

    result = read_fits_as_dict("x.fits")

    for result_data, expected in zip(result.values(), fits_file.time_series.T):
        assert_array_almost_equal(result_data, expected)


def test_get_kepler_filenames__invalid_kepid(invalid_kepid: int):
    """Test check whether ValueError is raised when `kepid` is outside the range 1-999 999 999."""
    with pytest.raises(ValueError):
        get_kepler_filenames(kepid=invalid_kepid, cadence=Cadence.LONG, data_dir="test/")


@pytest.mark.parametrize("cadence", [Cadence.LONG, Cadence.SHORT])
@patch("gaia.io.kepler.get_quarter_prefixes")
def test_get_kepler_filenames__return_correct_filenames_format(
    mock_get_quarter_prefixes: Mock, cadence: Cadence, quarter_prefixes: list[str]
):
    """Test check whether returned filenames are of the format
    `{data_dir}/{kepid:09}/kplr{kepid:09}-{quarter_prefix}_{cadence}.fits`."""
    test_dir = "kepler/test/dir"
    kepid = 12345678
    kepid_str = f"{kepid:09d}"
    expected = [
        f"{test_dir}/{kepid_str}/kplr{kepid_str}-{prefix}_{cadence.value}.fits"
        for prefix in quarter_prefixes
    ]
    mock_get_quarter_prefixes.return_value = quarter_prefixes

    result = get_kepler_filenames(test_dir, kepid, cadence)

    assert result == expected


def test_read_tce__invalid_kepid(invalid_kepid: int):
    """Test check whether ValueError is raised when `kepid` is outside the range 1-999 999 999."""
    with pytest.raises(ValueError):
        read_tce("test.csv", invalid_kepid)


@patch("gaia.io.kepler.mp_read")
def test_read_tce__missing_file(mock_read: Mock):
    """Test check whether FileNotFoundError is raised when a file is missing."""
    mock_read.side_effect = FileNotFoundError()

    with pytest.raises(FileNotFoundError):
        read_tce("test.csv")


@pytest.fixture(name="tce_df")
def fixture_tce_df(request) -> pd.DataFrame:
    """Return a test pd.DataFrame with missing key."""
    records = [
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
    data = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]

    if hasattr(request, "param"):
        missing_key = request.param
        del records[0][missing_key]
        data = np.delete(data, missing_key - 1, axis=1)

    records.extend(list(map(tuple, data)))
    return create_df(records)


@pytest.mark.parametrize(
    "tce_df",
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ids=[
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
    ],
    indirect=True,
)
@patch("gaia.io.kepler._read_csv_as_df")
def test_read_tce__missing_required_csv_column(mock_read_csv: Mock, tce_df: pd.DataFrame):
    """Test check whether KeyError is raised when a required CSV column is missing."""
    mock_read_csv.return_value = tce_df

    with pytest.raises(KeyError):
        read_tce("x.csv")


@patch("gaia.io.kepler._read_csv_as_df")
def test_read_tce__no_tce_for_specific_kepid(read_mock: Mock, tce_df: pd.DataFrame):
    """Test check whether MissingKOI is raised when no TCE found for a specified `kepid`."""
    read_mock.return_value = tce_df

    with pytest.raises(MissingKOI):
        read_tce("x.csv", kepid=99)


@patch("gaia.io.kepler._read_csv_as_df")
def test_read_tce__filter_kepid(mock_read_csv: Mock, tce_df: pd.DataFrame):
    """Test check whether TCEs are filtered when specified `kepid`."""
    mock_read_csv.return_value = tce_df
    kepid = 1
    expected = expected = [
        TCE.from_dict(x.to_dict()) for _, x in tce_df.iterrows() if x["kepid"] == kepid
    ]

    result = read_tce("x.csv", kepid)

    assert result == expected


@patch("gaia.io.kepler._read_csv_as_df")
def test_read_tce__no_filter_kepid(read_mock: Mock, tce_df: pd.DataFrame):
    """Test check whether TCEs are NOT filtered when `kepid` not specified."""
    read_mock.return_value = tce_df
    expected = expected = [TCE.from_dict(x.to_dict()) for _, x in tce_df.iterrows()]

    result = read_tce("x.csv")

    assert result == expected


def test_read_stellar_params__invalid_kepid(invalid_kepid: int):
    """Test check whether ValueError is raised when `kepid` is outside the range 1-999 999 999."""
    with pytest.raises(ValueError):
        read_stellar_params("x.csv", kepid=invalid_kepid)


@patch("gaia.io.kepler._read_csv_as_df")
def test_read_stellar_params__missing_file(mock_read_csv: Mock):
    """Test check whether FileNotFoundError is raised when a file is missing."""
    mock_read_csv.side_effect = FileNotFoundError()

    with pytest.raises(FileNotFoundError):
        read_stellar_params("x.csv")


@pytest.fixture(name="star_df")
def fixture_star_df(request) -> pd.DataFrame:
    records = [["kepid", "teff", "radius", "mass", "dens", "logg", "feh"]]
    data = [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [2, 2, 3, 4, 5, 6, 7]]

    if hasattr(request, "param"):
        missing_key = request.param
        del records[0][missing_key]
        data = np.delete(data, missing_key - 1, axis=1)

    records.extend(list(map(tuple, data)))
    return create_df(records)


@patch("gaia.io.kepler._read_csv_as_df")
def test_read_stellar_params__filter_kepid(mock_read_csv: Mock, star_df: pd.DataFrame):
    """Test check whether TCEs are filtered when specified `kepid`."""
    kepid = 1
    mock_read_csv.return_value = star_df
    expected = [
        StellarParameters.from_dict(x.to_dict())
        for _, x in star_df.iterrows()
        if x["kepid"] == kepid
    ]

    result = read_stellar_params("x.csv", kepid)

    assert result == expected


@patch("gaia.io.kepler._read_csv_as_df")
def test_read_stellar_params__no_filter_kepid(mock_read_csv: Mock, star_df: pd.DataFrame):
    """Test check whether TCEs are NOT filtered `kepid` not specified."""
    kepid = 1
    mock_read_csv.return_value = star_df
    expected = expected = [
        StellarParameters.from_dict(x.to_dict())
        for _, x in star_df.iterrows()
        if x["kepid"] == kepid
    ]

    result = read_stellar_params("x.csv", kepid)

    assert result == expected


@patch("gaia.io.kepler._read_csv_as_df")
def test_read_stellar_params__no_params_for_specific_kepid(
    mock_read_csv: Mock, star_df: pd.DataFrame
):
    """Test check whether MissingKOI is raised when no stellar params found for a specified `kepid`."""
    mock_read_csv.return_value = star_df

    with pytest.raises(MissingKOI):
        read_stellar_params("x.csv", kepid=99)


@pytest.mark.parametrize(
    "star_df",
    [1, 2, 3, 4, 5, 6],
    ids=["teff", "radius", "mass", "dens", "logg", "feh"],
    indirect=True,
)
@patch("gaia.io.kepler._read_csv_as_df")
def test_read_stellar_params__missing_required_csv_column(
    mock_read_csv: Mock, star_df: pd.DataFrame
):
    """Test check whether KeyError is raised when a required CSV column is missing."""
    mock_read_csv.return_value = star_df

    with pytest.raises(KeyError):
        read_stellar_params("x.csv")


@pytest.fixture(name="fits_dict")
def fixture_fits_dict():
    return {"TIME": np.array([1, 2, 3, 4, 5]), "FLUX": np.array([6, 7, 8, 9, 10])}


@patch("gaia.io.kepler.get_kepler_filenames")
@patch("gaia.io.kepler.read_fits_as_dict")
def test_read_time_series__missing_files(mock_read_fits: Mock, mock_get_kplr_filenames: Mock):
    """Test check whether Fits File Error is raised when no file(s) found for a specified `cadence` and `kepid`."""
    mock_get_kplr_filenames.return_value = ["a.fits", "b.fits"]
    mock_read_fits.side_effect = [FileNotFoundError(), FileNotFoundError()]

    with pytest.raises(FitsFileError):
        read_time_series(123, "test/", Cadence.LONG)


def test_read_time_series__invalid_kepid(invalid_kepid: int):
    """Test check whether ValueError is raised when `kepid` is outside the range 1-999 999 999."""
    with pytest.raises(ValueError):
        read_time_series(invalid_kepid, "test/", Cadence.LONG)


@pytest.mark.parametrize("n_segments", [1, 5], ids=["single_segment", "multi_segments"])
@patch("gaia.io.kepler.get_kepler_filenames")
@patch("gaia.io.kepler.read_fits_as_dict")
def test_read_time_series__map_dict(
    mock_read_fits: Mock, mock_get_kplr_filenames: Mock, n_segments: int, fits_dict
):
    """Test check whether a correct `TimeSeries` object is returned after reading a FITS file."""
    mock_get_kplr_filenames.return_value = n_segments * ["test-file.fits"]
    fits_read_data = n_segments * [fits_dict]
    mock_read_fits.side_effect = fits_read_data
    include = {field: f"{field}_map" for field in fits_dict}
    expected_attrs = set(include.values())
    expected_data = {attr: [(x[attr]) for x in fits_read_data] for attr in include}

    result = read_time_series(1, "test/", Cadence.LONG, include=include)

    assert expected_attrs.issubset(set(dir(result)))

    for attr in include:
        assert getattr(result, include[attr]) == expected_data[attr]
