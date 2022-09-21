"""Unit tests for `gaia.io` module."""

# pylint: disable=redefined-outer-name

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from astropy.io import fits
from astropy.table import Table
from conftest import create_df
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from pytest_mock import MockerFixture
from tensorflow.python.framework.errors import NotFoundError  # pylint: disable=no-name-in-module
from typing_extensions import TypeAlias

from gaia import io
from gaia.enums import Cadence


@dataclass(frozen=True)
class GetQuartersCase:
    """Test case for `get_quarter_prefixes` function."""

    cadence: Cadence
    const_name: str
    prefixes: tuple[list[str], ...]
    expected_prefixes: list[str]


@pytest.mark.parametrize(
    "case",
    [
        GetQuartersCase(
            cadence=Cadence.LONG,
            const_name="_LONG_CADENCE_QUARTER_PREFIXES",
            prefixes=(["1"], ["2", "3"]),
            expected_prefixes=["1", "2", "3"],
        ),
        GetQuartersCase(
            cadence=Cadence.SHORT,
            const_name="_SHORT_CADENCE_QUARTER_PREFIXES",
            prefixes=(["4"], ["5", "6"]),
            expected_prefixes=["4", "5", "6"],
        ),
    ],
    ids=["cadence_long", "cadence_short"],
)
def test_get_quarter_prefixes__return_correct_list(
    case: GetQuartersCase, monkeypatch: MonkeyPatch
) -> None:
    """Test check whether a correct list of quarter prefixes is returned."""
    monkeypatch.setattr(io, case.const_name, case.prefixes)
    result = io.get_quarter_prefixes(case.cadence)
    assert result == case.expected_prefixes


@pytest.mark.parametrize("kepid", [0, 1_000_000_000])
def test_get_kepler_fits_paths__invalid_kepid(kepid: int) -> None:
    """
    Test check whether A ValueError is raised when `kepid` is outside of the range
    [1, 999 999 999].
    """
    with pytest.raises(ValueError):
        io.get_kepler_fits_paths("a/b/c", kepid, Cadence.LONG)


@dataclass(frozen=True)
class FITSPathsCase:
    """Test case for `get_kepler_fits_paths` function."""

    kepid: int
    cadence: Cadence
    data_dir: str
    filter_quarters: tuple[list[str], ...]
    quarters_prefixes: list[str]
    expected: list[str]


@pytest.mark.parametrize(
    "case",
    [
        FITSPathsCase(
            kepid=1,
            cadence=Cadence.LONG,
            data_dir="a/b/c",
            filter_quarters=("1",),
            quarters_prefixes=["1", "2", "3"],
            expected=["a/b/c/000000001/kplr000000001-1_llc.fits"],
        ),
        FITSPathsCase(
            kepid=111,
            cadence=Cadence.SHORT,
            data_dir="a/b/c",
            filter_quarters=("1", "3"),
            quarters_prefixes=["1", "2", "3"],
            expected=[
                "a/b/c/000000111/kplr000000111-1_slc.fits",
                "a/b/c/000000111/kplr000000111-3_slc.fits",
            ],
        ),
    ],
    ids=["single_quarter", "multiple_quarters"],
)
def test_get_kepler_fits_paths__filter_quarters(
    case: FITSPathsCase,
    mocker: MockerFixture,
) -> None:
    """Test check whether quarter prefixes are properly filtered based on a `quarters` parameter."""
    mocker.patch("gaia.io.get_quarter_prefixes", return_value=case.quarters_prefixes)
    result = io.get_kepler_fits_paths(
        case.data_dir, case.kepid, case.cadence, quarters=case.filter_quarters
    )
    assert set(result) == set(case.expected)


class TestDataFrameReader:
    """Unit tests for `DataFrameReader` class."""

    def test_read__file_not_found(self, mocker: MockerFixture) -> None:
        """Test check whether a FileNotFoundError is raised when there is no file."""
        mocker.patch("gaia.io.tf.io.gfile.GFile", side_effect=NotFoundError(None, None, None))

        with pytest.raises(FileNotFoundError):
            io.DataFrameReader().read("not/existent/file.csv")

    @pytest.fixture
    def test_df(self) -> pd.DataFrame:
        """Create a simple test pandas DataFrame."""
        return create_df([("A", "B"), (1, 2), (3, 4)])

    def test_read__return_correct_data_frame(
        self, test_df: pd.DataFrame, mocker: MockerFixture
    ) -> None:
        """Test check whether a correct pandas DataFrame is returned."""
        mocker.patch("gaia.io.pd.read_csv", return_value=test_df)
        result = io.DataFrameReader().read("x/y/z.csv")
        assert_frame_equal(result, test_df)


class TestFITSTimeSeriesReader:
    """Unit tests for `FITSTimeSeriesReader` class."""

    FITSFile: TypeAlias = tuple[fits.HDUList, list[str], np.ndarray]

    def test_read__unexpected_error_occurs(self, mocker: MockerFixture) -> None:
        """Test check whether a FITSReadingError is raised when there is an unexpected error."""
        mocker.patch("gaia.io.fits.open", side_effect=Exception())
        mocker.patch("gaia.io.tf.io.gfile.GFile", side_effect=Exception())

        with pytest.raises(io.FITSReadingError):
            io.FITSTimeSeriesReader().read("not/existen/file.fits")

    def test_read__file_not_found(self, mocker: MockerFixture) -> None:
        """Test check whether a FileNotFoundError is raised when no file is found."""
        mocker.patch("gaia.io.tf.io.gfile.GFile", side_effect=NotFoundError(None, None, None))

        with pytest.raises(FileNotFoundError):
            io.FITSTimeSeriesReader().read("not/existen/file.fits")

    def test_read__no_specified_extension_hdu(self, mocker: MockerFixture) -> None:
        """Test check whether a MissingExtensionHDUError is raised when no extension HDU is found."""
        mocker.patch("gaia.io.fits.open", side_effect=KeyError())

        with pytest.raises(io.MissingExtensionHDUError):
            io.FITSTimeSeriesReader().read("existen/file.fits")

    @pytest.fixture
    def fits_file(self, request: SubRequest) -> FITSFile:
        """Create FITS file content with columns and records."""
        fields: list[str]
        data = tuple[list[str], ...]
        fields, data = request.param
        hdu1 = fits.PrimaryHDU()
        ts_data = np.stack(data, axis=1)
        time_series = Table(names=fields, data=ts_data)
        hdu2 = fits.BinTableHDU(name="LIGHTCURVE", data=time_series)
        hdu3 = fits.ImageHDU(name="APERTURE")
        return fits.HDUList([hdu1, hdu2, hdu3]), fields, np.array(data)

    @pytest.mark.parametrize(
        "fits_file",
        [
            (["A", "B"], ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])),
            (["A"], ([[1, 2, 3, 4, 5]])),
        ],
        indirect=True,
        ids=["one_column", "two_columns"],
    )
    def test_read__return_correct_dict(self, mocker: MockerFixture, fits_file: FITSFile) -> None:
        """Test check whether a correct dict is returned after reading the FITS file."""
        hdu_content, fields, data = fits_file
        mocker.patch("gaia.io.fits.open", return_value=hdu_content)
        expected = dict(zip(fields, data))

        result = io.FITSTimeSeriesReader().read("a/b/c.fits")

        assert result.keys() == expected.keys()
        assert_array_equal(list(result.values()), list(expected.values()))
