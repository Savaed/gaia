from pathlib import Path

import numpy as np
import pytest
from astropy.io.fits.column import Column
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.header import Header

from gaia.io import FileSaver, read_fits
from tests.conftest import assert_dict_with_numpy_equal


@pytest.fixture
def saver(tmp_path):
    """Return an instance of `FileSaver`."""
    return FileSaver(tables_dir=tmp_path.as_posix(), time_series_dir=tmp_path.as_posix())


@pytest.fixture()
def path_error(mocker):
    """Factory function to mock `Path.write_bytes()` to raise a specific error."""

    def _mock_error(error):
        mocker.patch.object(Path, "write_bytes", side_effect=error())

    return _mock_error


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
def test_save_table__write_error_occured(error, saver, path_error):
    """Test that correct error is raised when write table error occurred."""
    path_error(error)
    with pytest.raises(error):
        saver.save_table("tab1.csv", b"test data")


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
def test_save_time_series__write_error_occured(error, saver, path_error):
    """Test that correct error is raised when  write time series error occurred."""
    path_error(error)
    with pytest.raises(error):
        saver.save_time_series("ts1.fits", b"test data")


def test_save_table__save_data(saver):
    """Test that a table is correctly saved."""
    table = b"test data"
    name = "tab1.csv"
    saver.save_table(name, table)
    saved_table = Path(f"{saver._tables_dir}/{name}").read_bytes()
    assert saved_table == table


def test_save_time_series__save_data(saver):
    """Test that a time series is correctly saved."""
    series = b"test data"
    name = "tab1.csv"
    saver.save_time_series(name, series)
    saved_series = Path(f"{saver._time_series_dir}/{name}").read_bytes()
    assert saved_series == series


@pytest.fixture
def fits_header():
    """Return `astropy.io.fits.header.Header` with test metadata."""
    # Keys must be uppercase. See `astropy.io.fits.card.Card.normalize_keyword()` for more details.
    return Header({"A": 1, "B": "xyz"})


@pytest.fixture
def fits_data():
    """Return `astropy.io.fits.fitsrec.Fits_rec` with two columns 'x' and 'y' test tabular data."""
    return FITS_rec.from_columns(
        [
            Column(name="x", format="D", array=np.array([1, 2])),
            Column(name="y", format="E", array=np.array([3.0, 4.0])),
        ],
    )


def test_read_fits__file_not_found(mocker):
    """Test that `FileNotFoundError` is raised when no file was found."""
    mocker.patch("gaia.io.fits.getdata", side_effect=FileNotFoundError())
    with pytest.raises(FileNotFoundError):
        read_fits("test/file.fits", "HEADER", meta=())


def test_read_fits__data_header_not_found(mocker):
    """Test that `KeyError` is raised when no data HDU header was found."""
    mocker.patch("gaia.io.fits.getdata", side_effect=KeyError())
    with pytest.raises(KeyError):
        read_fits("test/file.fits", "HEADER", meta=())


def test_read_fits__data_column_not_found(mocker):
    """Test that `KeyError` is raised when any of data column were not found."""
    mocker.patch("gaia.io.fits.getdata", side_effect=KeyError())
    with pytest.raises(KeyError):
        read_fits("test/file.fits", "HEADER", columns={"x"}, meta=())


def test_read_fits__metadata_column_not_found(mocker):
    """Test that `KeyError` is raised when any of metadata column were not found."""
    mocker.patch("gaia.io.fits.getheader", side_effect=KeyError())
    with pytest.raises(KeyError):
        read_fits("test/file.fits", "HEADER", meta={"not_existent_column"})


@pytest.mark.parametrize("meta", [None, (), [], set()], ids=["none", "tuple", "list", "set"])
def test_read_fits__meta_empty_or_none(meta, mocker, fits_data):
    """Test that metadata is not read from the FITS file header."""
    mocker.patch("gaia.io.fits.getdata", return_value=fits_data)
    getheader_mock = mocker.patch("gaia.io.fits.getheader")
    read_fits("test/file.fits", data_header="HEADER", columns={"x"}, meta=meta)
    assert getheader_mock.call_count == 0


@pytest.mark.parametrize(
    "meta_columns,expected",
    [
        ({"A"}, {"A": 1}),
        ({"A", "B"}, {"A": 1, "B": "xyz"}),
    ],
    ids=["single_column", "all_columns"],
)
def test_read_fits__return_correct_meta(meta_columns, expected, mocker, fits_header):
    """Test that metadata is read correctly."""
    mocker.patch("gaia.io.fits.getheader", return_value=fits_header)
    mocker.patch("gaia.io.fits.getdata")
    result = read_fits("test/file.fits", data_header="HEADER", meta=meta_columns)
    assert result == expected


@pytest.mark.parametrize("columns", [(), [], set()], ids=["tuple", "list", "set"])
def test_read_fits__empty_columns(columns, mocker):
    """Test that no data is read when parameter `columns` is an empty sequence."""
    getdata_mock = mocker.patch("gaia.io.fits.getdata")
    read_fits("test/file.fits", data_header="HEADER", columns=columns, meta=())
    assert getdata_mock.call_count == 0


@pytest.mark.parametrize(
    "columns,expected",
    [
        (None, {"x": np.array([1, 2]), "y": np.array([3.0, 4.0])}),
        ({"x"}, {"x": np.array([1, 2])}),
        ({"x", "y"}, {"x": np.array([1, 2]), "y": np.array([3.0, 4.0])}),
        ({}, {}),
    ],
    ids=["all_columns_by_default", "single_column", "two_columns", "no_data"],
)
def test_read_fits__return_correct_data(columns, expected, mocker, fits_data):
    """Test that correct data is returned."""
    mocker.patch("gaia.io.fits.getdata", return_value=fits_data)
    result = read_fits("test/file.fits", data_header="HEADER", columns=columns)
    assert_dict_with_numpy_equal(result, expected)
