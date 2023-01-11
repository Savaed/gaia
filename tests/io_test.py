from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tensorflow.python.framework.errors_impl as tf_error
from astropy.io import fits
from astropy.table import Table

from gaia.io import FileMode, FileSaver, read, read_fits_table, write
from tests.conftest import assert_dict_with_numpy_equal


TEST_FILEPATH = "a/b/c.txt"


@pytest.fixture
def not_found_error(mocker):
    """Mock `tf.io.gfile.GFile.__enter__()` to raise a `NotFoundError`."""
    mocker.patch(
        "gaia.io.tf.io.gfile.GFile.__enter__",
        side_effect=tf_error.NotFoundError("node", "operation", "msg"),
    )


@pytest.fixture
def permission_error(mocker):
    """Mock `tf.io.gfile.GFile.__enter__()` to raise a `PermissionDeniedError`."""
    mocker.patch(
        "gaia.io.tf.io.gfile.GFile.__enter__",
        side_effect=tf_error.PermissionDeniedError("node", "operation", "msg"),
    )


@pytest.mark.usefixtures("not_found_error")
@pytest.mark.parametrize("mode", [FileMode.READ, FileMode.READ_BINARY])
def test_read__file_not_found(mode):
    """Test check whether FileNotFoundError is raised when a file is not found."""
    with pytest.raises(FileNotFoundError):
        read(TEST_FILEPATH, mode)


@pytest.mark.usefixtures("permission_error")
@pytest.mark.parametrize("mode", [FileMode.READ, FileMode.READ_BINARY])
def test_read__permission_denied(mode):
    """Test check whether PermissionError is raised when a user has no permission to read."""
    with pytest.raises(PermissionError):
        read(TEST_FILEPATH, mode)


@pytest.mark.parametrize(
    "mode,expected",
    [
        (FileMode.READ, "test"),
        (FileMode.READ_BINARY, b"test"),
    ],
)
def test_read__read_content(mode, expected, mocker):
    """Test check whether the contents of the file are read correctly."""
    mocker.patch(
        "gaia.io.tf.io.gfile.GFile.__enter__",
        return_value=Mock(**{"read.return_value": expected}),
    )
    result = read(TEST_FILEPATH, mode)
    assert result == expected


@pytest.mark.usefixtures("not_found_error")
@pytest.mark.parametrize("mode,data", [(FileMode.WRITE, "test"), (FileMode.WRITE_BINARY, b"test")])
def test_write__file_not_found(mode, data):
    """Test check whether FileNotFoundError is raised when a file is not found."""
    with pytest.raises(FileNotFoundError):
        write(TEST_FILEPATH, data, mode)


@pytest.mark.usefixtures("permission_error")
@pytest.mark.parametrize("mode,data", [(FileMode.WRITE, "test"), (FileMode.WRITE_BINARY, b"test")])
def test_write__permission_denied(mode, data):
    """Test check whether PermissionError is raised when a user has no permission to write."""
    with pytest.raises(PermissionError):
        write(TEST_FILEPATH, data, mode)


@pytest.mark.parametrize("mode,data", [(FileMode.WRITE, "test"), (FileMode.WRITE_BINARY, b"test")])
def test_write__write_content(mode, data, mocker):
    """Test check whether the data is written to the file without errors."""
    mocker.patch("gaia.io.tf.io.gfile.GFile.__enter__", return_value=Mock())
    write(TEST_FILEPATH, data, mode)


class TestFileSaver:
    @pytest.fixture
    def local_saver(self, tmp_path):
        return FileSaver(tables_dir=str(tmp_path), time_series_dir=str(tmp_path))

    @pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
    def test_save_table__error_occured(self, error, local_saver, mocker):
        mocker.patch("gaia.io.write", side_effect=error())
        with pytest.raises(error):
            local_saver.save_table("tab1.csv", b"test data")

    @pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
    def test_save_time_series__error_occured(self, error, local_saver, mocker):
        mocker.patch("gaia.io.write", side_effect=error())
        with pytest.raises(error):
            local_saver.save_time_series("ts1.fits", b"test data")

    @pytest.fixture
    def saver(self, request, local_saver):
        path = request.param
        if path.startswith("gs://"):
            return FileSaver(
                tables_dir=f"{path}/tables",
                time_series_dir=f"{path}/ts",
            )  # pragma: no cover
        return local_saver

    # NOTE: GCS case may be tested by simply asssert that `saver.save_table()` was called
    @pytest.mark.parametrize(
        "saver",
        [
            "./local/path",
            pytest.param("gs://t/est", marks=pytest.mark.skip(reason="no easy way to test GCS")),
        ],
        indirect=True,
        ids=["local", "GCS"],
    )
    def test_save_table__saving_data(self, saver):
        data = b"test data"
        name = "tab1.csv"
        saver.save_table(name, data)
        assert Path(f"{saver._tables_dir}/{name}").read_bytes() == data

    # NOTE: GCS case may be tested by simply asssert that `saver.save_time_series()` was called
    @pytest.mark.parametrize(
        "saver",
        [
            "./local/path",
            pytest.param("gs://t/est", marks=pytest.mark.skip(reason="no easy way to test GCS")),
        ],
        indirect=True,
        ids=["local", "GCS"],
    )
    def test_save_time_series__saving_data(self, saver):
        data = b"test data"
        name = "tab1.csv"
        saver.save_time_series(name, data)
        assert Path(f"{saver._time_series_dir}/{name}").read_bytes() == data


def test_read_fits_table__file_not_found(mocker):
    """Test check whether FileNotFoundError is raised when no file found."""
    mocker.patch("gaia.io.read", side_effect=FileNotFoundError())
    with pytest.raises(FileNotFoundError):
        read_fits_table(TEST_FILEPATH, "test")


def test_read_fits_table__permission_denied(mocker):
    """Test check whether PermissionError is raised when the user has no access to the file."""
    mocker.patch("gaia.io.read", side_effect=PermissionError())
    with pytest.raises(PermissionError):
        read_fits_table(TEST_FILEPATH, "test")


def fits_content(fields, data, hdu_extension):
    hdu1 = fits.PrimaryHDU()
    table = Table(names=fields, data=np.stack(data, axis=1))
    hdu2 = fits.BinTableHDU(name=hdu_extension, data=table)
    return fits.HDUList([hdu1, hdu2])


TEST_HDU_EXTENSION = "TEST_HEADER"


@pytest.fixture
def fits_file():
    """Return a test FITS file with one extension which is a table with columns 'a', 'b'.

    The extension name is specified in the `TEST_HDU_EXTENSION` constant.

    The table is as follows:
    +-------+
    | a | b |
    +=======+
    | 1 | 4 |
    | 2 | 5 |
    | 3 | 6 |
    +-------+
    """
    data = ([1, 2, 3], [4, 5, 6])
    fields = ("a", "b")
    return fits_content(fields, data, TEST_HDU_EXTENSION)


def test_read_fits_table__invalid_header(mocker, fits_file):
    """Test check whether KeyError is raised when specified HDU extension (header) not found."""
    mocker.patch("gaia.io.read", return_value=b"")
    mocker.patch("gaia.io.fits.open", return_value=fits_file)
    with pytest.raises(KeyError):
        read_fits_table(TEST_FILEPATH, f"{TEST_HDU_EXTENSION}_INVALID")


def test_read_fits_table__unsupported_fields(fits_file, mocker):
    """Test check whether ValueError is raised when specified fields are not present in a file."""
    mocker.patch("gaia.io.read", return_value=b"")
    mocker.patch("gaia.io.fits.open", return_value=fits_file)
    with pytest.raises(ValueError, match="Fields .* not present"):
        read_fits_table(TEST_FILEPATH, TEST_HDU_EXTENSION, fields={"a", "c"})


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"a"}, {"a": np.array([1, 2, 3])}),
        ({"a", "b"}, {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}),
    ],
    ids=["one_of_two", "all"],
)
def test_read_fits_table__read_specific_fields(fields, expected, fits_file, mocker):
    """Test check whether only specific fields are read from a file."""
    mocker.patch("gaia.io.read", return_value=b"")
    mocker.patch("gaia.io.fits.open", return_value=fits_file)
    result = read_fits_table(TEST_FILEPATH, TEST_HDU_EXTENSION, fields)
    assert_dict_with_numpy_equal(result, expected)


@pytest.mark.parametrize("fields", [None, set()], ids=["none", "empty_set"])
@pytest.mark.parametrize(
    "expected",
    [{"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}],
    ids=["two_columns_file"],
)
def test_read_fits_table__read_all_fields_by_default(fields, expected, fits_file, mocker):
    """Test check whether all fields are read when `fields` is an empty set or None."""
    mocker.patch("gaia.io.read", return_value=b"")
    mocker.patch("gaia.io.fits.open", return_value=fits_file)
    result = read_fits_table(TEST_FILEPATH, TEST_HDU_EXTENSION, fields)
    assert_dict_with_numpy_equal(result, expected)
