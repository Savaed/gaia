from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tensorflow.python.framework.errors_impl as tf_errors
from astropy.io import fits
from astropy.table import Table

from gaia.io import FileMode, FileSaver, read, read_fits_table, write


TEST_FILEPATH = "./test-dir/test-file.txt"


@pytest.fixture
def not_found_error(mocker):
    """Mock `tf.io.gfile.GFile.__enter__()` to raise a `NotFoundError`."""
    mocker.patch(
        "gaia.io.tf.io.gfile.GFile.__enter__",
        side_effect=tf_errors.NotFoundError("node", "operation", "msg"),
    )


@pytest.fixture
def permission_error(mocker):
    """Mock `tf.io.gfile.GFile.__enter__()` to raise a `PermissionDeniedError`."""
    mocker.patch(
        "gaia.io.tf.io.gfile.GFile.__enter__",
        side_effect=tf_errors.PermissionDeniedError("node", "operation", "msg"),
    )


@pytest.mark.usefixtures("not_found_error")
@pytest.mark.parametrize("mode", [FileMode.READ, FileMode.READ_BINARY])
def test_read__file_not_found(mode):
    """Test that `FileNotFoundError` is raised when a file is not found."""
    with pytest.raises(FileNotFoundError):
        read(TEST_FILEPATH, mode)


@pytest.mark.usefixtures("permission_error")
@pytest.mark.parametrize("mode", [FileMode.READ, FileMode.READ_BINARY])
def test_read__permission_denied(mode):
    """Test that `PermissionError` is raised when a user has no permission to read."""
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
    """Test that the contents of the file are read correctly."""
    mocker.patch(
        "gaia.io.tf.io.gfile.GFile.__enter__",
        return_value=Mock(**{"read.return_value": expected}),
    )
    result = read(TEST_FILEPATH, mode)
    assert result == expected


@pytest.mark.usefixtures("not_found_error")
@pytest.mark.parametrize("mode,data", [(FileMode.WRITE, "test"), (FileMode.WRITE_BINARY, b"test")])
def test_write__file_not_found(mode, data):
    """Test that `FileNotFoundError` is raised when a file is not found."""
    with pytest.raises(FileNotFoundError):
        write(TEST_FILEPATH, data, mode)


@pytest.mark.usefixtures("permission_error")
@pytest.mark.parametrize("mode,data", [(FileMode.WRITE, "test"), (FileMode.WRITE_BINARY, b"test")])
def test_write__permission_denied(mode, data):
    """Test that `PermissionError` is raised when a user has no permission to write."""
    with pytest.raises(PermissionError):
        write(TEST_FILEPATH, data, mode)


@pytest.mark.parametrize("mode,data", [(FileMode.WRITE, "test"), (FileMode.WRITE_BINARY, b"test")])
def test_write__write_content(mode, data, mocker):
    """Test that the data is written to the file without errors."""
    mocker.patch("gaia.io.tf.io.gfile.GFile.__enter__", return_value=Mock())
    write(TEST_FILEPATH, data, mode)


class TestFileSaver:
    @pytest.fixture
    def local_saver(self, tmp_path):
        """Return an instance of `FileSaver` configured for a local environment."""
        return FileSaver(tables_dir=tmp_path.as_posix(), time_series_dir=tmp_path.as_posix())

    @pytest.fixture(params=[FileNotFoundError, PermissionError])
    def write_error(self, request, mocker):
        """Mock and return `gaia.io.read()` to raise `FileNotFoundError` and `PermissionError`."""
        error = request.param
        mocker.patch("gaia.io.write", side_effect=error)
        return error

    def test_save_table__write_error_occured(self, write_error, local_saver):
        """Test that correct error is raised when write table error occurred."""
        with pytest.raises(write_error):
            local_saver.save_table("tab1.csv", b"test data")

    def test_save_time_series__write_error_occured(self, write_error, local_saver):
        """Test that correct error is raised when  write time series error occurred."""
        with pytest.raises(write_error):
            local_saver.save_time_series("ts1.fits", b"test data")

    @pytest.fixture
    def saver(self, request, local_saver):
        """Return the `FileSaver` instance configured for your local or GCP environment.

        Object configuration is based on the specified data directory path.
        """
        path = request.param
        if path.startswith("gs://"):
            return FileSaver(
                tables_dir=f"{path}/tables",
                time_series_dir=f"{path}/ts",
            )  # pragma: no cover
        return local_saver

    @pytest.mark.parametrize(
        "saver",
        [
            "./local/path",
            pytest.param("gs://gcs/path", marks=pytest.mark.skip(reason="no easy way to test GCS")),
        ],
        indirect=True,
        ids=["local", "GCS"],
    )
    def test_save_table__save_data(self, saver):
        """Test that a table is saved without errors."""
        table = b"test data"
        name = "tab1.csv"
        saver.save_table(name, table)
        saved_table = Path(f"{saver._tables_dir}/{name}").read_bytes()
        assert saved_table == table

    @pytest.mark.parametrize(
        "saver",
        [
            "./local/path",
            pytest.param("gs://gcs/path", marks=pytest.mark.skip(reason="no easy way to test GCS")),
        ],
        indirect=True,
        ids=["local", "GCS"],
    )
    def test_save_time_series__save_data(self, saver):
        """Test that a time series is saved without errors."""
        series = b"test data"
        name = "tab1.csv"
        saver.save_time_series(name, series)
        saved_series = Path(f"{saver._time_series_dir}/{name}").read_bytes()
        assert saved_series == series


def test_read_fits_table__file_not_found(mocker):
    """Test that `FileNotFoundError` is raised when no file found."""
    mocker.patch("gaia.io.read", side_effect=FileNotFoundError())
    with pytest.raises(FileNotFoundError):
        read_fits_table(TEST_FILEPATH, "test")


def test_read_fits_table__permission_denied(mocker):
    """Test that `PermissionError` is raised when the user has no permission to read."""
    mocker.patch("gaia.io.read", side_effect=PermissionError())
    with pytest.raises(PermissionError):
        read_fits_table(TEST_FILEPATH, "test")


def fits_content(fields, data, hdu_extension):
    """Create a `astropy.fits.HDUList` which is a representation of FITS file.

    Include columns and data under specifed HDU extension.
    """
    hdu1 = fits.PrimaryHDU()
    table = Table(names=fields, data=np.stack(data, axis=1))
    hdu2 = fits.BinTableHDU(name=hdu_extension, data=table)
    return fits.HDUList([hdu1, hdu2])
