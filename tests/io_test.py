from pathlib import Path
from unittest.mock import Mock

import pytest
import tensorflow.python.framework.errors_impl as tf_error

from gaia.io import FileMode, FileSaver, read, write


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
