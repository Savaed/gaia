from pathlib import Path

import pytest

from gaia.io import FileSaver


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
