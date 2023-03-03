import bz2
import gzip
import lzma
import pickle
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock

import numpy as np
import pytest
import tensorflow.python.framework.errors_impl as tf_errors
from astropy.io import fits
from astropy.table import Table

from gaia.io import (
    CsvTableReader,
    FileMode,
    FileSaver,
    ReaderError,
    TimeSeriesPickleReader,
    read,
    read_fits_table,
    write,
)
from tests.conftest import assert_dict_with_numpy_equal


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


TEST_HDU_EXTENSION = "TEST_HEADER"


@pytest.fixture
def fits_file():
    """Return a test FITS file with one extension which contains a table:
    ------
     a  b
    ------
     1  4
     2  5
     3  6
    ------
    """
    data = ([1, 2, 3], [4, 5, 6])
    fields = ("a", "b")
    return fits_content(fields, data, TEST_HDU_EXTENSION)


@pytest.fixture
def fits_open(fits_file, mocker):
    """Mock `gaia.io.fits.open()` to write a test content of `fits_file` fixture."""
    mocker.patch("gaia.io.fits.open", return_value=fits_file)


@pytest.fixture
def read_empty_bytes(mocker):
    """Mock `gaia.io.read()` to return `b''`."""
    mocker.patch("gaia.io.read", return_value=b"")


@pytest.mark.usefixtures("read_empty_bytes", "fits_open")
def test_read_fits_table__invalid_header():
    """Test that `KeyError` is raised when specified HDU extension (header) not found."""
    with pytest.raises(KeyError):
        read_fits_table(TEST_FILEPATH, f"{TEST_HDU_EXTENSION}_INVALID")


@pytest.mark.usefixtures("read_empty_bytes", "fits_open")
def test_read_fits_table__unsupported_fields():
    """Test that `ValueError` is raised when specified fields are not present in a FITS file."""
    with pytest.raises(ValueError, match="Fields .* not present"):
        read_fits_table(TEST_FILEPATH, TEST_HDU_EXTENSION, fields={"a", "c"})


@pytest.mark.usefixtures("read_empty_bytes", "fits_open")
@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"a"}, {"a": np.array([1, 2, 3])}),
        ({"a", "b"}, {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}),
    ],
    ids=["one_of_two", "all"],
)
def test_read_fits_table__read_specific_fields(fields, expected):
    """Test that only specific fields are read from a FITS file."""
    result = read_fits_table(TEST_FILEPATH, TEST_HDU_EXTENSION, fields)
    assert_dict_with_numpy_equal(result, expected)


@pytest.mark.usefixtures("read_empty_bytes", "fits_open")
@pytest.mark.parametrize("fields", [None, set()], ids=["none", "empty_set"])
@pytest.mark.parametrize(
    "expected",
    [{"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}],
    ids=["two_columns_file"],
)
def test_read_fits_table__read_all_fields_by_default(fields, expected):
    """Test that all fields are read from FITS file when `fields` is an empty set or `None`."""
    result = read_fits_table(TEST_FILEPATH, TEST_HDU_EXTENSION, fields)
    assert_dict_with_numpy_equal(result, expected)


TEST_PICKLE_DICT = {"a": 1, "b": 2}


@pytest.fixture
def pickle_dir(tmp_path):
    """Prepare test directory with 'test-file-1.pkl' and 'test-file-2.pkl'. Return a path to it."""
    (tmp_path / "test-file-1.pkl").write_bytes(pickle.dumps(TEST_PICKLE_DICT))
    (tmp_path / "test-file-2.pkl").write_bytes(pickle.dumps(TEST_PICKLE_DICT))
    return tmp_path


@pytest.fixture
def read_pickle(mocker):
    """Mock `gaia.io.read()` to return pickled `TEST_PICKLE_DICT`."""
    mocker.patch("gaia.io.read", return_value=pickle.dumps(TEST_PICKLE_DICT))


@pytest.fixture
def pickle_reader(pickle_dir):
    """Return `PickleReader` instance with data_dir set to directory from `pickle_dir` fixture."""
    return TimeSeriesPickleReader[dict[int, str]](
        data_dir=pickle_dir.as_posix(),
        id_path_pattern="test-file-{id}.pkl",
    )


@pytest.fixture
def is_dir_true(mocker):
    """Mock `gaia.io.tf.io.gfile.isdir()` to return `True`."""
    mocker.patch("gaia.io.tf.io.gfile.isdir", return_value=True)


@pytest.mark.usefixtures("is_dir_true")
def test_pickle_reader_read__id_not_found(pickle_reader):
    """Test that `KeyError` is raised when no file matching the passed ID is found."""
    with pytest.raises(KeyError):
        pickle_reader.read("999")


@pytest.mark.parametrize(
    "is_data_dir_output,read_output",
    [
        (PermissionError(), PermissionError()),
        (True, PermissionError()),
        (False, FileNotFoundError()),
        (Exception(), Exception()),
    ],
    ids=[
        "data_dir_permission_denied",
        "files_permission_denied",
        "data_dir_not_found",
        "generic_is_dir_error",
    ],
)
def test_pickle_reader_read__data_resource_error(
    is_data_dir_output,
    read_output,
    mocker,
    pickle_reader,
):
    """Test that `ReaderError` is raised when data read error occur."""
    mocker.patch("gaia.io.tf.io.gfile.isdir", side_effect=[is_data_dir_output])
    mocker.patch("gaia.io.read", side_effect=[read_output])
    with pytest.raises(ReaderError):
        pickle_reader.read("1")


@pytest.mark.usefixtures("is_dir_true", "read_pickle")
def test_pickle_reader_read__decompression_error():
    """Test that `ReaderError` is raised when data decompression error occur."""
    decompress_fn_mock = Mock(side_effect=Exception("Decompression error"))
    reader = TimeSeriesPickleReader[dict[int, str]](
        data_dir="Sdf",
        id_path_pattern="sdf",
        decompression_fn=decompress_fn_mock,
    )
    with pytest.raises(ReaderError):
        reader.read("1")


@pytest.mark.usefixtures("is_dir_true", "read_pickle")
def test_pickle_reader_read__read_without_decompression(pickle_reader):
    """Test that the correct dictionary is returned without file decompression."""
    result = pickle_reader.read("1")
    assert result == TEST_PICKLE_DICT


@pytest.mark.usefixtures("is_dir_true")
@pytest.mark.parametrize(
    "compression_fn,decompression_fn",
    [
        (gzip.compress, gzip.decompress),
        (bz2.compress, bz2.decompress),
        (lzma.compress, lzma.decompress),
    ],
    ids=["gzip", "bz2", "lzma"],
)
def test_pickle_reader_read__read_with_decompression(compression_fn, decompression_fn, mocker):
    """Test that the correct dictionary is returned with file decompression."""
    compressed_data = compression_fn(pickle.dumps(TEST_PICKLE_DICT))
    mocker.patch("gaia.io.read", return_value=compressed_data)
    reader = TimeSeriesPickleReader[dict[str, int]](
        data_dir="dfsd",
        id_path_pattern="sdfsdf",
        decompression_fn=decompression_fn,
    )
    result = reader.read("1")
    assert result == TEST_PICKLE_DICT


TEST_CSV_TABLE = """
    id,col1,col2
    1,10,20
    2,30,40
"""


@pytest.fixture
def csv_table_path(tmp_path):
    """Return a path to the test CSV file with following content:
    id,col1,col2
    1,10,20
    2,30,40
    """
    path = tmp_path / "test-table.csv"
    path.write_text(dedent(TEST_CSV_TABLE))
    return path.as_posix()


@pytest.fixture
def csv_table_reader(csv_table_path):
    """Return an instance of `CsvTableReader` with test data file."""
    return CsvTableReader(csv_table_path)


@pytest.mark.parametrize(
    "read_error",
    [PermissionError(), FileNotFoundError(), Exception()],
    ids=["permission_denied", "file_not_found", "generic_read_error"],
)
def test_csv_table_reader_read__file_read_error(read_error, csv_table_reader, mocker):
    """Test that `ReaderError` is raised when cannot read the underlying CSV file."""
    mocker.patch("gaia.io.read", side_effect=read_error)
    with pytest.raises(ReaderError):
        csv_table_reader.read()


@pytest.fixture
def read_csv(mocker):
    """Mock `gaia.io.read()` to return `TEST_CSV_TABLE` as bytes."""
    read_mock = mocker.patch("gaia.io.read", return_value=dedent(TEST_CSV_TABLE).encode())
    return read_mock


@pytest.mark.usefixtures("read_csv")
def test_csv_table_reader_read__read_without_mapping(csv_table_reader):
    """Test that a correct object is returned without field names mapping."""
    result = csv_table_reader.read()
    assert result == [{"id": 1, "col1": 10, "col2": 20}, {"id": 2, "col1": 30, "col2": 40}]


@pytest.mark.usefixtures("read_csv")
@pytest.mark.parametrize(
    "mapping,expected",
    [
        (
            {"col1": "column1", "col2": "column2"},
            [{"id": 1, "column1": 10, "column2": 20}, {"id": 2, "column1": 30, "column2": 40}],
        ),
        (
            {"col1": "column1"},
            [{"id": 1, "column1": 10, "col2": 20}, {"id": 2, "column1": 30, "col2": 40}],
        ),
    ],
    ids=["map_all", "map_only_few_keys"],
)
def test_csv_table_reader_read__read_with_mapping(mapping, expected, csv_table_path):
    """Test that a correct object is returned with field names mapping."""
    reader = CsvTableReader(csv_table_path, mapping)
    result = reader.read()
    assert result == expected


def test_csv_table_reader_read__mapping_error(csv_table_path):
    """Test that ReaderError is raised when cannot correctly map dicts keys."""
    reader = CsvTableReader(csv_table_path, {"asdad": "asdasd"})
    with pytest.raises(ReaderError):
        reader.read()
