import bz2
import gzip
import lzma
import pickle
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock

import numpy as np
import pytest
import tensorflow.python.framework.errors_impl as tf_error
from astropy.io import fits
from astropy.table import Table

from gaia.io import CsvTableReader, FileMode, FileSaver, PickleReader, read, read_fits_table, write
from tests.conftest import assert_dict_with_numpy_equal, create_df


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
    """Return a test FITS file with one extension which is a table:
    +-------+
    | a | b |
    +=======+
    | 1 | 4 |
    | 2 | 5 |
    | 3 | 6 |
    +-------+
    The extension name is specified in the `TEST_HDU_EXTENSION` constant.
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


TEST_PICKLE_DICT = {"a": 1, "b": 2}


@pytest.fixture
def pickle_dir(tmp_path):
    """Prepare and return temporary test directory with pkl files."""
    (tmp_path / "test-file-1.pkl").write_bytes(pickle.dumps(TEST_PICKLE_DICT))
    (tmp_path / "test-file-2.pkl").write_bytes(pickle.dumps(TEST_PICKLE_DICT))
    return tmp_path


@pytest.fixture
def pickle_reader(pickle_dir):
    """Return a `PickleReader` instance that search files in the `pickle_dir` directory."""
    return PickleReader[dict[int, str]](
        data_dir=pickle_dir.as_posix(),
        id_path_pattern="test-file-{id}.pkl",
    )


def test_pickle_read__id_not_found(pickle_reader):
    """Test that FileNotFoundError is raised when no file matching the passed ID is found."""
    with pytest.raises(FileNotFoundError):
        pickle_reader.read("999")


def test_pickle_read__read_depickled_file_without_decompression(pickle_reader):
    """Test that the correct dictionary is returned without file decompression."""
    result = pickle_reader.read("1")
    assert result == [TEST_PICKLE_DICT]


@pytest.mark.parametrize(
    "compression_fns,file_extension",
    [
        ((gzip.compress, gzip.decompress), "gz"),
        ((bz2.compress, bz2.decompress), "bz2"),
        ((lzma.compress, lzma.decompress), "xz"),
    ],
    ids=["gzip", "bz2", "lzma"],
)
def test_pickle_read__read_depickled_file_with_decompression(
    compression_fns,
    file_extension,
    tmp_path,
):
    """Test that the correct dictionary is returned with file decompression."""
    compress_fn, decompress_fn = compression_fns
    data = compress_fn(pickle.dumps(TEST_PICKLE_DICT))
    (tmp_path / f"test-file-1.{file_extension}").write_bytes(data)
    reader = PickleReader[dict[str, int]](
        tmp_path.as_posix(),
        f"test-file-{{id}}.{file_extension}",
        decompress_fn,
    )
    result = reader.read("1")
    assert result == [TEST_PICKLE_DICT]


TEST_CSV_TABLE = """
    id,col1,col2
    1,10,20
    2,30,40
"""


@pytest.fixture
def csv_df():
    """Return test `pd.DataFrame` table:
    +----+------+------+
    | id | col1 | col2 |
    +====+======+======+
    | 1  | 10   | 20   |
    | 2  | 30   | 40   |
    +----+------+------+
    """
    return create_df((("id", "col1", "col2"), (1, 10, 20), (2, 30, 40)))


@pytest.fixture
def csv_table_path(tmp_path):
    """Return a path to the test CSV file:
    id,col1,col2
    1,10,20
    2,30,40
    """
    path = tmp_path / "test-table.csv"
    path.write_text(dedent(TEST_CSV_TABLE))
    return path


@pytest.fixture
def csv_table_reader(csv_table_path):
    """Return an instance of `CsvTableReader` with test data file."""
    return CsvTableReader[dict[str, int]](csv_table_path.as_posix(), "id")  # type: ignore


def test_csv_table_read__file_not_found(csv_table_reader):
    """Test that FileNotFoundError is raised when no source CSV file is found."""
    csv_table_reader._source = "sdfsdf"
    with pytest.raises(FileNotFoundError):
        csv_table_reader.read("1")


def test_csv_table_reader_red__id_not_found(csv_table_reader, mocker, csv_df):
    """Test that KeyError is raised when no object matching the passed ID is found."""
    mocker.patch("gaia.io.pd.read_csv", return_value=csv_df)
    with pytest.raises(KeyError):
        csv_table_reader.read("999")


def test_csv_table_reader_red__return_correct_dict_without_mapping(
    csv_table_reader,
    csv_df,
    mocker,
):
    """Test that a correct object is returned without field names mapping."""
    mocker.patch("gaia.io.pd.read_csv", return_value=csv_df)
    result = csv_table_reader.read("1")
    assert result == [{"id": 1, "col1": 10, "col2": 20}]


@pytest.mark.parametrize(
    "id_,mapping,expected",
    [
        ("1", {"col1": "column1", "col2": "column2"}, [{"id": 1, "column1": 10, "column2": 20}]),
        ("1", {"col1": "column1"}, [{"id": 1, "column1": 10, "col2": 20}]),
    ],
    ids=["map-all", "map-only-few-keys"],
)
def test_csv_table_reader_red__return_correct_dict_with_mapping(
    id_,
    mapping,
    expected,
    csv_table_path,
    mocker,
    csv_df,
):
    """Test that a correct object is returned with field names mapping."""
    mocker.patch("gaia.io.pd.read_csv", return_value=csv_df)
    r = CsvTableReader[dict[str, int]](csv_table_path.as_posix(), "id", mapping)  # type: ignore
    result = r.read(id_)
    assert result == expected


def test_csv_table_reader_read__read_underlying_data_only_once(csv_table_reader, mocker, csv_df):
    """Check that underlying data is only read once."""
    pd_read_csv_mock = mocker.patch("gaia.io.pd.read_csv", return_value=csv_df)
    csv_table_reader.read("1")
    csv_table_reader.read("2")
    pd_read_csv_mock.assert_called_once()
