import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pytest
from astropy.io.fits.column import Column
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.header import Header

from gaia.io import (
    DataNotFoundError,
    FileSaver,
    JsonNumpyEncoder,
    MissingColumnError,
    ParquetReader,
    ParquetTableReader,
    ReaderError,
    create_dir_if_not_exist,
    read_fits,
)
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
        read_fits("test/file.fits", "HEADER", meta=[])


def test_read_fits__data_header_not_found(mocker):
    """Test that `KeyError` is raised when no data HDU header was found."""
    mocker.patch("gaia.io.fits.getdata", side_effect=KeyError())
    with pytest.raises(KeyError):
        read_fits("test/file.fits", "HEADER", meta=[])


def test_read_fits__data_column_not_found(mocker):
    """Test that `KeyError` is raised when any of data column were not found."""
    mocker.patch("gaia.io.fits.getdata", side_effect=KeyError())
    with pytest.raises(KeyError):
        read_fits("test/file.fits", "HEADER", columns=["x"], meta=[])


def test_read_fits__metadata_column_not_found(mocker):
    """Test that `KeyError` is raised when any of metadata column were not found."""
    mocker.patch("gaia.io.fits.getheader", side_effect=KeyError())
    with pytest.raises(KeyError):
        read_fits("test/file.fits", "HEADER", meta=["not_existent_column"])


@pytest.mark.parametrize("meta", [None, []], ids=["none", "list"])
def test_read_fits__meta_empty_or_none(meta, mocker, fits_data):
    """Test that metadata is not read from the FITS file header."""
    mocker.patch("gaia.io.fits.getdata", return_value=fits_data)
    getheader_mock = mocker.patch("gaia.io.fits.getheader")
    read_fits("test/file.fits", data_header="HEADER", columns=["x"], meta=meta)
    assert getheader_mock.call_count == 0


@pytest.mark.parametrize(
    "meta_columns,expected",
    [
        (["A"], {"A": 1}),
        (["A", "B"], {"A": 1, "B": "xyz"}),
    ],
    ids=["single_column", "all_columns"],
)
def test_read_fits__return_correct_meta(meta_columns, expected, mocker, fits_header):
    """Test that metadata is read correctly."""
    mocker.patch("gaia.io.fits.getheader", return_value=fits_header)
    mocker.patch("gaia.io.fits.getdata")
    result = read_fits("test/file.fits", data_header="HEADER", meta=meta_columns)
    assert result == expected


def test_read_fits__empty_columns(mocker):
    """Test that no data is read when parameter `columns` is an empty sequence."""
    getdata_mock = mocker.patch("gaia.io.fits.getdata")
    read_fits("test/file.fits", data_header="HEADER", columns=[], meta=[])
    assert getdata_mock.call_count == 0


@pytest.mark.parametrize(
    "columns,expected",
    [
        (None, {"x": np.array([1, 2]), "y": np.array([3.0, 4.0])}),
        (["x"], {"x": np.array([1, 2])}),
        (["x", "y"], {"x": np.array([1, 2]), "y": np.array([3.0, 4.0])}),
        ([], {}),
    ],
    ids=["all_columns_by_default", "single_column", "two_columns", "no_data"],
)
def test_read_fits__return_correct_data(columns, expected, mocker, fits_data):
    """Test that correct data is returned."""
    mocker.patch("gaia.io.fits.getdata", return_value=fits_data)
    result = read_fits("test/file.fits", data_header="HEADER", columns=columns)
    assert_dict_with_numpy_equal(result, expected)


@pytest.mark.parametrize(
    "obj,expected",
    [
        ([1], "[1]"),
        (np.array([1, 2, 3]), "[1, 2, 3]"),
        ({"a": 1, "b": "string", "c": [1, 2, 3]}, '{"a": 1, "b": "string", "c": [1, 2, 3]}'),
        (
            {"a": 1, "b": "string", "c": np.array([1, 2, 3])},
            '{"a": 1, "b": "string", "c": [1, 2, 3]}',
        ),
        (
            {"a": 1, "b": "string", "c": np.array([1, 2, 3, 4]).reshape((2, 2))},
            '{"a": 1, "b": "string", "c": [[1, 2], [3, 4]]}',
        ),
    ],
    ids=[
        "one_element_list",
        "1D_numpy_array",
        "complex_object",
        "complex_object_with_1D_numpy_array",
        "complex_object_with_2D_numpy_array",
    ],
)
def test_json_numpy_decoder__decode_object(obj, expected):
    """Test that encoding object without numpy array is correct."""
    actual = json.dumps(obj, cls=JsonNumpyEncoder)
    assert actual == expected


@pytest.fixture
def parquet_file(tmp_path):
    """Prepare test parquet file and return a path to their parent directory."""
    connection = duckdb.connect(":memory:")
    connection.sql("CREATE TABLE t (A INT, B VARCHAR, C DOUBLE[]);")
    connection.sql(
        "INSERT INTO t VALUES (1, 'data1', [1.0, 2.0, 3.0]), (2, 'data2', [4.0, 5.0, 6.0]);",
    )
    connection.sql(f"COPY t TO '{tmp_path / '1.parquet'}';")
    return tmp_path


def test_parquet_reader_read_by_id__file_not_found(tmp_path):
    """Test that `FileNotFoundError` is raised when file for specific ID was not found."""
    reader = ParquetReader(tmp_path)
    with pytest.raises(DataNotFoundError):
        reader.read_by_id(1)


@pytest.mark.parametrize("read_error", [duckdb.IOException, duckdb.InvalidInputException])
def test_parquet_reader_read_by_id__cannot_read_file(read_error, parquet_file, mocker):
    """Test that `ReaderError` is raised when a file cannot be read."""
    mocker.patch("gaia.io.duckdb.sql", side_effect=read_error)
    reader = ParquetReader(parquet_file)
    with pytest.raises(ReaderError):
        reader.read_by_id(1)


@pytest.mark.parametrize(
    "id_pattern",
    [None, re.compile(r"\d*(?=\.parquet)")],
    ids=["id_as_filename", "id_as_regex"],
)
@pytest.mark.parametrize(
    "columns,expected",
    [
        (None, [dict(A=1, B="data1", C=[1.0, 2.0, 3.0]), dict(A=2, B="data2", C=[4.0, 5.0, 6.0])]),
        (("A",), [dict(A=1), dict(A=2)]),
        (
            ("A", "B", "C"),
            [dict(A=1, B="data1", C=[1.0, 2.0, 3.0]), dict(A=2, B="data2", C=[4.0, 5.0, 6.0])],
        ),
    ],
    ids=["all_columns_by_default", "specific_column", "specific_columns"],
)
def test_parquet_reader_read_by_id__read(columns, expected, id_pattern, parquet_file):
    """Test that file is read correctly."""
    reader = ParquetReader(parquet_file, id_pattern, columns)
    actual = reader.read_by_id(1)
    assert actual == expected


def test_parquet_reader_read_by_id__cannot_retrieve_id_from_path(tmp_path):
    """Test that `DataNotFoundError` is raised when ID could not be retrieved from the filepath."""
    (tmp_path / "abc.parquet").touch()  # A valid file for id=1 should be named '1.parquet'
    reader = ParquetReader(tmp_path, re.compile(r".*(?<=file)"))  # ID=everything before 'file' text
    with pytest.raises(DataNotFoundError):
        reader.read_by_id(1)


def test_parquet_table_reader_read__file_not_found(tmp_path):
    """Test that `ReaderError` is raised when no file was found."""
    filepath = tmp_path / "file.parquet"
    reader = ParquetTableReader(filepath)
    with pytest.raises(ReaderError, match=filepath.as_posix()):
        reader.read()


def test_parquet_table_reader_read__columns_not_found(parquet_file):
    """Test that `MissingColumnError` is raised when the requested column was not found."""
    filepath = next(parquet_file.glob("*.parquet"))
    reader = ParquetTableReader(filepath)
    with pytest.raises(MissingColumnError):
        reader.read(["A", "X"])


@pytest.fixture(params=["corrupted_file", "permission_denied", "invalid_format"])
def invalid_parquet(request, tmp_path):
    """Return a path to test parquet file.

    This return one of the following paths:
      - to corrupted (empty) file,
      - to write-only file,
      - to file in unsupported (.txt) format.
    """
    path = tmp_path / "file.parquet"
    match request.param:
        case "corrupted_file":
            path.touch()  # Empty file, no parquet meta
        case "permission_denied":
            path.touch(277)  # Write-only file, can't be read
        case "invalid_format":  # pragma: no cover
            path = tmp_path / "file.txt"
            path.touch()

    return path


def test_parquet_table_reader_read__read_error(invalid_parquet):
    """Test that `ReaderError` is raised when cannot read a file."""
    reader = ParquetTableReader(invalid_parquet)
    with pytest.raises(ReaderError):
        reader.read()


@pytest.mark.parametrize(
    "columns,where,expected",
    [
        (None, "WHERE A=1", [{"A": 1, "B": "data1", "C": [1.0, 2.0, 3.0]}]),
        (["A", "B"], "WHERE B='data2'", [{"A": 2, "B": "data2"}]),
        (["A", "B"], None, [{"A": 1, "B": "data1"}, {"A": 2, "B": "data2"}]),
        (
            None,
            None,
            [
                {"A": 1, "B": "data1", "C": [1.0, 2.0, 3.0]},
                {"A": 2, "B": "data2", "C": [4.0, 5.0, 6.0]},
            ],
        ),
    ],
    ids=["where", "columns_and_where", "columns", "default"],
)
def test_parquet_table_reader_read__return_correct_data(columns, where, expected, parquet_file):
    """Test that file is read correctly."""
    filepath = next(parquet_file.glob("*.parquet"))
    reader = ParquetTableReader(filepath)
    actual = reader.read(columns, where)
    assert all(a == b for a, b in zip(actual, expected))


def test_create_dir_if_not_exist__path_to_file(tmp_path):
    """Test that `ValueError` is raised when a path points to a file not directory."""
    filepath = tmp_path / "test.txt"
    filepath.touch()
    with pytest.raises(ValueError):
        create_dir_if_not_exist(filepath)


@pytest.mark.parametrize("invalid_path", [132, 1.23, ["file", "path", "/"]])
def test_create_dir_if_not_exist__invalid_path(invalid_path):
    """Test that `TypeError` is raised when path is not `str`, or `os.PathLike` object."""
    with pytest.raises(TypeError):
        create_dir_if_not_exist(invalid_path)


@pytest.fixture
def create_path(tmp_path):
    """Return a temporary path to a file or directory."""

    def create(path_final_part):
        if not path_final_part:
            return tmp_path

        return tmp_path / path_final_part

    return create


@pytest.mark.parametrize("path_final_part", [None, Path("a"), "a/b"])
def test_create_dir_if_not_exist__create_path(path_final_part, create_path):
    """Test that directory is correctly created."""
    dir_path = create_path(path_final_part)
    create_dir_if_not_exist(dir_path)
    assert dir_path.is_dir()
