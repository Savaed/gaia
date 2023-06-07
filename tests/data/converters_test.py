import asyncio
import copy
import re
from contextlib import suppress
from pathlib import Path
from unittest.mock import Mock

import duckdb
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from gaia.data.converters import (
    ConverterError,
    CsvConverter,
    FitsConverter,
    FitsConvertingOutputFormat,
    FitsConvertingSettings,
    UnsupportedFileFormatError,
)
from tests.conftest import create_df


@pytest.fixture
def mock_glob(mocker):
    """Factory function to mock `glob.glob()` to return a specific paths."""

    def _mock(glob_return_values):
        mocker.patch("gaia.data.converters.glob.glob", side_effect=glob_return_values)

    return _mock


@pytest.fixture
def create_csv_file():
    """Factory function to save a csv file and return its path."""

    def _create(header, records, filename):
        header_text = ",".join(header) + "\n"
        records_text = []
        for record in records:
            records_text.append(",".join(map(str, record)))

        file_content = header_text + "\n".join(records_text)
        filename.write_text(file_content)

    return _create


@pytest.fixture
def create_single_csv_file(create_csv_file, tmp_path):
    """Create a single csv file and return its path."""
    header = ["a", "b"]
    records = [[1, 2.0], [3, 4.0]]
    filepath = tmp_path / "file.csv"
    create_csv_file(header, records, filepath)
    return filepath


@pytest.mark.parametrize(
    "filepath",
    [Path("file.csv"), "file.csv", "**/*.csv"],
    ids=["single_file", "single_file_as_path", "regex_patern"],
)
def test_csv_converter_convert__input_files_not_found(filepath, mock_glob):
    """Test that `FileNotFoundError` is raised when no input file(s) was found."""
    mock_glob([[]])
    with pytest.raises(FileNotFoundError):
        CsvConverter().convert(filepath, Path("file.json"))


@pytest.mark.parametrize(
    "inputs,glob_return",
    [
        ("file.json", ["file.json"]),
        (Path("file.py"), ["file.py"]),
        ("./**/*.(xml|py)", ["./file.py", "./folder1/file.py", "./folder1/folder2/file.xml"]),
    ],
    ids=[
        "single_file_as_string",
        "single_file_as_path",
        "regex_pattern",
    ],
)
def test_csv_converter_convert__unsupported_input_files(inputs, glob_return, mock_glob):
    """Test that `ValueError` is raised when input file(s) are unsupported."""
    mock_glob([glob_return])
    with pytest.raises(UnsupportedFileFormatError):
        CsvConverter().convert(inputs, Path("file.json"))


def test_csv_converter_convert__unsupported_output_file():
    """Test that `ValueError` is raised when output file are unsupported e.g. xml."""
    with pytest.raises(UnsupportedFileFormatError):
        CsvConverter().convert("file.csv", Path("file.xml"))


def test_csv_converter_convert__invalid_include_column(mock_glob, create_single_csv_file):
    """Test that `ValueError` is raised when no column to select from input the file was found."""
    mock_glob([[create_single_csv_file.as_posix()]])
    with pytest.raises(ValueError):
        CsvConverter().convert(
            create_single_csv_file,
            Path("file.json"),
            include_columns=["not_existent_column"],
        )


def test_csv_converter_convert__invalid_mapping_column(mock_glob, create_single_csv_file):
    """Test that `ValueError` is raised when no column to rename was found."""
    mock_glob([[create_single_csv_file.as_posix()]])
    with pytest.raises(ValueError):
        CsvConverter().convert(
            create_single_csv_file,
            Path("file.json"),
            columns_mapping={"not_existent_column": "x"},
        )


@pytest.mark.parametrize("output", ["output.json", "output.parquet"])
@pytest.mark.parametrize(
    "columns,mapping,expected",
    [
        (None, None, create_df((["a", "b"], [1, 2.0], [3, 4.0]))),
        (["a"], None, create_df((["a"], [1], [3]))),
        (["a"], {"a": "A"}, create_df((["A"], [1], [3]))),
        (["a", "b"], None, create_df((["a", "b"], [1, 2.0], [3, 4.0]))),
    ],
    ids=[
        "all_columns_by_default",
        "specific_column",
        "map_columns_names",
        "specific_columns",
    ],
)
def test_csv_converter_convert__convert_single_file_correctly(
    columns,
    mapping,
    expected,
    output,
    create_single_csv_file,
    tmp_path,
):
    """Test that data from one input file is converted correctly to supported format."""
    output_path = tmp_path / output
    CsvConverter().convert(
        create_single_csv_file,
        output=output_path,
        include_columns=columns,
        columns_mapping=mapping,
    )
    actual = duckdb.sql(f"FROM '{output_path}';").df()

    # Python's `int` data type when only positive numbers are used is internally cast to `uint`, so
    # the dtype is different but the values ​​are the same, therefore `check_dtype=False` is ok.
    assert_frame_equal(actual, expected, check_dtype=False)


@pytest.fixture
def create_two_csv_files(create_csv_file, tmp_path):
    """Create two csv files with the same structure and return their parent directory path."""
    header = ["a", "b"]
    records1 = [[1, 2.0], [3, 4.0]]
    records2 = [[5, 6.0], [7, 8.0]]
    create_csv_file(header, records1, tmp_path / "file1.csv")
    create_csv_file(header, records2, tmp_path / "file2.csv")
    return tmp_path


@pytest.mark.parametrize("output", ["output.json", "output.parquet"])
@pytest.mark.parametrize(
    "columns,mapping,expected",
    [
        (None, None, create_df((["a", "b"], [1, 2.0], [3, 4.0], [5, 6.0], [7, 8.0]))),
        (["a"], None, create_df((["a"], [1], [3], [5], [7]))),
        (["a"], {"a": "A"}, create_df((["A"], [1], [3], [5], [7]))),
        (["a", "b"], None, create_df((["a", "b"], [1, 2.0], [3, 4.0], [5, 6.0], [7, 8.0]))),
    ],
    ids=[
        "all_columns_by_default",
        "specified_column",
        "map_columns_names",
        "specified_columns",
    ],
)
def test_csv_converter_convert__convert_multiple_files_correctly(
    columns,
    mapping,
    expected,
    output,
    create_two_csv_files,
    tmp_path,
):
    """Test that data from one input file is converted correctly to supported format."""
    output_path = tmp_path / output
    files_pattern = f"{create_two_csv_files}/*.csv"
    CsvConverter().convert(
        files_pattern,
        output=output_path,
        include_columns=columns,
        columns_mapping=mapping,
    )
    # The order of the data from multiple files is not deterministic, so sort it.
    actual = duckdb.sql(f"FROM '{output_path}' ORDER BY a;").df()

    # Python's `int` data type when only positive numbers are used is internally cast to `uint`, so
    # the dtype is different but the values ​​are the same, therefore `check_dtype=False` ok.
    assert_frame_equal(actual, expected, check_dtype=False)


@pytest.fixture
def fits_convert_settings():
    """Return FITS converting settings."""
    return FitsConvertingSettings(
        data_header="TEST_HDU",
        data_columns=["col1", "col2"],
        meta_columns=["meta1", "meta2"],
        names_map=dict(col1="column1"),
        output_format=FitsConvertingOutputFormat.PARQUET,
    )


@pytest.fixture
def create_fits_converter(tmp_path):
    """Factory function to create an instance of `FitsConverter`."""

    def create(settings):
        converter = FitsConverter(settings)
        converter._checkpoint_filepath = tmp_path / "test_FitsConverter_checkpoint.txt"
        converter._tmp_time_series_path = tmp_path / "test_FitsConverter_buffer.json"
        return converter

    return create


@pytest.fixture
def fits_converter(fits_convert_settings, create_fits_converter):
    """Return an instance of `FitsConverter`."""
    return create_fits_converter(fits_convert_settings)


@pytest.mark.asyncio
async def test_fits_converter_convert__input_file_not_found(fits_converter, tmp_path):
    """Test that `FileNotFoundError` is raised when input is a single, not existing file."""
    input_filepath = Mock(spec=Path, **{"exists.return_value": False})
    with pytest.raises(FileNotFoundError):
        await fits_converter.convert(input_filepath, tmp_path, re.compile(".*"))


@pytest.fixture
def fits_files(tmp_path):
    """Prepare FITS time series empty files and return a path to their parent directory."""
    for file in ("target_id_1.fits", "target_id_2.fits"):
        (tmp_path / file).touch()

    return tmp_path


@pytest.mark.asyncio
async def test_fits_converter_convert__cannot_retrieve_target_id_from_file_paths(
    fits_files,
    mocker,
    fits_converter,
    tmp_path,
):
    """Test that input files are skiped when cannot retrieve target id from theirs paths."""
    read_fits_mock = mocker.patch("gaia.data.converters.read_fits")
    inputs = f"{fits_files}/*.fits"

    # The correct regex is '(?<=id_)\d*'
    await fits_converter.convert(inputs, tmp_path, re.compile("(?<=kplr).*"))
    read_fits_mock.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_error",
    [OSError, KeyError],
    ids=["invalid_file", "invalid_header/columns"],
)
async def test_fits_converter_convert__cannot_read_fits_file(
    read_error,
    fits_files,
    mocker,
    fits_converter,
    tmp_path,
):
    """Test that `ConverterError` is raised when cannot read a FITS input file."""
    mocker.patch("gaia.data.converters.read_fits", side_effect=read_error)
    inputs = f"{fits_files}/*.fits"
    with pytest.raises(ConverterError):
        await fits_converter.convert(inputs, tmp_path, re.compile(r"(?<=id_)\d*"))


TEST_FITS_TIME_SERIES = {
    "col1": np.array([1.0, 2.0, 3.0]),
    "col2": np.array([4.0, np.nan, 6.0]),
    "meta1": 1,
    "meta2": "test",
}


@pytest.mark.asyncio
async def test_fits_converter_convert__cannot_save_output_file(
    fits_files,
    mocker,
    fits_converter,
    tmp_path,
):
    """Test that `ConverterError` is raised when cannot save converted output file."""
    mocker.patch("gaia.data.converters.read_fits", return_value=TEST_FITS_TIME_SERIES)
    mocker.patch("gaia.data.converters.duckdb.execute", side_effect=duckdb.IOException)
    inputs = f"{fits_files}/*1.fits"
    with pytest.raises(ConverterError):
        await fits_converter.convert(inputs, tmp_path, re.compile(r"(?<=id_)\d*"))


@pytest.mark.asyncio
async def test_fits_converter_convert__resume_converting(
    fits_files,
    mocker,
    fits_converter,
    tmp_path,
):
    """Test that the next conversion after a failure will not convert already converted files."""
    mocker.patch(
        "gaia.data.converters.read_fits",
        side_effect=[TEST_FITS_TIME_SERIES, KeyError, TEST_FITS_TIME_SERIES],
    )
    duckdb_mock = mocker.patch("gaia.data.converters.duckdb.execute")
    inputs = f"{fits_files}/*.fits"

    with suppress(Exception):  # Don't care about this error
        await fits_converter.convert(inputs, tmp_path, re.compile(r"(?<=id_)\d*"))

    await fits_converter.convert(inputs, tmp_path, re.compile(r"(?<=id_)\d*"))

    # Expected call count: 2 for the first file and 2 for the second. If the checkpoint mechanism
    # didn't work, it will be 6 calls (2 for the first file and after the failure 2 for the first
    # file [again] and 2 for the second).
    assert duckdb_mock.call_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("data_columns", [None, ["col1", "col2"]])
@pytest.mark.parametrize("meta_columns", [None, ["meta1", "meta2"]])
async def test_fits_converter_convert__read_only_specific_columns(
    meta_columns,
    data_columns,
    mocker,
    fits_files,
    tmp_path,
    create_fits_converter,
):
    """Test that only specified data/meta columns are read from FITS files."""
    read_fits_mock = mocker.patch(
        "gaia.data.converters.read_fits",
        return_value=TEST_FITS_TIME_SERIES,
    )

    settings = FitsConvertingSettings(
        data_columns=data_columns,
        data_header="TEST_HDU",
        meta_columns=meta_columns,
        names_map=None,
        output_format=FitsConvertingOutputFormat.PARQUET,
    )
    converter = create_fits_converter(settings)
    inputs = f"{fits_files}/*1.fits"
    await converter.convert(inputs, tmp_path, re.compile(r"(?<=id_)\d*"))
    read_fits_mock.assert_called_once_with(
        fits_files / "target_id_1.fits",
        "TEST_HDU",
        data_columns,
        meta_columns,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", iter(FitsConvertingOutputFormat))
async def test_fits_converter_convert__rename_columns(
    output_format,
    mocker,
    fits_files,
    tmp_path,
    create_fits_converter,
    fits_convert_settings,
):
    """Test that data/meta columns are renaming if requried."""
    mocker.patch("gaia.data.converters.read_fits", return_value=TEST_FITS_TIME_SERIES)
    settings = copy.copy(fits_convert_settings)
    settings.output_format = output_format
    converter = create_fits_converter(settings)

    # Data from `read_fits()` has columns: ['column1', 'col2', 'meta1', 'meta2']
    # and rename_map is 'col1' -> 'column1'.
    expected = create_df(
        (
            ["column1", "col2", "meta1", "meta2"],
            [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], 1, "test"],
        ),
    )
    inputs = f"{fits_files}/*1.fits"
    await converter.convert(inputs, tmp_path, re.compile(r"(?<=id_)\d*"))
    actual = duckdb.sql(f"FROM '{tmp_path / f'1.{output_format.value}'}';").df()
    assert_frame_equal(actual, expected, check_dtype=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", iter(FitsConvertingOutputFormat))
async def test_fits_converter_convert__group_inputs_for_single_target_id(
    output_format,
    mocker,
    tmp_path,
    create_fits_converter,
    fits_convert_settings,
):
    """Test that input files related to the same `target_id` are saved as a single output file."""
    mocker.patch("gaia.data.converters.read_fits", side_effect=[TEST_FITS_TIME_SERIES] * 2)

    for filepath in ("first_target_id_1.fits", "second_target_id_1.fits"):
        (tmp_path / filepath).touch()

    settings = copy.copy(fits_convert_settings)
    settings.output_format = output_format
    converter = create_fits_converter(settings)

    expected = create_df(
        (
            ["column1", "col2", "meta1", "meta2"],
            [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], 1, "test"],  # Data from 'first_target_id_1.fits'
            [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], 1, "test"],  # Data from 'second_target_id_1.fits'
        ),
    )
    inputs = f"{tmp_path}/*.fits"
    await converter.convert(inputs, tmp_path, re.compile(r"(?<=id_)\d*"))
    actual = duckdb.sql(f"FROM '{tmp_path / f'1.{output_format.value}'}';").df()
    assert_frame_equal(actual, expected, check_dtype=False)


@pytest.mark.asyncio
async def test_fits_converter_convert__cancel_reading(mocker, fits_files, fits_converter, tmp_path):
    """Test that all reading tasks are gracefully cancelled and `CancelledError` is raised."""
    mocker.patch("gaia.data.converters.read_fits", side_effect=asyncio.CancelledError)
    inputs = f"{fits_files}/*.fits"
    with pytest.raises(asyncio.CancelledError):
        await fits_converter.convert(inputs, tmp_path, re.compile(r"(?<=id_)\d*"))


@pytest.fixture(params=[True, False], ids=["dir_exists", "dir_not_exists"])
def output_dir(request, tmp_path):
    """Return existent and not existent directory ."""
    if request.param:
        return tmp_path
    return tmp_path / "sdfsdfsd/"


@pytest.mark.asyncio
async def test_fits_converter_convert__create_output_dir_if_not_exist(
    output_dir,
    fits_converter,
    fits_files,
    mocker,
):
    """Test that the output directory is created if not exist."""
    mocker.patch("gaia.data.converters.read_fits")
    mocker.patch("gaia.data.converters.duckdb.execute")

    inputs = f"{fits_files}/*.fits"
    await fits_converter.convert(inputs, output_dir, re.compile(r"(?<=id_)\d*"))
    assert output_dir.is_dir()
