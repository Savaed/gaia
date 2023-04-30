from pathlib import Path

import duckdb
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from gaia.data.converters import (
    CsvConverter,
    FitsConverter,
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
            include_columns={"not_existent_column"},
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
def mock_read_fits(mocker):
    """Factory function to mock `gaia.io.read_fits()` to return a FITS file (meta)data."""

    def _mock(read_return_values):
        return mocker.patch("gaia.data.converters.read_fits", side_effect=read_return_values)

    return _mock


@pytest.fixture
def create_converter(tmp_path):
    """Factory function to create an instance of `FitsConverter`.

    This assigns meta and buffer files to temporary paths so they are removed after tests.
    """

    def _create(config):
        converter = FitsConverter(config)
        base_dir = tmp_path / "FitsConverter"
        base_dir.mkdir()
        converter._checkpoint_filepath = base_dir / "test_checkpoint.txt"
        converter._buffer_filepath = base_dir / "test_buffer.json"
        return converter

    return _create


@pytest.fixture
def converter(create_converter):
    """Return basic `FitsConverter` instance."""
    config = FitsConvertingSettings(
        data_header="TEST_DATA_HDU",
        data_columns=None,
        meta_columns=None,
        names_map=None,
        rotation_bytes=2 * 1024 * 1024 * 1024,  # 2BG
        num_async_readers=1,  # Read sequentially
    )
    return create_converter(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("inputs", ["file.fits", Path("file.fits"), "dir/file*.fits"])
async def test_fits_converter_convert__files_not_found(inputs, mock_glob, converter):
    """Test that `FileNotFoundError` is raised when no FITS file(s) was found."""
    mock_glob([[]])
    with pytest.raises(FileNotFoundError):
        await converter.convert(inputs, Path("file.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("inputs", [Path("file.txt"), "file.txt", "dir/*.txt"])
async def test_fits_converter_convert__unsupported_input_files(inputs, mock_glob, converter):
    """Test that `ValueError` is raised when any of input files has unsupported file format."""
    mock_glob([[str(inputs)]])
    with pytest.raises(UnsupportedFileFormatError):
        await converter.convert(inputs, Path("file.json"))


@pytest.mark.asyncio
async def test_fits_converter_convert__unsupported_output_file(converter):
    """Test that `ValueError` is raised when the output file is not supported."""
    with pytest.raises(UnsupportedFileFormatError):
        await converter.convert("file.fits", Path("file.txt"))


TEST_FITS_DATA = {
    "data_column1": np.array([1, 2, 3]),
    "data_column2": np.array([4.0, 5.0, 6.0]),
    "META_COLUMN1": "metadata1",
    "META_COLUMN2": 1.2,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data_columns,meta_columns",
    [
        (["data_column1", "missing_data_column"], ["META_COLUMN1", "META_COLUMN2"]),
        (["data_column1", "data_column2"], ["META_COLUMN1", "MISSING_META_COLUMN"]),
        (["data_column1", "missing_data_column"], ["META_COLUMN1", "MISSING_META_COLUMN"]),
    ],
    ids=["in_data", "in_metadata", "both"],
)
async def test_fits_converter_convert__missing_column(
    data_columns,
    meta_columns,
    mock_glob,
    mock_read_fits,
    converter,
):
    """Test that `KeyError` is raised when no data/metadata column found in FITS read file."""
    filepath = "file.fits"
    mock_glob([[filepath]])
    mock_read_fits(KeyError)
    converter._config.data_columns = data_columns
    converter._config.meta_columns = meta_columns
    with pytest.raises(KeyError):
        await converter.convert(filepath, Path("file.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("meta_columns", [None, {"META_COLUMN1"}, {"META_COLUMN1", "META_COLUMN2"}])
@pytest.mark.parametrize("data_columns", [None, {"data_column1"}, {"data_column1", "data_column2"}])
async def test_fits_converter_convert__read_only_specified_columns(
    data_columns,
    meta_columns,
    mock_glob,
    mock_read_fits,
    tmp_path,
    converter,
):
    """Test that only specified columns are read from a FITS file."""
    filepath = "file.fits"
    mock_glob([[filepath]])
    read_mock = mock_read_fits([TEST_FITS_DATA])
    converter._config.data_columns = data_columns
    converter._config.meta_columns = meta_columns
    await converter.convert(filepath, tmp_path / "file.json")
    read_mock.assert_called_with(
        Path(filepath),
        converter._config.data_header,
        data_columns,
        meta_columns,
    )


@pytest.fixture
def create_output_files(tmp_path):
    """Factory function to cerate test output files for `FitsConverter.convert()`."""

    def _mock(outputs):
        for existent_output in outputs:
            (tmp_path / existent_output).touch()

        return tmp_path / "file.json"

    return _mock


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "existent_outputs,expected",
    [
        ({}, {"file.json", "file-1.json"}),
        ({"file.json"}, {"file.json", "file-1.json", "file-2.json"}),
        ({"file.json", "file-1.json"}, {"file.json", "file-1.json", "file-2.json", "file-3.json"}),
    ],
    ids=["no_file", "one_file_without_prefix", "several_files_one_with_prefix"],
)
async def test_fits_converter_convert__output_files_rotation(
    existent_outputs,
    expected,
    mock_glob,
    mock_read_fits,
    create_output_files,
    converter,
):
    """Test that output files are wrote with correct suffix if any suffix needed."""
    mock_glob([["file1.fits", "file2.fits"]])  # Read 2 files
    mock_read_fits([TEST_FITS_DATA, TEST_FITS_DATA])
    output_path = create_output_files(existent_outputs)
    converter._config.rotation_bytes = 0  # Save a single output for each input files batch
    await converter.convert("file*.fits", output_path)

    # Skip directory which includes test meta and buffer files
    actual = {path for path in output_path.parent.iterdir() if path.is_file()}
    expected_absolute_paths = {output_path.parent / filename for filename in expected}
    assert actual == expected_absolute_paths


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "columns_map,expected",
    [
        (
            {"data_column1": "col1", "META_COLUMN1": "meta1"},
            {"col1", "data_column2", "meta1", "META_COLUMN2"},
        ),
        (
            None,
            {"data_column1", "data_column2", "META_COLUMN1", "META_COLUMN2"},
        ),
        (
            {},
            {"data_column1", "data_column2", "META_COLUMN1", "META_COLUMN2"},
        ),
        (
            {"data_column1": "col1", "abc": "ABC"},
            {"col1", "data_column2", "META_COLUMN1", "META_COLUMN2"},
        ),
    ],
    ids=["rename_several_columns", "no_renaming", "no_renaming", "skip_not_existent_column"],
)
async def test_fits_converter_convert__rename_columns(
    columns_map,
    expected,
    mock_glob,
    mock_read_fits,
    tmp_path,
    converter,
):
    """Test that columns are renamed correctly."""
    filepath = "file.fits"
    output = tmp_path / "file.json"
    mock_glob([[filepath]])
    mock_read_fits([TEST_FITS_DATA])
    converter._config.names_map = columns_map
    await converter.convert(filepath, output)
    actual = duckdb.sql(f"SELECT * FROM '{output}' LIMIT 1;").columns
    assert set(actual) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["file.json", "file.parquet"])
async def test_fits_converter_convert__skip_already_processed_files_after_error(
    output,
    mock_glob,
    mock_read_fits,
    tmp_path,
    converter,
):
    """Test that the already processed files are not reprocessed after resuming the work after
    interrupting the conversion.
    """
    # We want convert file 'file1.fits' and 'file2.fits', but when first read 'file2.fits' an error
    # occured. So we start converting again, but the first attempt to read raise an error.
    # In this scenario at the third converting process only 'file2.fits' should be read as '
    # file1.fits' was succeffuly processed on our first converting process.
    # 2 errors is to ensure that
    mock_glob(
        [
            ["file1.fits", "file2.fits", "file3.fits"],
            ["file1.fits", "file2.fits", "file3.fits"],
            ["file1.fits", "file2.fits", "file3.fits"],
        ],
    )
    mock_read_fits([TEST_FITS_DATA, KeyError, TEST_FITS_DATA, KeyError, TEST_FITS_DATA])
    output_path = tmp_path / output

    try:
        await converter.convert("file.fits", output_path)
    except KeyError:
        # Skip to next converting process
        pass

    try:
        await converter.convert("file.fits", output_path)
    except KeyError:
        # Skip to next converting process
        pass

    await converter.convert("file.fits", output_path)
    actual = duckdb.sql(f"SELECT * FROM '{output_path}';").df().sort_index(axis=1)  # Sort columns
    expected = create_df(
        (
            TEST_FITS_DATA.keys(),
            TEST_FITS_DATA.values(),
            TEST_FITS_DATA.values(),
            TEST_FITS_DATA.values(),
        ),
    ).sort_index(
        axis=1,
    )  # Sort columns
    assert_frame_equal(actual, expected)


@pytest.mark.asyncio
async def test_fits_converter_convert__save_ramaining_buffer_on_exit(
    mock_glob,
    mock_read_fits,
    converter,
    tmp_path,
):
    """Test that ."""
    filepath = "file.fits"
    mock_glob([[filepath]])
    mock_read_fits([TEST_FITS_DATA])
    output = tmp_path / "file.json"
    expected = create_df(
        (
            TEST_FITS_DATA.keys(),
            TEST_FITS_DATA.values(),
        ),
    ).sort_index(
        axis=1,
    )  # Sort columns
    await converter.convert(filepath, output)
    actual = duckdb.sql(f"FROM '{output}';").df().sort_index(axis=1)  # Sort columns
    assert_frame_equal(actual, expected)
