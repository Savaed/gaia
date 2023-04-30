from pathlib import Path

import duckdb
import pytest
from pandas.testing import assert_frame_equal

from gaia.data.converters import CsvConverter, UnsupportedFileFormatError
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
