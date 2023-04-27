from pathlib import Path
from unittest.mock import Mock

import duckdb
import pytest
from pandas.testing import assert_frame_equal

from gaia.data.converters import CsvConverter
from tests.conftest import create_df


@pytest.fixture
def create_csv_file(tmp_path):
    """Factory function to save a csv file. Return the path to it."""

    def _create(header, records, filename):
        filepath = tmp_path / filename
        header_text = ",".join(header) + "\n"
        records_text = []
        for record in records:
            records_text.append(",".join(map(str, record)))

        file_content = header_text + "\n".join(records_text)
        filepath.write_text(file_content)
        return filepath

    return _create


@pytest.fixture
def single_csv_file(create_csv_file):
    """Create a single csv file. Return a file path to it."""
    header = ["a", "b"]
    records = [[1, 2.0], [3, 4.0]]
    return create_csv_file(header, records, "file.csv")


@pytest.mark.parametrize(
    "filepath",
    [Path("file.csv"), "file.csv", "./**/*.csv"],
    ids=["single_file", "single_file_as_path", "regex_patern"],
)
def test_csv_converter_convert__files_not_found(filepath, tmp_path):
    """Test that `FileNotFoundError` is raised when no input file was found."""
    with pytest.raises(FileNotFoundError):
        CsvConverter().convert(tmp_path / filepath, "test_file.json")


@pytest.mark.parametrize(
    "input,glob_return",
    [
        ("file.json", ["file.json"]),
        (Path("file.py"), ["file.py"]),
        ("./**/*.(xml|py)", ["./file.py", "./folder1/file.py", "./folder1/folder2/file.xml"]),
    ],
    ids=[
        "single_file",
        "single_file_as_path",
        "regex_pattern",
    ],
)
def test_csv_converter_convert__unsupported_input_files(input, glob_return, mocker):
    """Test that `ValueError` is raised when input file(s) are unsupported."""
    mocker.patch("gaia.data.converters.glob.glob", return_value=glob_return)
    with pytest.raises(ValueError):
        CsvConverter().convert(input, "file.json")


def test_csv_converter_convert__unsupported_output_file(mocker):
    """Test that `ValueError` is raised when output file are unsupported e.g. xml."""
    mocker.patch("gaia.data.converters.glob.glob", return_value=["file.csv"])
    with pytest.raises(ValueError):
        CsvConverter().convert("file.csv", "file.xml")


def test_csv_converter_convert__invalid_include_column(mocker):
    """Test that `ValueError` is raised when no column to select from input the file was found."""
    invalid_column = "not_existent_column"
    db_connection_mock = Mock(
        **{
            "execute.side_effect": duckdb.BinderException(
                f'Binder Error: Referenced column "{invalid_column}" not found in FROM clause!',
            ),
        },
    )
    mocker.patch("gaia.data.converters.duckdb.connect", return_value=db_connection_mock)
    mocker.patch("gaia.data.converters.glob.glob", return_value=["file.csv"])
    with pytest.raises(ValueError):
        CsvConverter().convert("file.csv", "file.json")


def test_csv_converter_convert__invalid_mapping_column(mocker):
    """Test that `ValueError` is raised when no column to rename was found."""
    invalid_column = "not_existent_column"
    db_connection_mock = Mock(
        **{
            "execute.side_effect": [
                None,
                duckdb.BinderException(
                    f'Binder Error: Referenced column "{invalid_column}" not found in FROM clause!',
                ),
            ],
        },
    )
    mocker.patch("gaia.data.converters.duckdb.connect", return_value=db_connection_mock)
    mocker.patch("gaia.data.converters.glob.glob", return_value=["file.csv"])
    with pytest.raises(ValueError):
        CsvConverter().convert("file.csv", "file.json", columns_mapping={invalid_column: "x"})


@pytest.mark.parametrize("output", ["output.json", "output.parquet"])
@pytest.mark.parametrize(
    "columns, mapping, expected",
    [
        (None, None, create_df((["a", "b"], [1, 2.0], [3, 4.0]))),
        (["a"], None, create_df((["a"], [1], [3]))),
        (["a"], {"a": "A"}, create_df((["A"], [1], [3]))),
        (["a", "b"], None, create_df((["a", "b"], [1, 2.0], [3, 4.0]))),
    ],
    ids=[
        "all_columns_by_default",
        "specified_column",
        "map_columns_names",
        "specified_columns",
    ],
)
def test_csv_converter_convert__convert_single_file_correctly(
    columns,
    mapping,
    expected,
    output,
    single_csv_file,
    tmp_path,
):
    """Test that data from one input file is converted correctly to supported format."""
    output = (tmp_path / output).as_posix()
    CsvConverter().convert(
        single_csv_file,
        output=output,
        include_columns=columns,
        columns_mapping=mapping,
    )
    result = duckdb.sql(f"FROM {output!r};").df()

    # Python's `int` data type when only positive numbers are used is internally cast to `uint`, so
    # the dtype is different but the values ​​are the same, therefore `check_dtype=False` ok.
    assert_frame_equal(result, expected, check_dtype=False)


@pytest.fixture
def two_csv_files(create_csv_file):
    """Create two csv files with the same structure. Return the path to their parent directory."""
    header = ["a", "b"]
    records1 = [[1, 2.0], [3, 4.0]]
    records2 = [[5, 6.0], [7, 8.0]]
    create_csv_file(header, records1, "file1.csv")
    return create_csv_file(header, records2, "file2.csv").parent


@pytest.mark.parametrize("output", ["output.json", "output.parquet"])
@pytest.mark.parametrize(
    "columns, mapping, expected",
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
    two_csv_files,
    tmp_path,
):
    """Test that data from one input file is converted correctly to supported format."""
    output = (tmp_path / output).as_posix()
    files_pattern = f"{two_csv_files.as_posix()}/*.csv"
    CsvConverter().convert(
        files_pattern,
        output=output,
        include_columns=columns,
        columns_mapping=mapping,
    )
    # The order of the data from multiple files is not deterministic, so sort it
    result = duckdb.sql(f"FROM {output!r} ORDER BY a;").df()

    # Python's `int` data type when only positive numbers are used is internally cast to `uint`, so
    # the dtype is different but the values ​​are the same, therefore `check_dtype=False` ok.
    assert_frame_equal(result, expected, check_dtype=False)
