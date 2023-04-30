import glob
import re
from pathlib import Path
from typing import TypeAlias

import duckdb

from gaia.io import Columns


PathOrPattern: TypeAlias = Path | str


class UnsupportedFileFormatError(Exception):
    """Raised when the file is in an unsupported format."""


class CsvConverter:
    _SUPPORTED_OUTPUT_FILES = (".json", ".parquet")
    _TMP_TABLE = "tmp"

    def convert(
        self,
        inputs: PathOrPattern,
        output: Path,
        include_columns: Columns | None = None,
        columns_mapping: dict[str, str] | None = None,
    ) -> None:
        """Convert a csv file to json or parquet format with optional column renaming.

        Args:
            filepath (PathOrPattern): Input csv file path or glob pattern to many csv files
            output (PathOrPattern): Path to the output file
            include_columns (Columns | None, optional): What columns to include in the output file.
                If None then all columns will be included. Defaults to None.
            columns_mapping (dict[str, str] | None, optional): Old to new column names mapping.
                If None then no renaming is performed. Defaults to None.

        Raises:
            FileNotFoundError: Input file(s) not found
            ValueError: Column to select or rename not found in the input file(s) OR unsupported
                input/output file(s) formats
        """
        self._validate_output_file(output)
        input_filepaths = [Path(path) for path in glob.glob(str(inputs), recursive=True)]

        if not input_filepaths:
            raise FileNotFoundError(f"No files found matching the pattern '{inputs}'")

        self._validate_input_files(input_filepaths)
        connection = duckdb.connect(":memory:")
        self._create_tmp_table(inputs, include_columns, connection)

        if columns_mapping:
            self._rename_columns(inputs, columns_mapping, connection)

        output_file_extension = str(output).rpartition(".")[-1]
        compression = " (COMPRESSION ZSTD)" if output_file_extension == "parquet" else ""
        connection.execute(f"COPY {self._TMP_TABLE} TO '{output}'{compression};")
        connection.execute(f"FROM '{output}' LIMIT 1;")  # Validate the converted file

    def _create_tmp_table(
        self,
        inputs: PathOrPattern,
        include_columns: Columns | None,
        connection: duckdb.DuckDBPyConnection,
    ) -> None:
        columns = ",".join(include_columns) if include_columns else "*"
        try:
            connection.execute(
                f"CREATE TABLE {self._TMP_TABLE} AS SELECT {columns} FROM '{inputs}';",
            )
        except duckdb.BinderException as ex:
            column = re.search(r'(?<=column )"\w*', str(ex)).group()  # type: ignore
            raise ValueError(
                f"{column} specified in 'include_columns' parameter not found in the source CSV {inputs}",  # noqa
            )

    def _rename_columns(
        self,
        inputs: PathOrPattern,
        columns_mapping: dict[str, str],
        connection: duckdb.DuckDBPyConnection,
    ) -> None:
        for old_column, new_column in columns_mapping.items():
            try:
                connection.execute(
                    f"ALTER TABLE {self._TMP_TABLE} RENAME {old_column} TO {new_column};",
                )
            except duckdb.BinderException:
                raise ValueError(
                    f"{old_column} specified in 'columns_mapping' parameter not found in the source CSV {inputs}",  # noqa
                )

    def _validate_output_file(self, output: Path) -> None:
        if output.suffix not in self._SUPPORTED_OUTPUT_FILES:
            raise UnsupportedFileFormatError(
                f"Unsupported output file format. Only '{', '.join(self._SUPPORTED_OUTPUT_FILES)}' files are supported",  # noqa
            )

    def _validate_input_files(self, inputs: list[Path]) -> None:
        if any(path.suffix not in {".csv"} for path in inputs):
            raise UnsupportedFileFormatError("Only 'csv' files are supported")
