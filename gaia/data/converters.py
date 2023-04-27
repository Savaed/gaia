import glob
import re
from pathlib import Path

import duckdb

from gaia.io import Columns


class CsvConverter:
    _SUPPORTED_OUTPUT_FILES = ("json", "parquet")
    _TMP_TABLE = "tmp"

    def convert(
        self,
        filepath: str | Path,
        output: str | Path,
        include_columns: Columns | None = None,
        columns_mapping: dict[str, str] | None = None,
    ) -> None:
        """Convert a csv file to json or parquet format with optional column renaming.

        Args:
            filepath (str | Path): Input csv file path or glob pattern to many csv files
            output (str | Path): Path to the output file
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

        if not (input_filepaths := glob.glob(str(filepath), recursive=True)):
            raise FileNotFoundError(f"No files found matching the pattern '{filepath}'")

        self._validate_input_files(input_filepaths)
        connection = duckdb.connect(":memory:")
        self._create_tmp_table(filepath, include_columns, connection)

        if columns_mapping:
            self._rename_columns(filepath, columns_mapping, connection)

        output_file_extension = str(output).rpartition(".")[-1]
        compression = " (COMPRESSION ZSTD)" if output_file_extension == "parquet" else ""
        connection.execute(f"COPY {self._TMP_TABLE} TO '{output}'{compression};")
        connection.execute(f"FROM '{output}' LIMIT 1;")  # Validate the converted file

    def _create_tmp_table(
        self,
        filepath: Path | str,
        include_columns: Columns | None,
        connection: duckdb.DuckDBPyConnection,
    ) -> None:
        columns = ",".join(include_columns) if include_columns else "*"
        try:
            connection.execute(
                f"CREATE TABLE {self._TMP_TABLE} AS SELECT {columns} FROM '{filepath}';",
            )
        except duckdb.BinderException as ex:
            column = self._extract_column_from_error(ex)
            raise ValueError(
                f"{column} specified in 'include_columns' parameter not found in the source CSV {filepath}",  # noqa
            )

    def _rename_columns(
        self,
        filepath: str | Path,
        columns_mapping: dict[str, str],
        connection: duckdb.DuckDBPyConnection,
    ) -> None:
        for old_column, new_column in columns_mapping.items():
            try:
                connection.execute(
                    f"ALTER TABLE {self._TMP_TABLE} RENAME {old_column} TO {new_column};",
                )
            except duckdb.BinderException as ex:
                column = self._extract_column_from_error(ex)
                raise ValueError(
                    f"{column} specified in 'columns_mapping' parameter not found in the source CSV {filepath}",  # noqa
                )

    def _validate_output_file(self, path: str | Path) -> None:
        if str(path).rpartition(".")[-1] not in self._SUPPORTED_OUTPUT_FILES:
            raise ValueError(
                f"Supported output files are: {', '.join(self._SUPPORTED_OUTPUT_FILES)}",
            )

    def _validate_input_files(self, paths: list[str]) -> None:
        input_extensions = [path.rpartition(".")[-1] for path in paths]

        if any([ext != "csv" for ext in input_extensions]):
            raise ValueError("Only csv input files are supported")

    def _extract_column_from_error(self, ex: Exception) -> str:
        return re.search(r'(?<=column )"\w*', str(ex)).group()  # type: ignore
