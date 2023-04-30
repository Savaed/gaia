import asyncio
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TypeAlias

import duckdb

from gaia.io import Columns, FitsData, JsonNumpyEncoder, read_fits


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


@dataclass
class FitsConvertingSettings:
    """Configration to setup FITS files converting process."""

    data_header: str
    """HDU extension in which data is stored."""
    data_columns: Columns | None
    """Data columns to read."""
    meta_columns: Columns | None
    """Metadata columns to read"""
    names_map: dict[str, str] | None
    """Mapping from old columns name to the new ones."""
    rotation_bytes: int
    """Maximum buffer file size beyond which a new output file will be written."""
    num_async_readers: int
    """Number of asynchronous FITS files reads. Should be at least 1 (sequantial reading)."""


class FitsConverter:
    """File converter for FITS time series which allows converting files to JSON or PARQUET."""

    def __init__(self, config: FitsConvertingSettings) -> None:
        self._config = config
        self._processed_filepaths: list[str] = []

        base_path = Path().cwd() / self.__class__.__name__
        self._checkpoint_filepath = Path(f"{base_path}_checkpoint.txt")
        self._buffer_filepath = Path(f"{base_path}_buffer.json")

    async def convert(self, inputs: PathOrPattern, output: Path) -> None:
        """Convert one or many input FITS files to one or many output JSON or PARQUET files.

        This will create one or many output files based on the `config.rotation_bytes` parameter in
        that way when the internal buffer file size exceeded `config.rotation_bytes` the new output
        file is created. Note that the output file may not be the same size as the buffer and it
        depends on the format of the output file (for json output the file size will be roughly the
        same). If there is a file in the location pointed by `output` then the file is saved with
        the incremental integer suffix e.g. 'file-1.json', 'file-2.json' etc. For more information
        about available settings please see the `FitsConvertingSettings` documentation.

        The method implements a checkpoint mechanism that prevents to re-processed already converted
        files in case of any error occurs in the converting process.

        Args:
            inputs (PathOrPattern): Path or string pattern to the input file(s)
            output (Path): Path to the output file
            config: FitsConvertingSettings: Converting settings to control converting process

        Raises:
            FileNotFoundError: Input file(s) not found
            KeyError: Missing data/metadata column to read from file(s)
            UnsupportedFileFormatError: Any of the input files has a format other than FITS OR the
                output file(s) has a format other than JSON or parquet
        """
        try:
            await self._convert(inputs, output)
        except:  # noqa
            # On any error, if any input file has been processed,
            # save its file path to prevent reprocessing.
            if self._processed_filepaths:
                with open(self._checkpoint_filepath, "a") as f:
                    f.writelines(self._processed_filepaths)
            raise
        else:
            # If no errors occurred and all files were processed correctly, delete the buffer file.
            os.remove(self._checkpoint_filepath)

    async def _convert(self, inputs: PathOrPattern, output: Path) -> None:
        self._checkpoint_filepath.touch()
        self._validate_output_path(output)
        paths = self._get_input_paths(inputs)
        self._validate_input_paths(paths)
        suffix = self._determine_output_file_start_suffix(output)
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor() as pool:
            for path_batch in self._iterate_path_batches(paths):
                tasks = self._start_read_tasks(loop, pool, path_batch)
                results = await asyncio.gather(*tasks)  # Should raise on any error
                batch_data = self._handle_read_results(results)
                self._write_batch_data_to_buffer(batch_data)
                self._processed_filepaths.extend([p.as_posix() + "\n" for p in path_batch])

                if self._buffer_filepath.stat().st_size > self._config.rotation_bytes:
                    self._copy_data_from_buffer_to_output(suffix, output)
                    self._buffer_filepath.write_text("")  # Flush buffer file
                    suffix = suffix + 1 if suffix else 1

        # After the entire process save remaining buffer if any
        if self._buffer_filepath.stat().st_size:
            self._copy_data_from_buffer_to_output(suffix, output)
            os.remove(self._buffer_filepath)

    def _write_batch_data_to_buffer(self, batch_data: list[str]) -> None:
        with open(self._buffer_filepath, "a") as f:
            f.writelines(batch_data)
            f.write("\n")

    def _handle_read_results(self, results: list[FitsData]) -> list[str]:
        batch_data: list[str] = []

        for result in results:
            if self._config.names_map:
                result = {self._config.names_map.get(k, k): v for k, v in result.items()}

            # Dump FITS file content to json string. Use custom decoder for Numpy arrays.
            string_data = json.dumps(result, sort_keys=True, cls=JsonNumpyEncoder)
            batch_data.append(string_data)

        return batch_data

    def _copy_data_from_buffer_to_output(self, current_suffix: int | None, output: Path) -> None:
        prefix_str = f"-{current_suffix}" if current_suffix else ""
        output_path = f"{output.parent}/{output.stem}{prefix_str}{output.suffix}"
        compression = " (COMPRESSION ZSTD)" if output.suffix == ".parquet" else ""
        max_json_size = 2 * 1024 * 1024  # 2MB
        duckdb.execute(
            f"""
            COPY (
                FROM read_ndjson_auto(
                    '{self._buffer_filepath}',
                    maximum_object_size={max_json_size}
                    )
                )
            TO '{output_path}'{compression};
            """,
        )
        duckdb.execute(f"FROM '{output_path}' LIMIT 1;")  # Validate converted output

    def _start_read_tasks(
        self,
        loop: asyncio.AbstractEventLoop,
        pool: ThreadPoolExecutor,
        path_batch: list[Path],
    ) -> list[asyncio.Future[FitsData]]:
        return [
            loop.run_in_executor(
                pool,
                read_fits,
                path,
                self._config.data_header,
                self._config.data_columns,
                self._config.meta_columns,
            )
            for path in path_batch
        ]

    def _determine_output_file_start_suffix(self, output: Path) -> int | None:
        existent_outputs_paths = [
            path.stem for path in output.parent.glob(f"{output.stem}*{output.suffix}")
        ]

        prefix = None

        if existent_outputs_paths:
            existent_outputs_paths.sort(reverse=True)
            max_old_prefix = existent_outputs_paths[0].split("-")[-1]

            if max_old_prefix.isnumeric():
                prefix = int(max_old_prefix) + 1
            else:
                prefix = 1
        return prefix

    def _validate_input_paths(self, paths: list[Path]) -> None:
        if any(path.suffix not in {".fits"} for path in paths):
            raise UnsupportedFileFormatError(
                "Unsupported input file format. Only 'fits' files are supported",
            )

    def _validate_output_path(self, output: Path) -> None:
        if output.suffix not in (".parquet", ".json"):
            raise UnsupportedFileFormatError(
                "Unsupported output file format. Only 'json, parquet' files are supported",
            )

    def _get_input_paths(self, inputs: PathOrPattern) -> list[Path]:
        paths = set(glob.glob(str(inputs), recursive=True))
        if not paths:
            raise FileNotFoundError(f"No files found matching the pattern '{inputs}'")

        paths -= set(self._checkpoint_filepath.read_text().splitlines())
        return [Path(path) for path in paths]

    def _iterate_path_batches(self, paths: list[Path]) -> Iterator[list[Path]]:
        batch_size = self._config.num_async_readers
        yield from (paths[n : n + batch_size] for n in range(0, len(paths), batch_size))
