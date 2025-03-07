import asyncio
import glob
import json
import os
import re
from collections import defaultdict
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from re import Pattern
from typing import Protocol, TypeAlias

import duckdb
from rich.status import Status

from gaia.data.models import Id, Series
from gaia.io import Columns, FitsData, JsonNumpyEncoder, read_fits
from gaia.log import logger
from gaia.progress import ProgressBar


PathOrPattern: TypeAlias = Path | str

DUCKDB_COPY_ARGS = {".csv": "(HEADER 1);", ".parquet": "(COMPRESSION ZSTD);"}


class UnsupportedFileFormatError(Exception):
    """Raised when the file is in an unsupported format."""


class ConverterError(Exception):
    """Raised when there is a generic converting error."""


class Converter(Protocol):
    def convert(self, inputs: PathOrPattern, output: Path | str) -> None:
        ...


class AsyncConverter(Protocol):
    async def convert(self, inputs: PathOrPattern, output: Path | str) -> None:
        ...


class CsvConverter:
    _SUPPORTED_OUTPUT_FILES = (".json", ".parquet")
    _TMP_TABLE = "tmp"

    def __init__(self, include_columns: Columns | None = None) -> None:
        self._include_columns = include_columns

    def convert(self, inputs: PathOrPattern, output: Path | str) -> None:
        """Convert a csv file to json or parquet format.

        Args:
            filepath (PathOrPattern): Input csv file path or glob pattern to many csv files
            output (PathOrPattern): Path to the output file
            include_columns (Columns | None, optional): What columns to include in the output file.
            If None then all columns will be included. Defaults to None.

        Raises:
            FileNotFoundError: Input file(s) not found
            ValueError: Unsupported input/output file(s) formats
        """
        output = Path(output)
        log = logger.bind(input=inputs, output=output)
        self._validate_output_file(output)
        input_filepaths = [Path(path) for path in glob.glob(str(inputs), recursive=True)]

        if not input_filepaths:
            raise FileNotFoundError(f"No files found matching the pattern '{inputs}'")

        self._validate_input_files(input_filepaths)
        connection = duckdb.connect(":memory:")
        self._create_tmp_table(inputs, connection)

        copy_args = DUCKDB_COPY_ARGS.get(output.suffix, ";")
        connection.execute(f"COPY {self._TMP_TABLE} TO '{output}' {copy_args}")
        connection.execute(f"FROM '{output}' LIMIT 1;")  # Validate the converted file
        log.info("File converted")

    def _create_tmp_table(
        self,
        inputs: PathOrPattern,
        connection: duckdb.DuckDBPyConnection,
    ) -> None:
        columns = ",".join(self._include_columns) if self._include_columns else "*"
        try:
            connection.execute(
                f"CREATE TABLE {self._TMP_TABLE} AS SELECT {columns} FROM '{inputs}';",
            )
        except duckdb.BinderException as ex:
            column = re.search(r'(?<=column )"\w*', str(ex)).group()  # type: ignore
            raise ValueError(f"{column=} not found in the source CSV {inputs=}")

    def _validate_output_file(self, output: Path) -> None:
        if output.suffix not in self._SUPPORTED_OUTPUT_FILES:
            raise UnsupportedFileFormatError(
                f"Unsupported output file format. Only '{', '.join(self._SUPPORTED_OUTPUT_FILES)}' files are supported",  # noqa
            )

    def _validate_input_files(self, inputs: list[Path]) -> None:
        if any(path.suffix not in {".csv"} for path in inputs):
            raise UnsupportedFileFormatError("Only 'csv' files are supported")


class FitsConvertingOutputFormat(Enum):
    JSON = "json"
    PARQUET = "parquet"


@dataclass
class FitsConvertingSettings:
    """Configration to setup FITS files converting process."""

    data_header: str
    """HDU extension in which data is stored."""
    data_columns: Columns | None
    """Data columns to read."""
    meta_columns: Columns | None
    """Metadata columns to read"""
    path_target_id_pattern: Pattern[str]
    """Regex pattern for extracting an object ID from a file path"""
    output_format: FitsConvertingOutputFormat


_PathGroup: TypeAlias = dict[Id, list[Path]]


class FitsConverter:
    """Time series file converter to convert from FITS to json or parquet format."""

    def __init__(self, settings: FitsConvertingSettings) -> None:
        self._settings = settings
        self._processed_ids: list[Id] = []
        base_path = Path.cwd() / self.__class__.__name__
        self._checkpoint_filepath = Path(f"{base_path}_checkpoint.txt")
        self._buffor_filepath = Path("tmp.json")
        self._copy_params = {
            FitsConvertingOutputFormat.JSON: "",
            FitsConvertingOutputFormat.PARQUET: "(COMPRESSION ZSTD)",
        }

    async def convert(self, inputs: PathOrPattern, output: Path | str) -> None:
        """Convert FITS time series files to json or parquet format.

        Files for which the target ID cannot be retrieved are skipped. All individual files for the
        same target ID will be saved as single output file. This method resumes the conversion from
        the previous point after any failure based on the checkpoint file.

        Args:
            inputs (PathOrPattern): File path or regular expression to input file(s)
            output (Path | str): Output directory (it will be created if not exist)

        Raises:
            FileNotFoundError: Input file not found if passed a single file path as a Path object
            ConverterError: Cannot read FITS files (invalid file/header/columns) OR cannot save
            converted files
            CancelledError: Task has been cancelled
        """
        async with self._handle_tmp_files():
            await self._convert(inputs, Path(output), self._settings.path_target_id_pattern)

    async def _convert(
        self,
        inputs: PathOrPattern,
        output_dir: Path,
        path_target_id_pattern: Pattern[str],
    ) -> None:
        paths_count, input_paths = self._get_input_paths(inputs, path_target_id_pattern)

        if not input_paths:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor() as pool:
            with ProgressBar() as bar:
                convert_task_id = bar.add_task("Converting FITS files", total=paths_count)

                for target_id, paths in input_paths.items():
                    log = logger.bind(id=target_id)

                    log.debug("Reading FITS files")
                    time_series = await self._read(loop, pool, paths)

                    log.debug("Converting FITS files")
                    self._convert_time_series(target_id, time_series, output_dir)  # type: ignore

                    self._processed_ids.append(target_id)
                    bar.advance(convert_task_id, len(paths))
                    log.debug("FITS files converted")

        logger.info("All files converted")

    def _convert_time_series(
        self,
        target_id: Id,
        time_series: list[FitsData],
        output_dir: Path,
    ) -> None:
        tmp_series: list[dict[str, Series | str | int | float]] = []
        for series in time_series:
            tmp_series.append(series)

        with open(self._buffor_filepath, "w") as f:
            json.dump(tmp_series, f, cls=JsonNumpyEncoder)

        output_path, copy_params = self._get_output_params(target_id, output_dir)
        try:
            self._save_converted_file(output_path, copy_params)
        except duckdb.IOException as ex:
            raise ConverterError(ex)

    def _save_converted_file(self, output_path: Path, copy_params: str) -> None:
        duckdb.execute(
            f"""COPY
            (
                FROM read_json_auto(
                    '{self._buffor_filepath.as_posix()}',
                    maximum_object_size=10485760) -- 10MB
            )
            TO '{output_path}' {copy_params};""",
        )
        duckdb.execute(f"FROM '{output_path}' LIMIT 1;")

    def _get_output_params(self, target_id: Id, output_dir: Path) -> tuple[Path, str]:
        copy_params = self._copy_params[self._settings.output_format]
        output_path = output_dir / f"{target_id}.{self._settings.output_format.value}"
        return output_path, copy_params

    async def _read(
        self,
        loop: asyncio.AbstractEventLoop,
        pool: ThreadPoolExecutor,
        paths: list[Path],
    ) -> list[asyncio.Future[FitsData]]:
        tasks: list[asyncio.Future[FitsData]] = []
        try:
            tasks = [
                loop.run_in_executor(
                    pool,
                    read_fits,
                    path,
                    self._settings.data_header,
                    self._settings.data_columns,
                    self._settings.meta_columns,
                )
                for path in paths
            ]
            return asyncio.gather(*tasks)
        except asyncio.CancelledError:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            raise
        except Exception as ex:
            raise ConverterError(f"Cannot convert FITS file. {ex}")

    def _get_input_paths(
        self,
        inputs: PathOrPattern,
        path_target_id_pattern: Pattern[str],
    ) -> tuple[int, _PathGroup]:
        log = logger.bind(inputs=str(inputs))
        paths: _PathGroup = defaultdict(list)
        self._checkpoint_filepath.touch()

        with Status("Searching input files", spinner="earth"):
            if isinstance(inputs, Path) and not inputs.exists():
                raise FileNotFoundError(inputs)

            for path in glob.glob(str(inputs), recursive=True):
                try:
                    # Remove leading zeros.
                    # NOTE: Possible error for GUID with leading zeros.
                    id = path_target_id_pattern.search(path).group().lstrip("0")  # type: ignore
                except AttributeError:
                    log.bind(pattern=path_target_id_pattern.pattern).warning("Cannot retrieve id")
                    continue

                paths[id].append(Path(path))

        if processed_ids := set(self._checkpoint_filepath.read_text().splitlines()):
            logger.bind(ids_count=len(processed_ids)).info("Checkpoint file detected")
            paths = {id: paths for id, paths in paths.items() if id not in processed_ids}

        files_count = sum(map(len, paths.values()))
        log.info(f"Found {files_count} files for {len(paths)} targets")
        return files_count, paths

    @asynccontextmanager
    async def _handle_tmp_files(self) -> AsyncGenerator[None, None]:
        try:
            yield
        finally:
            if self._buffor_filepath.exists():
                os.remove(self._buffor_filepath)

            if self._processed_ids:
                ids = [str(id) + "\n" for id in self._processed_ids]
                with open(self._checkpoint_filepath, "a") as f:
                    f.writelines(ids)
