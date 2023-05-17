import json
from pathlib import Path
from typing import Any, Collection, Iterable, Pattern, Protocol, TypeAlias

import duckdb
import numpy as np
from astropy.io import fits

from gaia.data.models import Id, Series
from gaia.log import logger


class DataNotFoundError(Exception):
    """Raised when the requested data was not found."""


class ReaderError(Exception):
    """Raised when the resource cannot be read."""


class Saver(Protocol):
    def save_table(self, name: str, data: bytes) -> None:
        ...

    def save_time_series(self, name: str, data: bytes) -> None:
        ...


class FileSaver:
    def __init__(self, tables_dir: str, time_series_dir: str) -> None:
        self._tables_dir = tables_dir.rstrip("/")
        self._time_series_dir = time_series_dir.rstrip("/")

    def save_table(self, name: str, data: bytes) -> None:
        destination = Path(self._tables_dir) / name
        destination.write_bytes(data)

    def save_time_series(self, name: str, data: bytes) -> None:
        destination = Path(self._time_series_dir) / name
        destination.write_bytes(data)


FitsData: TypeAlias = dict[str, Series | int | float | str]
Columns: TypeAlias = Collection[str]


def read_fits(
    filepath: str | Path,
    data_header: str | int,
    columns: Columns | None = None,
    meta: Columns | None = None,
) -> FitsData:
    """Read FITS file data with optional header metadata.

    Args:
        filepath (str | Path): FITS file path
        data_header (str | int): HDU extension for data. If `int` then it is zero-indexed
        columns (Columns | None, optional): Data columns/fields to read. If None then all columns
            will be read. If empty sequence then no data will be returned. Defaults to None.
        meta (Columns | None, optional): Metadata columns/fields to read. If None then all columns
            will be read. If empty sequence then no data will be returned. Defaults to None.

    Raises:
        KeyError: Invalid data extension OR invalid data/metadata column
        FileNotFoundError: FITS file not found

    Returns:
        FitsData: Dictionary combined from data and metadata of the FITS file
    """
    metadata = {}

    if meta:
        header = dict(fits.getheader(filepath))
        metadata = {column: header[column] for column in meta or header}

    if columns is not None and len(columns) == 0:
        return metadata

    data = fits.getdata(filepath, data_header)
    data_columns = [column.name for column in data.columns]

    if columns:
        data_columns = [column for column in data_columns if column in columns]

    return metadata | {column: data[column] for column in data_columns}


class JsonNumpyEncoder(json.JSONEncoder):
    """Json encoder to serialize numpy arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)  # pragma: no cover


class ByIdReader(Protocol):
    def read_by_id(self, id: Id) -> list[dict[str, Any]]:
        ...


class ParquetReader:
    """Reader that allows reading a parquet file based on identifier contained in the file path."""

    def __init__(
        self,
        data_dir: Path,
        id_path_pattern: Pattern[str] | None = None,
        columns: Iterable[str] | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._id_pattern = id_path_pattern
        self._columns = set(columns) if columns else set()

        self._paths = self._get_paths(data_dir, id_path_pattern)

    def read_by_id(self, id: Id) -> list[dict[str, Any]]:
        """Read the parquet file that contains the ID in the file path.

        Args:
            id (Id): Data identifier. Must be present in the file path

        Raises:
            DataNotFoundError: No data (file) found for the specified ID
            ReaderError: Unable to read parquet file (corrupted file/permission denied etc.)

        Returns:
            list[dict[str, Any]]: List of records contained in the file
        """
        try:
            source_path = self._paths[str(id)]
        except KeyError:
            raise DataNotFoundError(f"No parquet file for {id=} found")

        try:
            raw_results = duckdb.sql(f"FROM '{source_path}';").df().to_dict("records")
        except Exception as ex:
            raise ReaderError(ex)

        results: list[dict[str, Any]] = []

        if self._columns:
            for result in raw_results:
                filtered_result = {
                    key: value for key, value in result.items() if key in self._columns
                }
                results.append(filtered_result)
            # `list(self._columns)` because of loguru bug. See comment in gaia/log.py in
            # format_key_value_context()
            logger.bind(id=id, columns=list(self._columns)).info("Parquet file filtered")
        else:
            results = raw_results

        return results

    def _get_paths(self, data_dir: Path, id_path_pattern: Pattern[str] | None) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        log = logger.bind(id_pattern=id_path_pattern, data_dir=data_dir.as_posix())
        log.debug("Searching parquet files")

        for path in data_dir.iterdir():
            if id_path_pattern:
                try:
                    id = id_path_pattern.search(path.as_posix()).group()  # type: ignore
                except AttributeError:
                    log.bind(path=path.as_posix()).warning("Cannot get ID from file path")
                    continue
            else:
                id = path.stem

            paths[id] = path

        log.debug(f"{len(paths)} files found")
        return paths
