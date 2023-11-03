import asyncio
import json
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import Any, Protocol, TypeAlias

import duckdb
import google.api_core.exceptions as gcp_exceptions
import numpy as np
from astropy.io import fits
from google.cloud import storage

from gaia.data.models import Id, Series
from gaia.log import logger
from gaia.progress import ProgressBar
from gaia.utils import get_chunks


class DataNotFoundError(Exception):
    """Raised when the requested data was not found."""


class ReaderError(Exception):
    """Raised when the resource cannot be read."""


@dataclass
class MissingColumnError(Exception):
    """Raised when requested column is missing in the source."""

    column: str


class Saver(Protocol):
    def save_table(self, name: str, data: bytes) -> None:
        ...

    def save_time_series(self, name: str, data: bytes) -> None:
        ...


class FileSaver:
    def __init__(self, tables_dir: Path, time_series_dir: Path) -> None:
        self._tables_dir = tables_dir.absolute()
        self._time_series_dir = time_series_dir.absolute()

    def save_table(self, name: str, data: bytes) -> None:
        destination = self._tables_dir / name
        destination.write_bytes(data)

    def save_time_series(self, name: str, data: bytes) -> None:
        destination = self._time_series_dir / name
        destination.write_bytes(data)


FitsData: TypeAlias = dict[str, Series | int | float | str]
Columns: TypeAlias = list[str]


def read_fits(
    filepath: str | Path,
    data_header: str | int,
    columns: Columns | None = None,
    meta: Columns | None = None,
) -> FitsData:
    """Read FITS file data with optional header metadata.

    Args:
        filepath (str | Path): FITS file path
        data_header (str | int): HDU extension for data. If `int` then it is one-indexed
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
        header = fits.getheader(filepath)
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
        data_dir: Path | str,
        id_path_pattern: Pattern[str] | None = None,
        columns: Iterable[str] | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._id_pattern = id_path_pattern
        self._columns = set(columns) if columns else set()

        self._paths = self._get_paths(id_path_pattern)

    def read_by_id(self, id: Id) -> list[dict[str, Any]]:
        """Read the parquet file that contains the ID in the file path.

        Args:
            id (Id): Data identifier. Must be present in the file path

        Raises:
            ResourceNotFoundError: No file found for the specified ID
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

    def _get_paths(self, id_path_pattern: Pattern[str] | None) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        log = logger.bind(id_pattern=id_path_pattern, data_dir=self._data_dir.as_posix())
        log.debug("Searching parquet files")

        for path in self._data_dir.iterdir():
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


class TableReader(Protocol):
    def read(
        self,
        columns: Iterable[str] | None = None,
        where_sql: str | None = None,
    ) -> list[dict[str, Any]]:
        ...


class ParquetTableReader:
    """Parquet table file reader with SQL-like data filtering."""

    def __init__(self, filepath: Path | str) -> None:
        self._connection = duckdb.connect(":memory:")
        self._filepath = Path(filepath)

    def read(
        self,
        columns: Iterable[str] | None = None,
        where_sql: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read a parquet table file.

        Args:
            columns (Iterable[str] | None, optional): Table columns to read. If None or empty then
            read all available columns. Defaults to None.
            where_sql (str | None, optional): SQL-like query to filter read records. Must start with
            'WHERE' statement. Defaults to None.

        Raises:
            MissingColumnError: Column(s) not found in the source file
            ReaderError: Unable to read a file (e.g. corrupted file/file not found)

        Returns:
            list[dict[str, Any]]: Records that match the `where_sql` condition with the specified
            columns read from the source file
        """
        if not self._filepath.is_file():
            raise ReaderError(f"No file {self._filepath} found")

        columns_sql = ", ".join(columns) if columns else "*"
        sql = f"SELECT {columns_sql} FROM read_parquet('{self._filepath}') {where_sql or ''};"
        try:
            results: list[dict[str, Any]] = self._connection.sql(sql).df().to_dict("records")
        except duckdb.BinderException as ex:
            raise MissingColumnError(get_duckdb_missing_column(ex))
        except Exception as ex:
            raise ReaderError(ex)

        return results


def get_duckdb_missing_column(ex: duckdb.BinderException) -> str:
    return re.search('".*"', str(ex)).group().strip("")  # type: ignore


def get_bucket_name(path: str) -> tuple[str, str]:
    """Split the full GCS blob name into the bucket name and the rest of the path.

    Args:
        path (str): Full name of blob e.g. `gs://bucket/folder/blob.txt`

    Returns:
        tuple[str, str]: Bucket name, rest of the blob name
    """
    bucket, _, rest = path.removeprefix("gs://").partition("/")
    return bucket, rest


def get_or_create_bucket(client: storage.Client, bucket_name: str) -> storage.Bucket:
    """Get an existing bucket or create one if it doesn't exist.

    Args:
        client (storage.Client): Google Cloud Storeage client
        bucket_name (str): Name of the bucket

    Returns:
        storage.Bucket: Bucket
    """
    try:
        return client.get_bucket(bucket_name)
    except gcp_exceptions.NotFound:
        return client.create_bucket(bucket_name)


async def copy_to_gcp(source: str | Path, destination: str) -> None:
    """Copy local files to Google Cloud Storage (GCS). Copied files have the same names.

    Args:
        source (str | Path): Local source file or directory path
        destination (str): Destination location on GCS. Should be bucket or folder (when copy
        folder) or blob (when copy single file).
    """
    source = Path(source)

    if not source.exists():
        raise FileNotFoundError(source)

    client = storage.Client()
    bucket_name, rest_path = get_bucket_name(destination)
    bucket = get_or_create_bucket(client, bucket_name)

    if source.is_file():
        bucket.blob(rest_path).upload_from_filename(source.as_posix())
    elif source.is_dir():
        filepaths = list(source.iterdir())

        with ProgressBar() as bar:
            copy_task_id = bar.add_task("Copying files", total=len(filepaths))
            loop = asyncio.get_running_loop()

            with ThreadPoolExecutor() as pool:
                for filepaths_chunk in get_chunks(filepaths, 25):
                    blobs = [
                        bucket.blob(f"{rest_path}/{filepath.name}" if rest_path else filepath.name)
                        for filepath in filepaths_chunk
                    ]
                    tasks = [
                        loop.run_in_executor(pool, blob.upload_from_filename, filepath)
                        for blob, filepath in zip(blobs, filepaths_chunk)
                    ]

                    # Don't consider errors from `tasks`.
                    # await asyncio.gather(tasks, return_exceptions=True)
                    try:
                        await asyncio.gather(*tasks)
                    except Exception as ex:
                        raise KeyError(ex)

                    bar.advance(copy_task_id, len(tasks))
    else:
        raise OSError("Only files and folders are supported")


def check_if_gcs_object_exist(bucket_or_blob_uri: str, check_folder: bool = False) -> bool:
    """Check if bucket, folder or blob exists at given GCS location.

    Unlike a local file system, there are no 'folders' in GCS. The 'folder' can only exists
    virtually when at least blob has it in it's name e.g.: 'gs://bucket/folder/blob.txt' there is a
    'folder'.

    Args:
        bucket_or_blob_uri (str): GCS bucket or blob location. May contain 'gs://' prefix.
        check_folder (bool, optional): Whether to check folder existance rather than blob. Defaults
        to False.

    Returns:
        bool: `True` if bucket, folder or blob exists at given location, `False` otherwise.
    """
    bucket_name, rest_path = get_bucket_name(bucket_or_blob_uri)
    client = storage.Client()

    try:
        bucket = client.get_bucket(bucket_name)
    except gcp_exceptions.NotFound:
        return False  # No bucket at all.

    if rest_path:
        if check_folder:
            blobs = bucket.list_blobs(prefix=rest_path, max_results=1)
            return bool(blobs)  # Folder exists if at least one blob has it in it's name.

        return bucket.blob(rest_path).exists()  # type: ignore

    return True  # `source` points directly to the bucket which we now exists.
