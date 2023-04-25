from pathlib import Path
from typing import Collection, Protocol, TypeAlias

from astropy.io import fits

from gaia.data.models import Series


Columns: TypeAlias = Collection[str]


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


def read_fits(
    filepath: str | Path,
    data_header: str | int,
    columns: Columns | None = None,
    meta: Columns | None = None,
) -> dict[str, Series | int | float | str]:
    """Read FITS file data with optional header metadata.

    Args:
        filepath (str | Path): FITS file path
        data_header (str | int): HDU extension for data. If `int` then it is zero-indexed
        columns (Columns | None, optional): Data columns/fields to read. If None then all columns
            will be read. If empty sequence then no data will be returned. Defaults to None.
        meta (Columns | None, optional): Metadata columns/fields to read. Defaults to None.

    Raises:
        KeyError: Invalid data extension OR invalid data/metadata column
        FileNotFoundError: FITS file not found

    Returns:
        dict[str, Series | int | float | str]: Dictionary combined from data and metadata
    """
    metadata = {}

    if meta:
        header = fits.getheader(filepath)
        metadata = {column: header[column] for column in meta}

    if columns is not None and len(columns) == 0:
        return metadata

    data = fits.getdata(filepath, data_header)
    data_columns = [column.name for column in data.columns]

    if columns:
        data_columns = [column for column in data_columns if column in columns]

    return metadata | {column: data[column] for column in data_columns}
