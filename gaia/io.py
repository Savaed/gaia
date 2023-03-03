"""I/O functions to use in local or cloud environments (GCP, AWS) or HDFS."""

import io
import numbers
import pickle
from collections.abc import Callable
from enum import Enum
from typing import Any, AnyStr, Generic, Protocol, TypeAlias, TypeVar

import pandas as pd
import tensorflow as tf
from astropy.io import fits
from tensorflow.python.framework import errors_impl as tf_errors


class FileMode(Enum):
    WRITE = "wb"
    WRITE_BINARY = "w"
    READ = "r"
    READ_BINARY = "rb"
    APPEND = "a"


def read(src: str, mode: FileMode = FileMode.READ_BINARY) -> AnyStr:
    """Read file content from local environment cloud (GCP, AWS) or HDFS.

    Args:
        src (str): Path to the source file
        mode (FileMode, optional): Read mode. Defaults to `FileMode.READ_BINARY`.

    Raises:
        FileNotFoundError: File not found
        PermissionError: No permissions for file or cloud environment

    Returns:
        AnyStr: File content as bytes or string
    """
    try:
        with tf.io.gfile.GFile(src, mode.value) as gf:
            return gf.read()  # type: ignore
    except tf_errors.NotFoundError:
        raise FileNotFoundError(src)
    except tf_errors.PermissionDeniedError as ex:
        raise PermissionError(ex)


def write(dest: str, data: AnyStr, mode: FileMode = FileMode.WRITE_BINARY) -> None:
    """Write data to a file in the local environment, cloud (GCP, AWS) or HDFS.

    In some environments, the destination file will be created if it does not exist (e.g. GCP),
    however in the local environment, this function will raise FileNotFoundError.

    Args:
        dest (str): Path to the destination file
        data (AnyStr): Data to write, bytes or string
        mode (FileMode, optional): Write mode. Defaults to `FileMode.WRITE_BINARY`.

    Raises:
        PermissionError: No permissions for file or cloud environment
        FileNotFoundError: File not found (in the local environment)
    """
    try:
        with tf.io.gfile.GFile(dest, mode.value) as gf:
            gf.write(data)
    except tf_errors.NotFoundError:
        raise FileNotFoundError(dest)
    except tf_errors.PermissionDeniedError as ex:
        raise PermissionError(ex)


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
        write(f"{self._tables_dir}/{name}", data)

    def save_time_series(self, name: str, data: bytes) -> None:
        write(f"{self._time_series_dir}/{name}", data)


def read_fits_table(src: str, header: str, fields: set[str] | None = None) -> dict[str, Any]:
    """Read the table from a FITS file under the specified `header`.

    Args:
        src (str): Path to the source file
        header (str): FITS header (HDU extension) to read from
        fields (set[str] | None, optional): Fields to preserve in the output dictionary.
            If None or empty set then all available fields are read. Defaults to None.

    Raises:
        ValueError: Any of specified fields are not present in the FITS file
        KeyError: FITS header not found

    Returns:
        dict[str, Any]: Table from FITS file as dictionary 'file_field' -> 'data'
    """
    with fits.open(io.BytesIO(read(src))) as hdul:  # type: ignore[arg-type]
        data = hdul[header]

        columns = {col.name for col in data.columns}

        if fields:
            not_existent_fields = fields - columns
            if not_existent_fields:
                raise ValueError(f"Fields {not_existent_fields} not present in the FITS file")
        else:
            fields = columns

        return {col: data.data[col] for col in columns if col in fields}


SimpleDict: TypeAlias = dict[str, str | numbers.Number]

T = TypeVar("T")


class ReaderError(Exception):
    """Raised when cannot read data resource (file, database etc.)."""


# T is a dict in form  dict[str, dict[str, Sequence[Number]]]
class TimeSeriesReader(Protocol[T]):  # type: ignore
    def read(self, id_: str) -> T:
        ...


class TableReader(Protocol):
    def read(self) -> list[SimpleDict]:
        ...


class TimeSeriesPickleReader(Generic[T]):
    """A reader for pickled data (data serialized with the `pickle` module), compressed or not.

    This assumes that files are organized in the structure of one file for one object and file
    paths contain the ID of that object e.g. 'some/path/file-{id}.pkl'.
    """

    def __init__(
        self,
        data_dir: str,
        id_path_pattern: str,
        decompression_fn: Callable[[Any], bytes] | None = None,
    ) -> None:
        self._data_dir = data_dir.rstrip("/")
        self._id_pattern = id_path_pattern
        self._decompression_fn = decompression_fn

    def read(self, id_: str) -> T:
        """Read the `.pkl` file for the specified ID. Use decompression if necessary.

        Args:
            id_ (str): The ID of the file/blob that must be extractable from the file path

        Raises:
            DataNotFoundError: No requested data found
            ReaderError: Cannot read data file(s). Mostly due to permission denied, connection
                error (for external resources only), decompression error, etc.

        Returns:
            T: Unpickled data. There is no guarantee that the data read is in `T` format
        """
        if not self._is_data_dir_available():
            raise ReaderError(f"Data directory {self._data_dir!r} is unavalilable")

        filepath = f"{self._data_dir}/{self._id_pattern.replace('{id}', id_)}"
        try:
            raw_bytes: bytes = read(filepath)  # type: ignore
        except PermissionError as ex:
            raise ReaderError(f"Cannot read PKL file {filepath!r}. {ex}")
        except FileNotFoundError:
            raise KeyError(id_)

        try:
            data = self._decompression_fn(raw_bytes) if self._decompression_fn else raw_bytes
        except Exception as ex:
            raise ReaderError(f"Decompression of {filepath!r} failed. {ex}")

        return pickle.loads(data)  # type: ignore[no-any-return]

    def _is_data_dir_available(self) -> bool:
        try:
            is_dir: bool = tf.io.gfile.isdir(self._data_dir)  # Make mypy happy
            return is_dir
        except Exception:  # Permission denied, connection error (for external resource like GCS)
            return False


class CsvTableReader:
    """A reader for tabular data stored in the CSV file format with optional keys mapping."""

    def __init__(self, source: str, mapping: dict[str, str] | None = None) -> None:
        self._source = source
        self._mapping = mapping

    def read(self) -> list[SimpleDict]:
        """Read the entire table stored in the CSV file.

        Raises:
            ReaderError: The CSV file could not be read OR the keys could not be mapped

        Returns:
            list[TSimpleDict]: List of rows as dictionaries (with optionally mapped keys)
        """
        try:
            data = pd.read_csv(io.BytesIO(read(self._source)))  # type: ignore
        except FileNotFoundError:
            raise ReaderError(f"Cannot read CSV file {self._source!r}. File not found")
        except Exception as ex:
            raise ReaderError(f"Cannot read CSV file {self._source!r}. {ex}")

        if self._mapping:
            try:
                data = data.rename(columns=self._mapping, errors="raise")
            except KeyError as ex:
                raise ReaderError(f"Cannot map {ex!r} key using {self._mapping}")

        return data.to_dict("records")  # type: ignore
