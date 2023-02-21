"""I/O functions to use in local or cloud environments (GCP, AWS) or HDFS."""

import io
import pickle
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, AnyStr, Generic, Protocol, TypeVar

import tensorflow as tf
from astropy.io import fits
from tensorflow.python.framework.errors_impl import NotFoundError, PermissionDeniedError


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
    except NotFoundError:
        raise FileNotFoundError(src)
    except PermissionDeniedError as ex:
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
    except NotFoundError:
        raise FileNotFoundError(dest)
    except PermissionDeniedError as ex:
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
        dict[str, Any]: Table from FITS file as dictionary 'file_field -> data'
    """
    with fits.open(io.BytesIO(read(src))) as hdul:  # type: ignore[arg-type]
        data = hdul[header]
        columns = {col.name for col in data.columns}

        if fields:
            not_existent_fields = fields - columns
            if not_existent_fields:
                s = ", ".join(not_existent_fields)
                raise ValueError(f"Fields {s} not present in the FITS file")
        else:
            fields = columns

        return {col: data.data[col] for col in columns if col in fields}


T = TypeVar("T", bound=dict[Any, Any])


class Reader(Protocol[T]):  # type: ignore[misc]
    def read(self, id_: str) -> T:
        ...


class PickleReader(Generic[T]):
    """A reader for pickled data (data serialized with the `pickle` module), compressed or not.

    This assumes that files are organized in the structure of one file for one object and file
    paths contain the ID of that object e.g. 'some/path/file-{id}.pkl'.
    """

    def __init__(
        self,
        data_dir: str,
        id_path_pattern: str,
        decompression_fn: Callable[[bytes], Any] | None = None,
    ) -> None:
        self._data_dir = data_dir.rstrip("/")
        self._id_pattern = id_path_pattern
        self._decompression_fn = decompression_fn

    def read(self, id_: str) -> T:
        """Read the pickle file for the specified ID.

        Use decompression if necessary (using the function passed to `PickleReader.__init__` if one
        was passed).

        Arguments:
            id_ (str): The ID of the file/blob that must be extractable from the file path

        Returns:
            T: Unpickled data. There is no guarantee that the data read is in `T` format.
        """
        path = Path(f"{self._data_dir}/{self._id_pattern.replace('{id}', id_)}")
        raw_data = (
            self._decompression_fn(path.read_bytes())
            if self._decompression_fn
            else path.read_bytes()
        )
        return pickle.loads(raw_data)  # type: ignore[no-any-return]
