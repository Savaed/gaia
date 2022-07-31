"""Basic I/O for Kepler data stored in various sources such as local or in the cloud."""

import io
import re
from typing import AnyStr, Optional, Protocol

from google.api_core import exceptions as googlex
from google.cloud import storage


_OptionalIO = Optional["DataIO"]


class DataIO(Protocol):
    def read(self, filename: str, mode: str = "rt") -> AnyStr:
        ...

    def write(self, filename: str, data: AnyStr, mode: str = "wt") -> None:
        ...


class LocalIO:
    def read(self, filename: str, mode: str = "rt") -> AnyStr:
        with open(filename, mode, encoding="utf-8") as f:
            return f.read()

    def write(self, filename: str, data: AnyStr, mode: str = "wt") -> None:
        with open(filename, mode, encoding="utf-8") as f:
            f.write(data)


class GcpIO:
    def read(self, filename: str, mode: str = "rt") -> AnyStr:
        blob = self._get_blob(filename)
        try:
            data = blob.download_as_bytes()
        except googlex.NotFound:
            raise FileNotFoundError(filename) from None
        return data if mode[1] == "b" else data.decode()

    def write(self, filename: str, data: AnyStr, mode: str = "wt") -> None:
        write_type, data_type = mode[0], mode[1]

        # As GCP only allows writing a file, if the mode is 'append', the logic of
        # this operation must follow strategy: read an old file, concatenate old
        # file and 'data', write concatenated data to a 'filename' with 'w' mode.
        if write_type == "a":
            try:
                old_data = self.read(filename, mode=f"r{data_type}")
            except googlex.NotFound:
                # File doesn't exist yet.
                old_data = "" if data_type == "t" else b""

            data = old_data + data

            # Change the 'append' mode to the 'write' mode.
            mode = "w" + data_type

        blob = self._get_blob(filename)
        data_stream = io.BytesIO(data) if isinstance(data, bytes) else io.StringIO(data)
        blob.upload_from_file(data_stream)

    def _get_blob(self, filename):
        norm_path = filename.replace("gs://", "")
        bucket, blob_name = norm_path.split("/")[0], "/".join(norm_path.split("/")[1:])
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket)
        return bucket.blob(blob_name)


def check_path_local(path: str) -> _OptionalIO:
    if re.match(r"^\.*/?([a-zA-Z0-9_-]+/)+[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+$", path):
        return LocalIO()


def check_path_gcp(path: str) -> _OptionalIO:
    if re.match(r"^gs://([a-zA-Z0-9_-]+/)+[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+$", path):
        return GcpIO()


DATA_IO_GET_FNS = [check_path_gcp, check_path_local]


def get_io(path: str) -> DataIO:
    for get_io_fn in DATA_IO_GET_FNS:
        if data_io := get_io_fn(path):
            return data_io

    raise ValueError(f"Unsupported {path=}")


def read(filename: str, mode: str = "rt") -> AnyStr:
    return get_io(filename).read(filename, mode)


def write(filename: str, data: AnyStr, mode: str = "wt") -> None:
    get_io(filename).write(filename, data, mode)
