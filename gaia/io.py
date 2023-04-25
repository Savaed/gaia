from pathlib import Path
from typing import Protocol, Sequence, TypeAlias


Columns: TypeAlias = Sequence[str]


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
