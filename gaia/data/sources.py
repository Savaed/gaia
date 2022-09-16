"""Data sources for """

# pylint: disable=method-cache-max-size-none

from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from typing import Optional, Protocol, TypeVar, Union

import numpy as np

from gaia.data.models import DictConvertibleObject, TimeSeries
from gaia.enums import Cadence
from gaia.io import Reader, get_kepler_fits_paths


T = TypeVar("T", bound=DictConvertibleObject)


# class KeplerTabularDataSource(Protocol[T]):
#     def get_all(self) -> list[T]:
#         ...

#     def get_by(self, predicate: Callable[[T], bool]) -> list[T]:
#         ...

#     def get_by_kepid(self, kepid: int) -> list[T]:
#         ...


# class CSVKeplerDataSource(Generic[T]):
#     def __init__(self, reader: Reader[pd.DataFrame]) -> None:
#         self.reader = reader

#     def get_all(self) -> list[T]:
#         raise NotImplementedError

#     def get_by(self, predicate: Callable[[T], bool]) -> list[T]:
#         raise NotImplementedError

#     def get_by_kepid(self, kepid: int) -> list[T]:
#         raise NotImplementedError


class KeplerTimeSeriesSource(Protocol):
    def get(self, kepid: int, field: str) -> TimeSeries:
        ...

    def get_for_quarter(self, kepid: int, field: str, quarter: Union[str, int]) -> TimeSeries:
        ...


class FITSReadingError(Exception):
    """Raised when any error in FITS file reading process occurs."""


class FITSKeplerTimeSeriesSource:
    def __init__(
        self,
        reader: Reader[dict[str, np.ndarray]],
        data_dir: str,
        cadence: Cadence,
        time_field: str = "TIME",
    ) -> None:
        self._reader = reader
        self._data_dir = data_dir
        self._cadence = cadence
        self._time_field = time_field

    @cache
    def get(self, kepid: int, field: str) -> TimeSeries:
        data = self._read(kepid, None)
        return TimeSeries(time=data[self._time_field], values=data[field])

    @cache
    def get_for_quarters(self, kepid: int, field: str, quarters: tuple[str, ...]) -> TimeSeries:
        data = self._read(kepid, quarters)
        return TimeSeries(time=data[self._time_field], values=data[field])

    def _read(self, kepid: int, quarters: Optional[tuple[str, ...]]) -> dict[str, list[np.ndarray]]:
        data = defaultdict(list)
        paths = get_kepler_fits_paths(self._data_dir, kepid, self._cadence, quarters)

        for path in paths:
            try:
                file_data = self._reader.read(path)
            except FileNotFoundError:
                # This is expected as there is no file for several quarters
                # TODO: Add logging WARN
                pass
            else:
                for key, value in file_data.items():
                    data[key].append(value)

        if not data:
            # There is not even one file for this KOI
            raise FITSReadingError(f"No files for {kepid=} and cadence='{self._cadence}'")

        return data
