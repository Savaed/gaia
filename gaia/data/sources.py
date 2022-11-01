"""Data sources for """

# pylint: disable=method-cache-max-size-none

from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from typing import Protocol, TypeVar

import numpy as np
import structlog

from gaia.data.models import DictConvertibleObject, TimeSeries
from gaia.enums import Cadence
from gaia.io import Reader, get_kepler_fits_paths


T = TypeVar("T", bound=DictConvertibleObject)


logger = structlog.stdlib.get_logger()


@dataclass
class MissingKoiError(Exception):
    """Raised when no KOI found for requested `kepid`."""

    kepid: int

    def __repr__(self) -> str:
        return f"Kepler Object of Interest (KOI) not found with id '{self.kepid}'"

    __str__ = __repr__


@dataclass
class InvalidColumnError(Exception):
    """Raised when there is no such field/column available."""

    column: str

    def __repr__(self) -> str:
        return f"No column '{self.column}' found"

    __str__ = __repr__


class KeplerTimeSeriesSource(Protocol):
    def get(self, kepid: int, field: str) -> TimeSeries:
        ...

    def get_for_quarters(self, kepid: int, field: str, quarters: tuple[str, ...]) -> TimeSeries:
        ...


class FITSKeplerTimeSeriesSource:
    def __init__(
        self,
        reader: Reader[dict[str, np.ndarray]],
        data_dir: str,
        cadence: Cadence,
        time_field: str = "TIME",
    ) -> None:
        self.log = logger.new()
        self._reader = reader
        self._data_dir = data_dir
        self._cadence = cadence
        self._time_field = time_field

    @cache
    def get(self, kepid: int, field: str) -> TimeSeries:
        """Get time series feature for a specific KOI.

        Args:
            kepid (int): ID of KOI
            field (str): Field (column) in FITS file corresponding with time series feature

        Returns:
            TimeSeries: Time series feature for a specific KOI
        """
        return self._read(kepid, field, None)

    @cache
    def get_for_quarters(self, kepid: int, field: str, quarters: tuple[str, ...]) -> TimeSeries:
        """Get time series feature for a specific KOI. Filter by Kepler mission quarters.

        Args:
            kepid (int): ID of KOI
            field (str): Field (column) in FITS file corresponding with time series feature
            quarters (tuple[str, ...]): Quarter prefixes for which time series should be returned

        Returns:
            TimeSeries: Time series feature for a specific KOI and quarters
        """
        return self._read(kepid, field, quarters=quarters)

    def _read(self, kepid: int, field: str, quarters: tuple[str, ...] | None = None) -> TimeSeries:
        """Read time series for a specific KOI and quarters from source FITS files)."""
        data = defaultdict(list)
        paths = get_kepler_fits_paths(self._data_dir, kepid, self._cadence, quarters)

        for path in paths:
            try:
                file_data = self._reader.read(path)
            except FileNotFoundError:
                # This is expected as there is no file for several quarters
                self.log.warning("File not found", path=path)
            else:
                for key, value in file_data.items():
                    data[key].append(value)

        data = dict(data)
        if not data:
            # There are no files for this KOI
            raise MissingKoiError(kepid=kepid)

        try:
            return TimeSeries(time=data[self._time_field], values=data[field])
        except KeyError as ex:
            raise InvalidColumnError(ex.args[0]) from ex
