from typing import Generic, TypeVar

from gaia.data.mappers import Mapper, MapperError
from gaia.data.models import (
    TCE,
    Id,
    StellarParameters,
    TimeSeries,
)
from gaia.io import ByIdReader, ReaderError


class MissingPeriodsError(Exception):
    """Raised when any of the requested observation periods are missing in the read time series."""


class DataStoreError(Exception):
    """Raised when generic error in data store object occurs."""


TTimeSeries = TypeVar("TTimeSeries", bound=TimeSeries)
TTce = TypeVar("TTce", bound=TCE)
TStellarParameters = TypeVar("TStellarParameters", bound=StellarParameters)


class TimeSeriesStore(Generic[TTimeSeries]):
    """Time series data store, which provides a convenient way to read periodic time series."""

    def __init__(self, mapper: Mapper[TTimeSeries], reader: ByIdReader) -> None:
        self._mapper = mapper
        self._reader = reader

    def get(self, id: Id, periods: tuple[str | int, ...] | None = None) -> list[TTimeSeries]:
        """Retrieve time series for target star or binary/multiple system.

        Args:
            target_id (Id): Target ID
            periods (tuple[str, ...] | None, optional): The observation periods for which time
                series should be returned. If None or empty tuple, the time series for all available
                periods will be returned. Defaults to None.

        Raises:
            DataNotFoundError: The requested time series was not found
            MissingPeriodsError: Any or requested observation periods is missing in read time series
            DataStoreError: Generic error related to mapping or reading the source data

        Returns:
            list[TTimeSeries]: A list of time series, each for an observation period, sorted by a
                period in ascending order
        """
        try:
            results = self._reader.read_by_id(id)
            mapped_time_series = list(map(self._mapper, results))
        except (ReaderError, MapperError) as ex:
            raise DataStoreError(f"Cannot get time series for {id=}: {ex}")

        if periods:
            actual_periods = {series["period"] for series in mapped_time_series}

            if missing_periods := set(periods) - actual_periods:
                raise MissingPeriodsError(missing_periods)

            mapped_time_series = [
                time_series_segment
                for time_series_segment in mapped_time_series
                if time_series_segment["period"] in periods
            ]

        return sorted(mapped_time_series, key=lambda series: series["period"])
