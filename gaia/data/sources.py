"""Data sources that provide data regardless of its format or source location.

They provide a convenient way to retrieve data such as time series or various scalar values
without worrying about what the data format is or where it is located.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from gaia.data.models import ID, TCE, PeriodicEvent, StellarParameters
from gaia.io import ReaderError, TableReader, TimeSeriesReader


class DataSourceError(Exception):
    """Raised when there is an error with the data resource itself.

    Mainly this error can be encountered when the data resources are not found, the connection to
    the resources is broken, the user does not have permission to the resources, etc.
    """


class DataNotFoundError(Exception):
    """Raised when requested data is not found."""

    def __init__(self, *ids: ID) -> None:
        self._ids = ids

    def __str__(self) -> str:  # pragma: no cover
        return f"No data found for search parameters: {self._ids}"


@dataclass
class MissingPeriodsError(Exception):
    """Raised when some required time series periods are missing."""

    periods: Iterable[str]

    def __str__(self) -> str:  # pragma: no cover
        missing_periods = ", ".join(self.periods)
        return f"Following periods are missing: {missing_periods}"


# TODO: Add proper type hints in the future
TTimeSeries = TypeVar("TTimeSeries")  # Any dict[str, Sequence[numbers.Number]].


class TimeSeriesSource(Generic[TTimeSeries]):
    """Data source for time series like e.g. light curves.

    The data dictionary from the specified reader must have keys matching the `TTimeSeries` keys.
    """

    def __init__(self, reader: TimeSeriesReader[dict[str, TTimeSeries]]) -> None:
        self._reader = reader

    def get(
        self,
        id_: ID,
        squeeze: bool = False,
        periods: Iterable[str] | None = None,
    ) -> dict[str, TTimeSeries]:
        """Retrieve time series for the particular observation target.

        Args:
            id_ (ID): The id of the target e.g. the id of KOI for Kepler dataset. `str` or `int`.
            squeeze (bool, optional): Whether to flatten series for all series parts into a single
                one. Defaults to False.
            periods (Iterable[str] | None, optional): Which periods (e.g. observation quarters for
                Kepler dataset) of the time series to return. If None then all available are
                returned. Defaults to None.

        Raises:
            DataNotFoundError: No requested time series found
            DataSourceError: The data could not be read due to an internal data resource problem
            MissingDataPartsError: Requested time series periods are unavailable

        Returns:
            dict[str, TTimeSeries]: Data in the format 'series_period' -> 'TTimeSeries'.
                If `squeeze` is set to True then all partial time series (separate series for each
                period of observations) are combined together and returned as a dictionary with
                only one key: 'all' and values as combined periods in `TTimeSeries` format.
        """
        try:
            series_dict = self._reader.read(str(id_))
        except KeyError:
            raise DataNotFoundError(id_)
        except ReaderError as ex:
            raise DataSourceError(ex)

        missing_periods = set(periods or []) - set(series_dict)
        if missing_periods:
            raise MissingPeriodsError(missing_periods)

        series = self._filter_periods(periods, series_dict)
        return self._squeeze(series) if squeeze else series

    def _filter_periods(
        self,
        periods: Iterable[str] | None,
        series_dict: dict[str, TTimeSeries],
    ) -> dict[str, TTimeSeries]:
        return (
            {part: value for part, value in series_dict.items() if part in periods}
            if periods
            else series_dict
        )

    def _squeeze(self, data: dict[str, TTimeSeries]) -> dict[str, TTimeSeries]:
        output: dict[str, list[npt.ArrayLike]] = defaultdict(list)
        for data_parts in data.values():
            for key, series in data_parts.items():  # type: ignore
                output[key].append(series)

        flat_series: dict[str, npt.NDArray[np.float_]] = {
            k: np.concatenate(v) for k, v in output.items()
        }
        return {"all": flat_series}  # type: ignore


TTCE = TypeVar("TTCE", bound=TCE)


class TceSource(Generic[TTCE]):
    """Data source for TCEs.

    The data dictionary from the specified reader must have keys matching the `TTCE` field names.
    """

    def __init__(self, reader: TableReader) -> None:
        self._reader = reader
        self._data: list[TTCE] = []

    @cached_property
    def tce_count(self) -> int:
        """Number of unique TCEs."""
        if not self._data:  # pragma: no cover
            self._load_data()
        return len(set(self._data))

    @cached_property
    def target_unique_ids(self) -> set[ID]:
        """Unique targets IDs."""
        if not self._data:  # pragma: no cover
            self._load_data()
        return {tce.target_id for tce in self._data}

    @cached_property
    def labels_distribution(self) -> dict[str, int]:
        """TCE labels distribution."""
        if not self._data:  # pragma: no cover
            self._load_data()
        return dict(Counter([tce.label for tce in self._data]))

    @cached_property
    def events(self) -> list[PeriodicEvent]:
        if not self._data:  # pragma: no cover
            self._load_data()
        return [tce.event for tce in self._data]

    def get_all_for_target(self, target_id: ID) -> list[TTCE]:
        """Retrieve all TCEs for a particular observation target if any.

        Args:
            target_id (ID): The id of the target e.g. the id of KOI for Kepler dataset.
                `str` or `int`.

        Raises:
            DataNotFoundError: No requested target found
            DataSourceError: The data could not be read due to a data resource or decompression
                error encountered

        Returns:
            list[TTCE]: All TCEs for the target. Can be an empty list
        """
        tces = self._get(lambda tce: tce.target_id == target_id)
        if not tces:
            raise DataNotFoundError(target_id)
        return tces

    def get_by_id(self, target_id: ID, tce_id: ID) -> TTCE:
        """Retrieve Threshold-Crossing Event (TCE) for a particular observation target.

        Args:
            target_id (ID): The id of the target e.g. the id of KOI for Kepler dataset.
                `str` or `int`.
            tce_id (ID): The id of the TCE. Usually TCE number. `str` or `int`

        Raises:
            DataNotFoundError: No requested target or TCE found
            DataSourceError: The data could not be read due to a data resource or decompression
                error encountered

        Returns:
            TTCE: Requested TCE
        """
        tces = self._get(lambda tce: tce.target_id == target_id and tce.tce_id == tce_id)
        if not tces:
            raise DataNotFoundError(target_id, tce_id)

        return tces[0]

    def get_by_name(self, name: str) -> TTCE:
        tces = self._get(lambda tce: tce.name == name)
        if not tces:
            raise DataNotFoundError(name)
        return tces[0]

    def _get(self, predicate: Callable[[TTCE], bool]) -> list[TTCE]:
        if not self._data:  # pragma: no cover
            self._load_data()
        return list(filter(predicate, self._data))

    def _load_data(self) -> None:
        # HACK: This is a `TTCE` real type at runtime. See: https://stackoverflow.com/a/63318205
        generic_type: type[TTCE] = self.__orig_class__.__args__[0]  # type: ignore

        try:
            data = self._reader.read()
            self._data = [generic_type.from_flat_dict(dct) for dct in data]
        except (TypeError, KeyError, ReaderError) as ex:
            raise DataSourceError(ex)


TParams = TypeVar("TParams", bound=StellarParameters)


class StellarParametersSource(Generic[TParams]):
    """Data source for physical properties of observation star or binary/multiple system target.

    The data dictionary from the specified reader must have keys matching the `TParams` keys.
    """

    def __init__(self, reader: TableReader) -> None:
        self._reader = reader
        self._data: list[TParams] = []  # can be a list of dicts to even faster processing

    def get(self, id_: ID) -> TParams:
        """Retrieve physical properties for a particular observation target.

        Args:
            _id (ID): The id of the target e.g. the id of KOI for Kepler dataset. `str` or `int`.

        Raises:
            DataNotFoundError: No requested target found
            DataSourceError: The data could not be read due to a data resource or decompression
                error encountered

        Returns:
            TTCE: Requested physical properties of star or binary/multiple system
        """
        params = self._get(lambda params: params.target_id == id_)
        if not params:
            raise DataNotFoundError(id_)
        return params[0]

    def _get(self, predicate: Callable[[TParams], bool]) -> list[TParams]:
        if not self._data:  # pragma: no cover
            try:
                data = self._reader.read()
            except ReaderError as ex:
                raise DataSourceError(ex)

            # HACK: This is `TParams` real type at runtime.
            # See: https://stackoverflow.com/a/63318205
            generic_type: type[TParams] = self.__orig_class__.__args__[0]  # type: ignore

            try:
                self._data = [generic_type.from_flat_dict(x) for x in data]
            except (TypeError, KeyError) as ex:
                raise DataSourceError(ex)

        return list(filter(predicate, self._data))
