from collections import Counter
from functools import cached_property
from typing import Any, Generic, TypedDict, TypeVar

from gaia.data.mappers import Mapper, MapperError, map_tce_label
from gaia.data.models import (
    TCE,
    Id,
    PeriodicEvent,
    StellarParameters,
    TceLabel,
    TimeSeries,
)
from gaia.io import (
    ByIdReader,
    DataNotFoundError,
    MissingColumnError,
    ReaderError,
    TableReader,
)


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


class StellarParametersStoreParamsSchema(TypedDict):
    """`StellarParametersStore` methods parameters to data columns names mapping."""

    id: str


class StellarParametersStore(Generic[TStellarParameters]):
    """Stellar physical properties datastore, which provides a convenient way to read properties."""

    def __init__(
        self,
        mapper: Mapper[TStellarParameters],
        reader: TableReader,
        parameters_schema: StellarParametersStoreParamsSchema,
    ) -> None:
        self._mapper = mapper
        self._reader = reader
        self._schema = parameters_schema

    def get(self, id: Id) -> TStellarParameters:
        """Retrieve physical properties for the target star or binary/multiple system.

        Args:
            id (Id): Target ID

        Raises:
            DataNotFoundError: The requested stellar parameters was not found
            KeyError: Required key not found in the store parameters schema
            DataStoreError: Generic error related to mapping or reading the source data

        Returns:
            TStellarParameters: The stellar physical properties of the target star or system
        """
        raw_params = self._read(id)
        mapped_params = self._map(raw_params[0])
        return mapped_params

    def _map(self, result: dict[str, Any]) -> TStellarParameters:
        try:
            return self._mapper(result)
        except MapperError as ex:
            raise DataStoreError(ex)

    def _read(self, id: Id) -> list[dict[str, Any]]:
        try:
            result = self._reader.read(where_sql=f"WHERE {self._schema['id']}={id}")
        except MissingColumnError:
            raise KeyError(
                f"The column {self._schema['id']} corresponding to the 'id' parameter of this "
                "method was not found in the data source",
            )
        except ReaderError as ex:
            raise DataStoreError(f"Cannot get stellar parameters for {id=}: {ex}")

        if not result:
            raise DataNotFoundError(id)

        return result


class TceStoreParamsSchema(TypedDict):
    """`TceStore` methods parameters to data columns names mapping."""

    target_id: str
    tce_id: str
    label: str
    name: str
    epoch: str
    duration: str
    period: str


class TceStore(Generic[TTce]):
    """Threshold-Crossing Events (TCE) datastore, which provides a convenient way to read TCEs."""

    def __init__(
        self,
        mapper: Mapper[TTce],
        reader: TableReader,
        parameters_schema: TceStoreParamsSchema,
    ) -> None:
        self._mapper = mapper
        self._reader = reader
        self._schema = parameters_schema

    def get_by_id(self, target_id: Id, tce_id: Id) -> TTce:
        """Retrieve TCE for the target star or binary/multiple system.

        Args:
            target_id (Id): Target ID
            tce_id (Id): TCE ID

        Raises:
            DataNotFoundError: The requested TCE was not found
            KeyError: Required key not found in the store parameters schema
            DataStoreError: Generic error related to mapping or reading the source data

        Returns:
            TTce: TCE for a specific target and  specified ID
        """
        schema = self._schema
        where = f"WHERE {schema['target_id']}={target_id} AND {schema['tce_id']}={tce_id}"
        try:
            raw_tce = self._reader.read(where_sql=where)
        except MissingColumnError as ex:
            raise KeyError(ex.column)
        except ReaderError as ex:
            raise DataStoreError(f"Cannot get TCE for {target_id=} and {tce_id=}: {ex}")

        if not raw_tce:
            raise DataNotFoundError(f"TCE for {target_id=} and {tce_id=} not found")

        mapped_tce = self._map(raw_tce[0])
        return mapped_tce

    def get_by_name(self, name: str) -> TTce:
        """Retrieve TCE for a specific name.

        Args:
            name (str): The name of TCE. For Kepler dataset it should be Kepler Name no KOI Name

        Raises:
            ValueError: Name is None or an empty string
            DataNotFoundError: Requested TCE was not found
            KeyError: Required key not found in the store parameters schema
            DataStoreError: Generic error related to mapping or reading the source data

        Returns:
            TTce: TCE with the specifed name
        """
        if not name:
            raise ValueError("Parameter 'name' cannot be None or empty string")

        try:
            result = self._reader.read(where_sql=f"WHERE {self._schema['name']}='{name.strip()}'")
        except MissingColumnError as ex:
            raise KeyError(ex.column)
        except ReaderError as ex:
            raise DataStoreError(f"Cannot get TCE for {name=}: {ex}")

        if not result:
            raise DataNotFoundError(f"TCE for {name=} not found")

        return self._map(result[0])

    def get_all_for_target(self, target_id: Id) -> list[TTce]:
        """Retrieve all TCEs for the target star or binary/multiple system.

        Args:
            target_id (Id): Target ID

        Raises:
            KeyError: Required key not found in the store parameters schema
            DataStoreError: Generic error related to mapping or reading the source data

        Returns:
            list[TTce]: All TCEs related to the specified target if any, otherwise an empty list
        """
        try:
            raw_tces = self._reader.read(where_sql=f"WHERE {self._schema['target_id']}={target_id}")
        except MissingColumnError as ex:
            raise KeyError(ex.column)
        except ReaderError as ex:
            raise DataStoreError(f"Cannot get all TCEs for {target_id=}: {ex}")

        mapped_tces = list(map(self._map, raw_tces))
        return mapped_tces

    @property
    def tce_count(self) -> int:
        """The number of all TCEs in the table."""
        try:
            tce_ids = self._reader.read({self._schema["tce_id"]})
        except MissingColumnError as ex:
            raise KeyError(ex.column)
        except ReaderError as ex:
            raise DataStoreError(f"Cannot get all TCE IDs: {ex}")

        return len(tce_ids)

    @property
    def unique_target_ids(self) -> list[Id]:
        """
        The number of unique IDs of the target star or binaries/multiples in ascending order.
        """
        try:
            target_ids = self._reader.read({self._schema["target_id"]})
        except MissingColumnError as ex:
            raise KeyError(ex.column)
        except ReaderError as ex:
            raise DataStoreError(f"Cannot get all target IDs: {ex}")

        unique_target_ids = {list(target_id.values())[0] for target_id in target_ids}
        return sorted(unique_target_ids)

    @cached_property
    def events(self) -> list[tuple[Id, Id, PeriodicEvent]]:
        """Transits for all TCEs in form of `(target_id, tce_id, event)`."""
        try:
            raw_tces = self._reader.read()
        except MissingColumnError as ex:
            raise KeyError(ex.column)
        except ReaderError as ex:
            raise DataStoreError(f"Cannot get all TCEs: {ex}")

        mapped_tces = map(self._map, raw_tces)
        return [(tce.target_id, tce.id, tce.event) for tce in mapped_tces]

    @property
    def labels_distribution(self) -> dict[TceLabel, int]:
        """Distribution of TCE labels. Ambiguous labels are assumed to be `TceLabel.UNKNOWN`."""
        distribution = dict.fromkeys(TceLabel, 0)  # Make sure all labels are present

        try:
            results = self._reader.read([self._schema["label"]])
        except MissingColumnError as ex:
            raise KeyError(ex.column)
        except ReaderError as ex:
            raise DataStoreError(f"Cannot get all TCE labels: {ex}")

        counts = Counter(map(self._map_tce_label, results))
        distribution |= dict(counts)

        if not distribution[TceLabel.FP]:
            distribution[TceLabel.FP] = distribution[TceLabel.NTP] + distribution[TceLabel.AFP]

        return distribution

    def _map_tce_label(self, label: dict[str, str]) -> TceLabel:
        return map_tce_label(list(label.values())[0])

    def _map(self, result: dict[str, Any]) -> TTce:
        try:
            mapped_tce = self._mapper(result)
        except MapperError as ex:
            raise DataStoreError(ex)

        return mapped_tce
