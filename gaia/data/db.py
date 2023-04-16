from collections import defaultdict
from dataclasses import fields
from typing import Any, Generic, Protocol, TypeAlias, TypeVar
from uuid import UUID

import duckdb
import numpy as np

from gaia.data.models import Id, Series, StellarParameters, TimeSeries


_QueryParameter: TypeAlias = str | int | float | complex | UUID
QueryParameters: TypeAlias = dict[str, _QueryParameter] | list[_QueryParameter]
QueryResult: TypeAlias = dict[str, Any]


class DataNotFoundError(Exception):
    """Raised when the requested data was not found."""


class DbContext(Protocol):
    def query(self, query: str, parameters: QueryParameters | None = None) -> list[QueryResult]:
        ...


class DuckDbContext:
    """Database context for in-memory OLAP DuckDB.

    See: https://duckdb.org/
    """

    def __init__(self, db_source: str) -> None:
        self._db_source = db_source

    def query(self, query: str, parameters: QueryParameters | None = None) -> list[QueryResult]:
        """Query the database using the DuckDB SQL dialect with optionally prepared statements.

        Placeholders in prepared statements can only be used for subtitute column value not e.g.
        column name. Valid placeholders are: `?` and named like `$param_name`.

        See more for prepared statements: https://duckdb.org/docs/api/python/dbapi#querying

        Args:
            query (str): Query to run against the database.
            parameters (QueryParameters | None, optional): Parameters to use in prepared statements
                in place of placeholders. For `?` placeholders, it should be a list. For `$name`
                placeholders it should be a dictionary. If the query is raw (has no placeholders),
                it should be `None` or an empty dictionary. Defaults to None.

        Returns:
            list[QueryResult]: List of the resulting records, which may be empty
        """
        db = duckdb.connect(self._db_source)
        result = db.execute(query, parameters).df()
        return result.to_dict("records")  # type: ignore


TTimeSeries = TypeVar("TTimeSeries", bound=TimeSeries)


class TimeSeriesRepository(Generic[TTimeSeries]):
    """A database repository that provides basic time series operations."""

    def __init__(self, db_context: DbContext, time_series_table: str) -> None:
        self._db_context = db_context
        self._table = time_series_table.strip()

    def get(self, target_id: Id, periods: tuple[str, ...] | None = None) -> TTimeSeries:
        """Retrieve time series for target star or binary/multiple system.

        The source table must contain at least the 'id', 'time' and 'period' columns.
        The other columns should have the same names as the `TTimeSeries` keys except
        'periods_mask' which is created based on the 'period' column.

        Args:
            target_id (Id): Target ID
            periods (tuple[str, ...] | None, optional): The observation periods for which time
                series should be returned. If None, the time series for all available periods will
                be returned. Defaults to None.

        Raises:
            DataNotFoundError: The requested time series was not found
            KeyError: `TTimeSeries` required key not found in db table

        Returns:
            TTimeSeries: Dictionary of time series combined from all specified periods. This ensures
                that all keys for `TTimeSeries` are present (all additional keys are ignored).
        """
        periods = periods or ()
        query = self._build_query_string(periods)
        results = self._db_context.query(query, parameters=[str(target_id), *periods])

        if not results:
            periods_msg = f"{periods=}" if periods else "all periods"
            raise DataNotFoundError(f"Time series for {target_id=} and {periods_msg} not found")

        return self._construct_output(target_id, results)

    def _construct_output(self, target_id: Id, results: list[QueryResult]) -> TTimeSeries:
        series_base = TimeSeries(id=str(target_id), time=np.array([]), periods_mask=[])
        tmp_remaining_series: dict[str, list[Series]] = defaultdict(list)
        required_keys = self.__orig_class__.__args__[0].__required_keys__  # type: ignore

        # We expect 'id', 'time' and 'period' to be in `result`
        for result in results:
            del result["id"]  # Not used, set before and remains the same for every result
            current_periods_mask = [result.pop("period")] * len(result["time"])
            series_base["periods_mask"].extend(current_periods_mask)
            series_base["time"] = np.concatenate([series_base["time"], result.pop("time")])

            # Set remaining time series
            for key, series in result.items():
                if key in required_keys:
                    tmp_remaining_series[key].append(series)

        remaining_series = {
            key: np.concatenate(series) for key, series in tmp_remaining_series.items()
        }
        return series_base | remaining_series  # type: ignore

    def _build_query_string(self, periods: tuple[str, ...]) -> str:
        query = f"SELECT * FROM {self._table} WHERE id=?"

        if periods:
            query += f" AND period IN ({'?, ' * len(periods)})"

        return query + ";"


TStellarParameters = TypeVar("TStellarParameters", bound=StellarParameters)


class StellarParametersRepository(Generic[TStellarParameters]):
    """A database repository that provides basic stellar parameters operations."""

    def __init__(self, db_context: DbContext, table_name: str) -> None:
        self._db_context = db_context
        self._table = table_name

    def get(self, id: Id) -> TStellarParameters:
        """Retrieve physical parameters for the target star or binary/multiple system.

        Arguments:
            id (Id): Target ID

        Insteps:
            DataNotFoundError: The requested stellar parameters was not found

        Returns:
            TStellarParameters: The stellar physical properties of the target star or system
        """
        generic_type: type[TStellarParameters] = self.__orig_class__.__args__[0]  # type: ignore
        result = self._db_context.query(f"SELECT * FROM {self._table} WHERE id=?;", [id])

        if not result:
            raise DataNotFoundError(f"Stellar parameters for {id=} not found")

        params_init_keys = [field.name for field in fields(generic_type) if field.init]
        params_init_kwargs = {k: v for k, v in result[0].items() if k in params_init_keys}
        return generic_type(**params_init_kwargs)
