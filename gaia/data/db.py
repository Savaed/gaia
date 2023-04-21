import re
from collections import defaultdict
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any, Generic, Protocol, TypeAlias, TypeVar
from uuid import UUID

import duckdb
import numpy as np

from gaia.data.models import TCE, Id, PeriodicEvent, Series, StellarParameters, TceLabel, TimeSeries


_QueryParameter: TypeAlias = str | int | float | complex | UUID
QueryParameters: TypeAlias = dict[str, _QueryParameter] | list[_QueryParameter]
QueryResult: TypeAlias = dict[str, Any]


class DataNotFoundError(Exception):
    """Raised when the requested data was not found."""


@dataclass
class MissingTableError(Exception):
    """Raised when the table does not exist in the database."""

    table: str


@dataclass
class MissingColumnError(Exception):
    """Raised when the column does not exist in the database table."""

    column: str


class DbRepositoryError(Exception):
    """Raised when there is an error in the database repository."""


class DbContext(Protocol):
    def query(self, query: str, parameters: QueryParameters | None = None) -> list[QueryResult]:
        ...


class DuckDbContext:
    """Database context for in-memory OLAP DuckDB.

    See: https://duckdb.org/
    """

    def __init__(self, db_source: str) -> None:
        self._db_source = db_source.strip()

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

        Raises:
            MissingTableError: The table was not found on the database
            MissingColumnError: The column was not found in the table

        Returns:
            list[QueryResult]: List of the resulting records, which may be empty
        """
        db = duckdb.connect(self._db_source)
        try:
            result = db.execute(query, parameters).df()
        except duckdb.BinderException as ex:
            if "column" in str(ex):
                column = re.search(r'(?<=column )"\w*', str(ex)).group()  # type: ignore
                raise MissingColumnError(column)
            raise
        except duckdb.CatalogException as ex:
            table = re.search(r"(?<=with name )\w*", str(ex)).group()  # type: ignore
            raise MissingTableError(table)

        return result.to_dict("records")  # type: ignore


TTimeSeries = TypeVar("TTimeSeries", bound=TimeSeries)


class TimeSeriesRepository(Generic[TTimeSeries]):
    """A database repository that provides basic time series operations."""

    def __init__(self, db_context: DbContext, table: str) -> None:
        self._db_context = db_context
        self._table = table.strip()

    def get(self, target_id: Id, periods: tuple[str, ...] | None = None) -> TTimeSeries:
        """Retrieve time series for target star or binary/multiple system.

        The source table must contain at least the 'id', 'time' and 'period' columns.
        The other columns should have the same names as the `TTimeSeries` keys except
        'periods_mask' which is created based on the 'period' column(s).

        Args:
            target_id (Id): Target ID
            periods (tuple[str, ...] | None, optional): The observation periods for which time
                series should be returned. If None or empty tuple, the time series for all available
                periods will be returned. Defaults to None.

        Raises:
            DataNotFoundError: The requested time series was not found
            DbRepositoryError: `TTimeSeries` required keys not found in table OR database table not
                found OR one of required columns 'id', 'time' and 'period' not found in the table

        Returns:
            TTimeSeries: Dictionary of time series combined for all specified periods. It is
                guaranteed that all keys for `TTimeSeries` are present (other keys are ignored).
        """
        periods = periods or ()
        query = self._build_query_string(periods)
        try:
            results = self._db_context.query(query, parameters=[str(target_id), *periods])
        except (MissingTableError, MissingColumnError) as ex:
            raise DbRepositoryError(ex)

        if not results:
            periods_msg = f"{periods=}" if periods else "all periods"
            raise DataNotFoundError(f"Time series for {target_id=} and {periods_msg} not found")

        return self._construct_output(target_id, results)

    def _construct_output(self, target_id: Id, results: list[QueryResult]) -> TTimeSeries:
        series_base = TimeSeries(id=str(target_id), time=np.array([]), periods_mask=[])
        tmp_remaining_series: dict[str, list[Series]] = defaultdict(list)
        required_keys: frozenset[str] = self.__orig_class__.__args__[0].__required_keys__  # type: ignore # noqa

        for result in results:
            # 'periods_mask' is obtained from 'period' key of each periodic time series
            missing_required_keys = (
                (set(required_keys) | {"period"}) - {"periods_mask"} - set(result)
            )
            if missing_required_keys:
                raise DbRepositoryError(
                    f"Required keys: '{', '.join(missing_required_keys)}' are missing in db result",
                )

            # We expect at least 'id', 'time' and 'period' to be in `result`
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

    def __init__(self, db_context: DbContext, table: str) -> None:
        self._db_context = db_context
        self._table = table

    def get(self, id: Id) -> TStellarParameters:
        """Retrieve physical properties for the target star or binary/multiple system.

        The source table must contain at least the 'id' column. The other columns should have the
        same names as the `TStellarParameters` fields.

        Arguments:
            id (Id): Target ID

        Raises:
            DataNotFoundError: The requested stellar parameters was not found
            DbRepositoryError: `TStellarParameters` initialize arguments not found in table
                OR database table not found

        Returns:
            TStellarParameters: The stellar physical properties of the target star or system
        """
        generic_type: type[TStellarParameters] = self.__orig_class__.__args__[0]  # type: ignore
        try:
            result = self._db_context.query(f"SELECT * FROM {self._table} WHERE id=?;", [id])
        except (MissingTableError, MissingColumnError) as ex:
            raise DbRepositoryError(ex)

        if not result:
            raise DataNotFoundError(f"Stellar parameters for {id=} not found")

        stellar_init_keys = [field.name for field in fields(generic_type) if field.init]
        stellar_init_kwargs = {k: v for k, v in result[0].items() if k in stellar_init_keys}
        missing_keys = set(stellar_init_keys) - set(stellar_init_kwargs)

        if missing_keys:
            raise DbRepositoryError(
                f"Cannot initialie object of type {generic_type}. Initialize arguments="
                f"'{', '.join(missing_keys)}' not found in the database result retreived from "
                f"table={self._table}",
            )

        return generic_type(**stellar_init_kwargs)


TTce = TypeVar("TTce", bound=TCE)


class TceRepository(Generic[TTce]):
    """A database repository that provides basic TCEs operations.

    The source table MUST contain columns with the same names as the `TTce` fields.
    All other columns will be ignored in the conversion from the database result to `TTce` objects.
    """

    def __init__(self, db_context: DbContext, tce_table: str) -> None:
        self._db_context = db_context
        self._table = tce_table.strip()

    def get_by_id(self, tce_id: Id, target_id: Id) -> TTce:
        """Retrieve TCE for the target star or binary/multiple system.

        Args:
            tce_id (Id): TCE ID
            target_id (Id): Target ID

        Raises:
            DataNotFoundError: The requested TCE was not found
            DbRepositoryError: `TTce` initialize arguments not found in table
                OR database table not found

        Returns:
            TTce: TCE for a specific target and with specified ID
        """
        result = self._query(
            f"SELECT * FROM {self._table} WHERE id=? AND target_id=?;",
            parameters=[tce_id, target_id],
        )
        if not result:
            raise DataNotFoundError(f"TCE for {tce_id=} and {target_id=} not found")

        return self._create_tces(result)[0]

    def get_by_name(self, name: str) -> TTce:
        """Retrieve TCE for a specific name.

        Args:
            name (str): The name of TCE. For Kepler dataset it should be Kepler Name no KOI Name

        Raises:
            ValueError: Name is None or an empty string
            DataNotFoundError: Requested TCE was not found
            DbRepositoryError: `TTce` initialize arguments not found in table
                OR database table not found

        Returns:
            TTce: TCE with the specifed name
        """
        if not name:
            raise ValueError("Parameter 'name' cannot be None or empty string")

        results = self._query(f"SELECT * FROM {self._table} WHERE name=?;", [name])
        if not results:
            raise DataNotFoundError(f"TCE for {name=} not found")

        return self._create_tces(results)[0]

    def get_for_target(self, target_id: Id) -> list[TTce]:
        """Retrieve all TCEs for the target star or binary/multiple system.

        Args:
            target_id (Id): Target ID

        Raises:
            DataNotFoundError: No TCEs for target was found
            DbRepositoryError: `TTce` initialize arguments not found in table
                OR database table not found

        Returns:
            list[TTce]: All TCEs related to the specified target
        """
        results = self._query(f"SELECT * FROM {self._table} WHERE target_id=?;", [target_id])
        if not results:
            raise DataNotFoundError(f"Target for {target_id=} not found")

        return self._create_tces(results)

    @cached_property
    def tce_count(self) -> int:
        """The number of all TCEs in the table.

        Raises:
            DbRepositoryError: Database table not found
        """
        try:
            result = self._db_context.query(f"SELECT count(*) FROM {self._table};")
        except MissingTableError:
            raise DbRepositoryError(f"TCE table {self._table} does not exist")

        count: int = list(result[0].values())[0]
        return count

    @property
    def unique_target_ids(self) -> list[Id]:
        """The number of unique IDs of target star or binary/multiple systems.

        Raises:
            DbRepositoryError: Database table not found OR required column 'target_id' not found
        """
        result = self._query(f"SELECT DISTINCT target_id FROM {self._table} ORDER BY target_id;")
        unique_ids = [list(row.values())[0] for row in result]
        return unique_ids

    @cached_property
    def events(self) -> list[tuple[int, int, PeriodicEvent]]:
        """Transits for all TCEs in form of `(target_id, tce_id, event)`.

        Raises
            DbRepositoryError: Database table not found OR required columns 'target_id', 'id,
                'duration', 'period', 'epoch' not found
        """
        results = self._query(f"SELECT target_id, id, duration, period, epoch FROM {self._table};")
        events = [
            (
                row["target_id"],
                row["id"],
                PeriodicEvent(epoch=row["epoch"], duration=row["duration"], period=row["period"]),
            )
            for row in results
        ]
        return events

    @cached_property
    def labels_distribution(self) -> dict[TceLabel, int]:
        """Distribution of TCE class labels. Ambiguous labels are assumed to be `Tce Label.UNKNOWN`.

        Raises:
            DbRepositoryError: Database table not found OR required column 'label' not found
        """
        distribution = dict.fromkeys(TceLabel, 0)  # Make sure all labels are present

        result = self._query(f'SELECT "label", count(*) FROM {self._table} GROUP BY "label";')
        if not result:
            return distribution

        for label, count in result[0].items():
            if label in [label_.name for label_ in TceLabel]:
                distribution[TceLabel[label]] = count
            elif label in [label_.value for label_ in TceLabel]:
                distribution[TceLabel(label)] = count
            else:
                # Any ambiguous label is assumed to be UNKNOWN
                distribution[TceLabel.UNKNOWN] += count

        if distribution[TceLabel.FP] == 0:
            distribution[TceLabel.FP] = distribution[TceLabel.NTP] + distribution[TceLabel.AFP]

        return distribution

    def _create_tces(self, results: list[QueryResult]) -> list[TTce]:
        generic_type: type[TTce] = self.__orig_class__.__args__[0]  # type: ignore
        tce_init_keys = [field.name for field in fields(generic_type) if field.init]
        missing_keys = set(tce_init_keys) - set(results[0])

        if missing_keys:
            raise DbRepositoryError(
                f"Cannot initialie object of type {generic_type}. Initialize arguments="
                f"'{', '.join(missing_keys)}' not found in the database result retreived from "
                f"table={self._table}",
            )

        tces: list[TTce] = []
        for result in results:
            tce_init_kwargs = {k: v for k, v in result.items() if k in tce_init_keys}
            tces.append(generic_type(**tce_init_kwargs))

        return tces

    def _query(self, query: str, parameters: QueryParameters | None = None) -> list[QueryResult]:
        try:
            return self._db_context.query(query, parameters)
        except (MissingColumnError, MissingTableError) as ex:
            raise DbRepositoryError(ex)
