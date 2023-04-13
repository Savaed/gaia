import numbers
from typing import Any, Protocol, Sequence, TypeAlias

import duckdb


QueryParameters: TypeAlias = dict[str, Any] | list[numbers.Number | str]
QueryResult: TypeAlias = dict[str, numbers.Number | bytes | str | Sequence[Any] | dict[Any, Any]]


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
