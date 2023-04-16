from dataclasses import dataclass
from unittest.mock import Mock

import duckdb
import numpy as np
import pytest

from gaia.data.db import (
    DataNotFoundError,
    DbContext,
    DbRepositoryError,
    DuckDbContext,
    MissingColumnError,
    MissingTableError,
    StellarParametersRepository,
    TimeSeriesRepository,
)
from gaia.data.models import Series, StellarParameters, TimeSeries
from tests.conftest import assert_dict_with_numpy_equal


@pytest.fixture
def duckdb_context(mocker):
    """Return an instance of `DuckDbContext` with mocked in-memory db connection and test data."""
    db = duckdb.connect(":memory:")
    db.execute("CREATE TABLE test_table(id INTEGER, col1 VARCHAR, col2 DOUBLE, col3 INTEGER[]);")
    data = [
        (1, "col1_1", 1.2, [1, 2, 3]),
        (2, "col1_2", 1.2, [4, 5, 6]),
        (3, "col1_3", 5.6, [7, 8, 9]),
        (4, "col1_4", 7.8, [10, 11, 12]),
    ]
    db.executemany("INSERT INTO test_table VALUES(?, ?, ?, ?);", parameters=data)
    mocker.patch("gaia.data.db.duckdb.connect", return_value=db)
    # Actually `db source` doesn't matter here because the db connection is mocked anyway
    return DuckDbContext(db_source=":memory:")


@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "SELECT * FROM test_table;",
            [
                {"id": 1, "col1": "col1_1", "col2": 1.2, "col3": [1, 2, 3]},
                {"id": 2, "col1": "col1_2", "col2": 1.2, "col3": [4, 5, 6]},
                {"id": 3, "col1": "col1_3", "col2": 5.6, "col3": [7, 8, 9]},
                {"id": 4, "col1": "col1_4", "col2": 7.8, "col3": [10, 11, 12]},
            ],
        ),
        (
            "SELECT * FROM test_table WHERE id=1;",
            [{"id": 1, "col1": "col1_1", "col2": 1.2, "col3": [1, 2, 3]}],
        ),
        ("SELECT col1, col2 FROM test_table WHERE id=1;", [{"col1": "col1_1", "col2": 1.2}]),
        (
            "SELECT col1 as column1, col2 FROM test_table WHERE id=1;",
            [{"column1": "col1_1", "col2": 1.2}],
        ),
        (
            "SELECT col1, col2 FROM test_table WHERE id IN(1,3);",
            [{"col1": "col1_1", "col2": 1.2}, {"col1": "col1_3", "col2": 5.6}],
        ),
    ],
    ids=["*", "*_where", "select_where", "select_with_alias", "select_where_in"],
)
def test_query__simple_select(query, expected, duckdb_context):
    """Test that simple SQL query returns correct data."""
    result = duckdb_context.query(query)
    assert result == expected


@pytest.mark.parametrize(
    "query,parameters,expected",
    [
        (
            "SELECT col1, col2 FROM test_table WHERE id IN($id1, $id2);",
            dict(id1=1, id2=3),
            [{"col1": "col1_1", "col2": 1.2}, {"col1": "col1_3", "col2": 5.6}],
        ),
        (
            "SELECT col1, col2 FROM test_table WHERE id IN(?, ?);",
            [1, 3],
            [{"col1": "col1_1", "col2": 1.2}, {"col1": "col1_3", "col2": 5.6}],
        ),
    ],
    ids=["params_dict", "params_list"],
)
def test_query__simple_select_with_parameters(query, parameters, expected, duckdb_context):
    """Test that simple SQL query returns correct data using prepared statements."""
    result = duckdb_context.query(query, parameters)
    assert result == expected


def test_query__table_not_found(duckdb_context):
    """Test that `MissingTableError` is raised when table was not found."""
    with pytest.raises(MissingTableError):
        duckdb_context.query("SELECT * FROM not_existent_table;")


def test_query__column_not_found(duckdb_context):
    """Test that `MissingColumnError` is raised when column was not found."""
    with pytest.raises(MissingColumnError):
        duckdb_context.query("SELECT not_existent_column FROM test_table;")


class TimeSeriesTest(TimeSeries):
    new_key: Series


@pytest.mark.parametrize(
    "target_id,periods,context_return_value,expected",
    [
        (
            1,
            None,
            [
                dict(id=1, period="period1", time=[1, 2, 3], new_key=[1, 2, 3]),
                dict(id=1, period="period2", time=[4, 5, 6], new_key=[4, 5, 6]),
            ],
            TimeSeriesTest(
                id="1",
                time=np.array([1, 2, 3, 4, 5, 6]),
                periods_mask=["period1", "period1", "period1", "period2", "period2", "period2"],
                new_key=np.array([1, 2, 3, 4, 5, 6]),
            ),
        ),
        (
            1,
            ("period1",),
            [dict(id=1, period="period1", time=[1, 2, 3], new_key=[1, 2, 3])],
            TimeSeriesTest(
                id="1",
                time=np.array([1, 2, 3]),
                periods_mask=["period1", "period1", "period1"],
                new_key=np.array([1, 2, 3]),
            ),
        ),
        (
            1,
            ("period1", "period2"),
            [
                dict(id=1, period="period1", time=[1, 2, 3], new_key=[1, 2, 3]),
                dict(id=1, period="period2", time=[4, 5, 6], new_key=[4, 5, 6]),
            ],
            TimeSeriesTest(
                id="1",
                time=np.array([1, 2, 3, 4, 5, 6]),
                periods_mask=["period1", "period1", "period1", "period2", "period2", "period2"],
                new_key=np.array([1, 2, 3, 4, 5, 6]),
            ),
        ),
    ],
    ids=["all_periods_by_default", "specific_period", "specific_periods"],
)
def test_time_series_repository_get__return_correct_series(
    target_id,
    periods,
    context_return_value,
    expected,
):
    """Test that correct data is returned."""
    db_context = Mock(spec=DbContext, **{"query.return_value": context_return_value})
    repo = TimeSeriesRepository[TimeSeriesTest](db_context, "test_table")
    result = repo.get(target_id=target_id, periods=periods)
    assert_dict_with_numpy_equal(result, expected)


def test_time_series_repository_get___missing_table():
    """Test that `DbRepositoryError` is raised when no time series table was found."""
    error = "missing_table"
    db_context = Mock(spec=DbContext, **{"query.side_effect": MissingTableError(error)})
    repo = TimeSeriesRepository[TimeSeriesTest](db_context, "test_table")
    with pytest.raises(DbRepositoryError, match=error):
        repo.get(target_id=1)


def test_time_series_repository_get___missing_column():
    """Test that `DbRepositoryError` is raised when no column specified in the query was found."""
    error = "missing_column"
    db_context = Mock(spec=DbContext, **{"query.side_effect": MissingColumnError(error)})
    repo = TimeSeriesRepository[TimeSeriesTest](db_context, "test_table")
    with pytest.raises(DbRepositoryError, match=error):
        repo.get(target_id=1)


def test_time_series_repository_get__time_series_not_found():
    """Test that `DataNotFoundError` is raised when no requested time series was found."""
    db_context = Mock(spec=DbContext, **{"query.return_value": []})
    repo = TimeSeriesRepository[TimeSeries](db_context, "test_table")
    with pytest.raises(DataNotFoundError):
        repo.get(target_id=1)


def test_time_series_repository_get__time_series_key_not_found_in_table():
    """Test that `DbRepositoryError` is raised when time series key was not found in the table."""
    db_context = Mock(spec=DbContext)
    db_context.query.return_value = [{"id": "1"}]  # No required 'time' and 'period' keys
    repo = TimeSeriesRepository[TimeSeries](db_context, "test_table")
    with pytest.raises(DbRepositoryError):
        repo.get(target_id=1)


def test_time_series_repository_get__ignore_optional_keys():
    """Test that ."""
    db_context = Mock(spec=DbContext)
    # `db_context` returns additional key 'additional_key'
    db_context.query.return_value = [
        {"id": "1", "period": "period1", "time": [1, 2, 3], "additional_key": 1.2},
    ]
    repo = TimeSeriesRepository[TimeSeries](db_context, "test_table")
    expected = TimeSeries(
        id="1",
        time=np.array([1, 2, 3]),
        periods_mask=["period1", "period1", "period1"],
    )
    result = repo.get(target_id=1)
    assert_dict_with_numpy_equal(result, expected)


@dataclass
class StellarParametersTesting(StellarParameters):
    name: str


def test_stellar_parameters_repository_get__data_not_found():
    """
    Test that `DataNotFoundError` is raised when stellar parameters object for specified id was not
    found.
    """
    db_context = Mock(spec=DbContext, **{"query.return_value": []})
    repo = StellarParametersRepository[StellarParametersTesting](db_context, "test")
    with pytest.raises(DataNotFoundError):
        repo.get(1)


def test_stellar_parameters_repository_get__return_correct_data():
    """Test that correct stellar parameters object is returned."""
    db_context = Mock(spec=DbContext, **{"query.return_value": [dict(id=1, name="test")]})
    expected = StellarParametersTesting(id=1, name="test")
    repo = StellarParametersRepository[StellarParametersTesting](db_context, "test")
    result = repo.get(1)
    assert result == expected
