from unittest.mock import Mock

import duckdb
import numpy as np
import pytest

from gaia.data.db import DataNotFoundError, DbContext, DuckDbContext, TimeSeriesRepository
from gaia.data.models import TimeSeries
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
    "query,parameters,expected",
    [
        (
            "SELECT * FROM test_table;",
            None,
            [
                {"id": 1, "col1": "col1_1", "col2": 1.2, "col3": [1, 2, 3]},
                {"id": 2, "col1": "col1_2", "col2": 1.2, "col3": [4, 5, 6]},
                {"id": 3, "col1": "col1_3", "col2": 5.6, "col3": [7, 8, 9]},
                {"id": 4, "col1": "col1_4", "col2": 7.8, "col3": [10, 11, 12]},
            ],
        ),
        (
            "SELECT * FROM test_table WHERE id=1;",
            None,
            [{"id": 1, "col1": "col1_1", "col2": 1.2, "col3": [1, 2, 3]}],
        ),
        ("SELECT col1, col2 FROM test_table WHERE id=1;", None, [{"col1": "col1_1", "col2": 1.2}]),
        (
            "SELECT col1 as column1, col2 FROM test_table WHERE id=1;",
            None,
            [{"column1": "col1_1", "col2": 1.2}],
        ),
        (
            "SELECT col1, col2 FROM test_table WHERE id IN(1,3);",
            None,
            [{"col1": "col1_1", "col2": 1.2}, {"col1": "col1_3", "col2": 5.6}],
        ),
        (
            "SELECT col1, col2 FROM test_table WHERE id IN($id1, $id2);",
            dict(id1=1, id2=3),
            [{"col1": "col1_1", "col2": 1.2}, {"col1": "col1_3", "col2": 5.6}],
        ),
        (
            "SELECT col1, col2, FROM test_table WHERE id IN(?, ?);",
            [1, 3],
            [{"col1": "col1_1", "col2": 1.2}, {"col1": "col1_3", "col2": 5.6}],
        ),
    ],
    ids=[
        "*",
        "*_where",
        "where",
        "with_alias",
        "where_in",
        "prepared_statements_with_dict",
        "prepared_statements_with_list",
    ],
)
def test_query__simple_select(query, parameters, expected, duckdb_context):
    """Test that simple SQL query returns correct data."""
    result = duckdb_context.query(query, parameters)
    assert result == expected


@pytest.mark.parametrize(
    "target_id,periods,context_return_value,expected",
    [
        (
            1,
            None,
            [
                dict(id=1, period="period1", time=[1, 2, 3]),
                dict(id=1, period="period2", time=[4, 5, 6]),
            ],
            TimeSeries(
                id="1",
                time=np.array([1, 2, 3, 4, 5, 6]),
                periods_mask=["period1", "period1", "period1", "period2", "period2", "period2"],
            ),
        ),
        (
            1,
            ("period1",),
            [dict(id=1, period="period1", time=[1, 2, 3])],
            TimeSeries(
                id="1",
                time=np.array([1, 2, 3]),
                periods_mask=["period1", "period1", "period1"],
            ),
        ),
        (
            1,
            ("period1", "period2"),
            [
                dict(id=1, period="period1", time=[1, 2, 3]),
                dict(id=1, period="period2", time=[4, 5, 6]),
            ],
            TimeSeries(
                id="1",
                time=np.array([1, 2, 3, 4, 5, 6]),
                periods_mask=["period1", "period1", "period1", "period2", "period2", "period2"],
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
    repo = TimeSeriesRepository[TimeSeries](db_context, "test_table")
    result = repo.get(target_id=target_id, periods=periods)
    assert_dict_with_numpy_equal(result, expected)


def test_time_series_repository_get__time_series_not_found():
    """Test that `DataNotFoundError` is raised when no requested time series was found."""
    db_context = Mock(spec=DbContext, **{"query.return_value": []})
    repo = TimeSeriesRepository[TimeSeries](db_context, "test_table")
    with pytest.raises(DataNotFoundError):
        repo.get(target_id=1)


def test_time_series_repository_get__time_series_key_not_found_in_table():
    """Test that `KeyError` is raised when no required time series key was found in db table."""
    db_context = Mock(spec=DbContext)
    db_context.query.return_value = [{"id": "1", "period": "1"}]  # No required 'time' key
    repo = TimeSeriesRepository[TimeSeries](db_context, "test_table")
    with pytest.raises(KeyError):
        repo.get(target_id=1)
