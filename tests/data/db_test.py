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
    TceRepository,
    TimeSeriesRepository,
)
from gaia.data.models import TCE, PeriodicEvent, Series, StellarParameters, TceLabel, TimeSeries
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


@pytest.fixture
def mock_db_context():
    """Factory function to create a mock of `DbContext` class."""

    def _mock(query_result=None, query_side_effect=None):
        db_context_mock = Mock(spec=DbContext)

        if query_result is not None:
            db_context_mock.query.return_value = query_result

        if query_side_effect is not None:
            db_context_mock.query.side_effect = query_side_effect

        return db_context_mock

    return _mock


@pytest.fixture(
    params=[MissingTableError("table1"), MissingColumnError("column1")],
    ids=["missing_table", "missing_column"],
)
def db_context_erroneous(request, mock_db_context):
    """Return a mock of `DbContext` that raise a specific error when call `query()` on it."""
    return mock_db_context(query_side_effect=request.param)


class TimeSeriesTesting(TimeSeries):
    series_values: Series


@pytest.fixture
def create_time_series_repo():
    """Factory function to create an instance of `TimeSeriesRepository[TimeSeriesTesting]` class."""

    def _create(db_context, table="test_table"):
        return TimeSeriesRepository[TimeSeriesTesting](db_context, table)

    return _create


@pytest.mark.parametrize(
    "target_id,periods,context_return_value,expected",
    [
        (
            1,
            None,
            [
                dict(id=1, period=1, time=[1, 2, 3], series_values=[1, 2, 3]),
                dict(id=1, period=2, time=[4, 5, 6], series_values=[4, 5, 6]),
            ],
            TimeSeriesTesting(
                id=1,
                time=np.array([1, 2, 3, 4, 5, 6]),
                periods_mask=np.array([1, 1, 1, 2, 2, 2]),
                series_values=np.array([1, 2, 3, 4, 5, 6]),
            ),
        ),
        (
            1,
            (1,),
            [dict(id=1, period=1, time=[1, 2, 3], series_values=[1, 2, 3])],
            TimeSeriesTesting(
                id=1,
                time=np.array([1, 2, 3]),
                periods_mask=np.array([1, 1, 1]),
                series_values=np.array([1, 2, 3]),
            ),
        ),
        (
            1,
            (1, 2),
            [
                dict(id=1, period=1, time=[1, 2, 3], series_values=[1, 2, 3]),
                dict(id=1, period=2, time=[4, 5, 6], series_values=[4, 5, 6]),
            ],
            TimeSeriesTesting(
                id=1,
                time=np.array([1, 2, 3, 4, 5, 6]),
                periods_mask=np.array([1, 1, 1, 2, 2, 2]),
                series_values=np.array([1, 2, 3, 4, 5, 6]),
            ),
        ),
        (
            1,
            (1,),
            [
                dict(
                    id=1,
                    period=1,
                    time=[1, 2, 3],
                    series_values=[1, 2, 3],
                    additional_key=[1, 2, 3],
                ),
            ],
            TimeSeriesTesting(
                id=1,
                time=np.array([1, 2, 3]),
                periods_mask=np.array([1, 1, 1]),
                series_values=np.array([1, 2, 3]),
            ),
        ),
    ],
    ids=["all_periods_by_default", "specific_period", "specific_periods", "ignore_additional_keys"],
)
def test_time_series_repository_get__return_correct_series(
    target_id,
    periods,
    context_return_value,
    expected,
    mock_db_context,
    create_time_series_repo,
):
    """Test that correct data is returned."""
    context = mock_db_context(context_return_value)
    repo = create_time_series_repo(context)
    result = repo.get(target_id=target_id, periods=periods)
    assert_dict_with_numpy_equal(result, expected)


def test_time_series_repository_get__db_context_error(
    db_context_erroneous,
    create_time_series_repo,
):
    """Test that `DbRepositoryError` is raised when no required table or column was found."""
    repo = create_time_series_repo(db_context_erroneous)
    with pytest.raises(DbRepositoryError, match="column|table"):
        repo.get(target_id=1)


def test_time_series_repository_get__time_series_not_found(
    mock_db_context,
    create_time_series_repo,
):
    """Test that `DataNotFoundError` is raised when no requested time series was found."""
    repo = create_time_series_repo(mock_db_context([]))
    with pytest.raises(DataNotFoundError):
        repo.get(target_id=1)


def test_time_series_repository_get__time_series_key_not_found_in_table(mock_db_context):
    """Test that `DbRepositoryError` is raised when time series key was not found in the table."""
    context = mock_db_context([{"id": "1"}])  # No required 'time' and 'period' keys
    repo = TimeSeriesRepository[TimeSeries](context, "test_table")
    with pytest.raises(DbRepositoryError):
        repo.get(target_id=1)


@dataclass
class StellarParametersTesting(StellarParameters):
    ...


@pytest.fixture
def create_stellar_params_repo():
    """
    Factory function to create an instance of
    `StellarParametersRepository[StellarParametersTesting]` class.
    """

    def _make_repo(db_context, table="test_table"):
        return StellarParametersRepository[StellarParametersTesting](db_context, table)

    return _make_repo


def test_stellar_parameters_repository_get__db_context_error(
    db_context_erroneous,
    create_stellar_params_repo,
):
    """Test that `DbRepositoryError` is raised when no required table or column was found."""
    repo = create_stellar_params_repo(db_context_erroneous)
    with pytest.raises(DbRepositoryError):
        repo.get(1)


def test_stellar_parameters_repository_get__data_not_found(
    mock_db_context,
    create_stellar_params_repo,
):
    """Test that `DataNotFoundError` is raised when stellar parameters was notfound."""
    repo = create_stellar_params_repo(mock_db_context([]))
    with pytest.raises(DataNotFoundError):
        repo.get(1)


def test_stellar_parameters_repository_get__missing_stellar_params_init_arguments(
    mock_db_context,
    create_stellar_params_repo,
):
    """
    Test that `DbRepositoryError` is raised when no `TStellarParameters` initialize arguments
    was found in db result dictionary.
    """
    context = mock_db_context([{"other_key": "abc"}])  # Missing required 'id' key
    repo = create_stellar_params_repo(context)
    with pytest.raises(DbRepositoryError):
        repo.get(1)


@pytest.mark.parametrize(
    "db_result,expected",
    [
        ({"id": 1}, StellarParametersTesting(id=1)),
        ({"id": 1, "additional_key": "abc"}, StellarParametersTesting(id=1)),
    ],
    ids=["normal", "ignore_additional_keys"],
)
def test_stellar_parameters_repository_get__return_correct_stellar_params(
    db_result,
    expected,
    mock_db_context,
    create_stellar_params_repo,
):
    """Test that correct stellar parameters object is returned."""
    context = mock_db_context([db_result])
    repo = create_stellar_params_repo(context)
    result = repo.get(1)
    assert result == expected


@dataclass
class TceTesting(TCE):
    ...


@pytest.fixture
def tce_repo():
    """Factory function to create an instance of `TceRepository[TceTesting]` class."""

    def _make_repo(db_context, table="test_table"):
        return TceRepository[TceTesting](db_context, table)

    return _make_repo


def test_tce_repository_get_by_id__db_context_error(db_context_erroneous, tce_repo):
    """Test that `DbRepositoryError` is raised when no required table or column was found."""
    repo = tce_repo(db_context_erroneous)
    with pytest.raises(DbRepositoryError):
        repo.get_by_id(tce_id=1, target_id=1)


def test_tce_repository_get_by_id__data_not_found(mock_db_context, tce_repo):
    """Test that `DataNotFoundError` is raised when no TCE was found."""
    repo = tce_repo(mock_db_context([]))
    with pytest.raises(DataNotFoundError):
        repo.get_by_id(tce_id=1, target_id=1)


@pytest.fixture
def db_context_missing_tce_init_args(mock_db_context):
    return mock_db_context([dict(tce_id=1, target_id=1, name="tce", label_text="PC")])


def test_tce_repository_get_by_id__missing_tce_init_arguments(
    db_context_missing_tce_init_args,
    tce_repo,
):
    """Test that `DbRepositoryError` is raised when required `TTce__init__()` kwargs are missing."""
    # `DbContext` returns dict with 'epoch', 'duration' and 'period' missing keys
    repo = tce_repo(db_context_missing_tce_init_args)
    with pytest.raises(DbRepositoryError):
        repo.get_by_id(tce_id=1, target_id=1)


@pytest.fixture(params=["normal", "additional_key"])
def tces(request):
    """Return database result (from `DbContext.query()`) and correct TCE."""
    tce = dict(
        id=1,
        target_id=1,
        epoch=1.2,
        duration=3.4,
        period=5.6,
        name="tce",
        label_text="PC",
    )
    tce_with_additional_key = tce | {"additional_key": "abc"}
    tce_object = TceTesting(**tce)  # type: ignore
    data = {"normal": (tce, tce_object), "additional_key": (tce_with_additional_key, tce_object)}
    return data[request.param]


def test_tce_repository_get_by_id__return_correct_tce(tces, mock_db_context, tce_repo):
    """Test that correct TCE is returned."""
    db_result, expected = tces
    context = mock_db_context([db_result])
    repo = tce_repo(context)
    result = repo.get_by_id(tce_id=1, target_id=1)
    assert result == expected


def test_tce_repository_get_by_name__db_context_error(db_context_erroneous, tce_repo):
    """Test that `DbRepositoryError` is raised when no required table or column was found."""
    repo = tce_repo(db_context_erroneous)
    with pytest.raises(DbRepositoryError):
        repo.get_by_name("tce")


def test_tce_repository_get_by_name__data_not_found(mock_db_context, tce_repo):
    """Test that `DataNotFoundError` is raised when no TCE was found."""
    repo = tce_repo(mock_db_context([]))
    with pytest.raises(DataNotFoundError):
        repo.get_by_name("tce")


def test_tce_repository_get_by_name__missing_tce_init_arguments(
    db_context_missing_tce_init_args,
    tce_repo,
):
    """Test that `DbRepositoryError` is raised when required `TTce__init__()` kwargs are missing."""
    repo = tce_repo(db_context_missing_tce_init_args)
    with pytest.raises(DbRepositoryError):
        repo.get_by_name("tce")


@pytest.mark.parametrize("name", [None, ""])
def test_tce_repository_get_by_name__name_is_empty_or_none(name, mock_db_context, tce_repo):
    """ "Test that `ValueError` is raised when specified TCE name is None or an empty string."""
    repo = tce_repo(mock_db_context())
    with pytest.raises(ValueError):
        repo.get_by_name(name)


# TODO: Test for several TCEs with the same name?


def test_tce_repository_get_by_name__return_correct_tce(tces, mock_db_context, tce_repo):
    """Test that correct TCE is returned."""
    db_result, expected = tces
    context = mock_db_context([db_result])
    repo = tce_repo(context)
    result = repo.get_by_name("tce")
    assert result == expected


def test_tce_repository_get_for_target__db_context_error(db_context_erroneous, tce_repo):
    """Test that `DbRepositoryError` is raised when no required table or column was found."""
    repo = tce_repo(db_context_erroneous)
    with pytest.raises(DbRepositoryError):
        repo.get_for_target(1)


def test_tce_repository_get_for_target__target_not_found(mock_db_context, tce_repo):
    """Test that `DataNotFoundError` is raised when no target was found."""
    repo = tce_repo(mock_db_context([]))
    with pytest.raises(DataNotFoundError):
        repo.get_for_target(1)


def test_tce_repository_get_for_target__missing_tce_init_arguments(
    db_context_missing_tce_init_args,
    tce_repo,
):
    """Test that `DbRepositoryError` is raised when required `TTce__init__()` kwargs are missing."""
    repo = tce_repo(db_context_missing_tce_init_args)
    with pytest.raises(DbRepositoryError):
        repo.get_for_target(1)


def test_tce_repository_get_for_target__return_correct_tces(tces, mock_db_context, tce_repo):
    """Test that (all) correct TCE(s) are returned."""
    db_result, expected = tces
    context = mock_db_context([db_result])
    repo = tce_repo(context)
    result = repo.get_by_name("tce")
    assert [result] == [expected]  # `result`, `expected` are a single object, but should be a list


def test_tce_repository_tce_count__table_not_found(mock_db_context, tce_repo):
    """Test that `DbRepositoryError` is raised when no required table was found."""
    context = mock_db_context(query_side_effect=MissingTableError("table"))
    repo = tce_repo(context)
    with pytest.raises(DbRepositoryError):
        repo.tce_count


@pytest.mark.parametrize("count", [0, 9])
def test_tce_repository_tce_count__return_correct_count(count, mock_db_context, tce_repo):
    """Test that correct count of all in TCEs the table is returned."""
    context = mock_db_context([{"count_star()": count}])
    repo = tce_repo(context)
    result = repo.tce_count
    assert result == count


def test_tce_repository_unique_target_ids__db_context_error(db_context_erroneous, tce_repo):
    """Test that `DbRepositoryError` is raised when no required table or column was found."""
    repo = tce_repo(db_context_erroneous)
    with pytest.raises(DbRepositoryError):
        repo.unique_target_ids


@pytest.mark.parametrize("ids", [[], [1], [1, 2]])
def test_tce_repository_unique_target_ids__return_correct_count(ids, mock_db_context, tce_repo):
    """Test that correct count of unique target IDs is returned."""
    db_result = [{"target_id": id_} for id_ in ids]
    context = mock_db_context(db_result)
    repo = tce_repo(context)
    result = repo.unique_target_ids
    assert result == ids


def test_tce_repository_events__db_context_error(db_context_erroneous, tce_repo):
    """Test that `DbRepositoryError` is raised when no required table or column was found."""
    repo = tce_repo(db_context_erroneous)
    with pytest.raises(DbRepositoryError):
        repo.events


@pytest.mark.parametrize(
    "db_result,expected",
    [
        ([], []),
        (
            [{"target_id": 1, "id": 2, "epoch": 1.2, "period": 3.4, "duration": 5.6}],
            [(1, 2, PeriodicEvent(epoch=1.2, duration=5.6, period=3.4))],
        ),
    ],
)
def test_tce_repository_events__return_correct_events(
    db_result,
    expected,
    mock_db_context,
    tce_repo,
):
    """Test that correct events are returned for all TCEs."""
    context = mock_db_context(db_result)
    repo = tce_repo(context)
    result = repo.events
    assert result == expected


def test_tce_repository_labels_distribution__db_context_error(db_context_erroneous, tce_repo):
    """Test that `DbRepositoryError` is raised when no required table or column was found."""
    repo = tce_repo(db_context_erroneous)
    with pytest.raises(DbRepositoryError):
        repo.labels_distribution


def test_tce_repository_labels_distribution__invalid_tce_labels(mock_db_context, tce_repo):
    """Test that any ambiguous/invalid label is treated as `TceLabel.UNKNOW`."""
    context = mock_db_context(
        [
            {"label_text": "PC", "count": 10},
            {"label_text": "abc", "count": 5},
        ],
    )  # Label 'abc' not in TceLabel
    expected = dict.fromkeys(TceLabel, 0) | {TceLabel.PC: 10, TceLabel.UNKNOWN: 5}
    repo = tce_repo(context)
    result = repo.labels_distribution
    assert result == expected


@pytest.mark.parametrize(
    "db_result,expected",
    [
        (
            [
                {"label_text": "PC", "count": 10},
                {"label_text": "NTP", "count": 20},
                {"label_text": "AFP", "count": 30},
                {"label_text": "FP", "count": 40},
                {"label_text": "UNKNOWN", "count": 10},
            ],
            {
                TceLabel.PC: 10,
                TceLabel.NTP: 20,
                TceLabel.AFP: 30,
                TceLabel.FP: 40,
                TceLabel.UNKNOWN: 10,
            },
        ),
        (
            [
                {"labels_text": "PC", "count": 10},
                {"labels_text": "NTP", "count": 20},
                {"labels_text": "AFP", "count": 30},
            ],
            {
                TceLabel.PC: 10,
                TceLabel.NTP: 20,
                TceLabel.AFP: 30,
                TceLabel.FP: 50,
                TceLabel.UNKNOWN: 0,
            },  # FP = AFP + NTP
        ),
        (
            [
                {"labels_text": "NTP", "count": 20},
                {"labels_text": "AFP", "count": 30},
                {"labels_text": "FP", "count": 40},
            ],
            {
                TceLabel.PC: 0,
                TceLabel.NTP: 20,
                TceLabel.AFP: 30,
                TceLabel.FP: 40,
                TceLabel.UNKNOWN: 0,
            },
        ),
        (
            [
                {"label_text": "PLANET_CANDIDATE", "count": 10},
                {"label_text": "NTP", "count": 20},
                {"label_text": "AFP", "count": 30},
                {"label_text": "FP", "count": 40},
            ],
            {
                TceLabel.PC: 10,
                TceLabel.NTP: 20,
                TceLabel.AFP: 30,
                TceLabel.FP: 40,
                TceLabel.UNKNOWN: 0,
            },
        ),
        (
            [
                {"labels_text": "PLANET_CANDIDATE", "count": 10},
                {"labels_text": "NTP", "count": 20},
                {"labels_text": "AFP", "count": 30},
                {"labels_text": "FP", "count": 40},
            ],
            {
                TceLabel.PC: 10,
                TceLabel.NTP: 20,
                TceLabel.AFP: 30,
                TceLabel.FP: 40,
                TceLabel.UNKNOWN: 0,
            },
        ),
        ([], dict.fromkeys(TceLabel, 0)),
    ],
)
def test_tce_repository_labels_distribution__return_correct_distribution(
    db_result,
    expected,
    mock_db_context,
    tce_repo,
):
    """Test that correct distribution is returned for all available TCE labels."""
    context = mock_db_context(db_result)
    repo = tce_repo(context)
    result = repo.labels_distribution
    assert result == expected
