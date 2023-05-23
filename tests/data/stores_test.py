from unittest.mock import Mock

import numpy as np
import pytest

from gaia.data.mappers import MapperError
from gaia.data.models import TCE, PeriodicEvent, StellarParameters, TceLabel, TimeSeries
from gaia.data.stores import (
    DataStoreError,
    MissingPeriodsError,
    StellarParametersStore,
    StellarParametersStoreParamsSchema,
    TceStore,
    TceStoreParamsSchema,
    TimeSeriesStore,
)
from gaia.io import ByIdReader, DataNotFoundError, MissingColumnError, ReaderError, TableReader
from tests.conftest import assert_dict_with_numpy_equal


@pytest.fixture
def time_series_reader():
    """Return a generic time series `ByIdReader` mock."""
    return Mock(spec=ByIdReader)


@pytest.fixture
def mapper():
    """
    Return a `Mapper` callable that takes a songle argument of any type and returns the unchanged
    passed object.
    """
    return lambda source: source


@pytest.fixture
def create_time_series_store():
    """Factory function to create `TimeSeriesStore[TimeSeriesTesting]` object."""

    def create(mapper, reader):
        return TimeSeriesStore[TimeSeries](mapper=mapper, reader=reader)

    return create


def test_time_series_store_get__resource_not_found(
    mapper,
    time_series_reader,
    create_time_series_store,
):
    """Test that `DataNotFoundError` is raised when no time series was found."""
    time_series_reader.read_by_id.side_effect = DataNotFoundError
    store = create_time_series_store(mapper, time_series_reader)
    with pytest.raises(DataNotFoundError):
        store.get(1)


def test_time_series_store_get__read_error(mapper, time_series_reader, create_time_series_store):
    """Test that `DataStoreError` is raised when generic read error occured."""
    time_series_reader.read_by_id.side_effect = ReaderError
    store = create_time_series_store(mapper, time_series_reader)
    with pytest.raises(DataStoreError):
        store.get(1)


TEST_TIME_SERIES = [
    TimeSeries(id=1, period=1, time=np.array([1.0, 2.0, 3.0])),
    TimeSeries(id=1, period=2, time=np.array([4.0, 5.0, 6.0])),
]


@pytest.fixture
def erroneous_mapper():
    """
    Return a `Mapper` callable that takes a single argument of any type and raises `MapperError`.
    """

    def mapper(_):
        raise MapperError

    return mapper


def test_time_series_store_get__map_error(
    time_series_reader,
    create_time_series_store,
    erroneous_mapper,
):
    """Test that `DataStoreError` is raised when generic map error occured."""
    time_series_reader.read_by_id.return_value = TEST_TIME_SERIES
    store = create_time_series_store(erroneous_mapper, time_series_reader)
    with pytest.raises(DataStoreError):
        store.get(1)


def test_time_series_store_get__missing_periods(
    mapper,
    time_series_reader,
    create_time_series_store,
):
    """
    Test that 'MissingPeriodsError' is raised when any required observation periods are missing.
    """
    time_series_reader.read_by_id.return_value = TEST_TIME_SERIES
    store = create_time_series_store(mapper, time_series_reader)
    with pytest.raises(MissingPeriodsError):
        # Only periods from 'TEST_TIME_SERIES' are available, in this case 1, 2.
        store.get(1, (9,))


@pytest.mark.parametrize(
    "periods,expected",
    [
        (None, TEST_TIME_SERIES),
        ((1, 2), TEST_TIME_SERIES),
        ((1,), [TimeSeries(id=1, period=1, time=np.array([1.0, 2.0, 3.0]))]),
    ],
    ids=["all_by_default", "all_specified", "single_specified"],
)
def test_time_series_store_get__return_correct_data(
    periods,
    expected,
    mapper,
    time_series_reader,
    create_time_series_store,
):
    """Test that correct list of time series is returned."""
    time_series_reader.read_by_id.return_value = TEST_TIME_SERIES
    store = create_time_series_store(mapper, time_series_reader)
    actual = store.get(1, periods)
    for a, b in zip(actual, expected):
        assert_dict_with_numpy_equal(a, b)


@pytest.fixture
def table_reader():
    """Return a generic `TableReader` mock."""
    return Mock(spec=TableReader)


@pytest.fixture
def create_stellar_store():
    """Factory function to create `StellarParametersStore[StellarParameters]` object."""

    def create(mapper, reader, schema=None):
        return StellarParametersStore[StellarParameters](
            mapper=mapper,
            reader=reader,
            parameters_schema=schema or StellarParametersStoreParamsSchema(id="id"),
        )

    return create


def test_stellar_parameters_store_get__data_not_found(mapper, table_reader, create_stellar_store):
    """Test that `DataNotFoundError` is raised when no requested stallar parameters was found."""
    table_reader.read.return_value = []
    store = create_stellar_store(mapper, table_reader)
    with pytest.raises(DataNotFoundError):
        store.get(1)


def test_stellar_parameters_store_get__read_error(mapper, table_reader, create_stellar_store):
    """Test that `DataStoreError` is raised when generic read error occured."""
    table_reader.read.side_effect = ReaderError
    store = create_stellar_store(mapper, table_reader)
    with pytest.raises(DataStoreError):
        store.get(1)


def test_stellar_parameters_store_get__parameters_schema_error(
    mapper,
    table_reader,
    create_stellar_store,
):
    """Test that `KeyError` is raised when parameters schema is invalid (e.g. column is missing)."""
    table_reader.read.side_effect = MissingColumnError("column")
    store = create_stellar_store(mapper, table_reader)
    with pytest.raises(KeyError):
        store.get(1)


TEST_STELLAR_PARAMETERS = [{"id": 1}]


def test_stellar_parameters_store_get__map_error(
    table_reader,
    create_stellar_store,
    erroneous_mapper,
):
    """Test that `DataStoreError` is raised when map error occured."""
    table_reader.read.return_value = TEST_STELLAR_PARAMETERS
    store = create_stellar_store(erroneous_mapper, table_reader)
    with pytest.raises(DataStoreError):
        store.get(1)


def test_stellar_parameters_store_get__return_correct_data(
    mapper,
    table_reader,
    create_stellar_store,
):
    """Test that correct stellar parameters is returned."""
    table_reader.read.return_value = TEST_STELLAR_PARAMETERS
    store = create_stellar_store(mapper, table_reader)
    actual = store.get(1)
    assert actual == TEST_STELLAR_PARAMETERS[0]


@pytest.fixture
def create_tce_store():
    """Factory function to create `TceStore[TCE]` object."""

    def create(mapper, reader, schema=None):
        return TceStore[TCE](
            mapper=mapper,
            reader=reader,
            parameters_schema=schema
            or TceStoreParamsSchema(
                target_id="target_id",
                tce_id="tce_id",
                label="label",
                duration="duration",
                epoch="epoch",
                name="name",
                period="period",
            ),
        )

    return create


def test_tce_store_get_by_id__read_error(mapper, table_reader, create_tce_store):
    """Test that `DataStoreError` is raised when generic read error occured."""
    table_reader.read.side_effect = ReaderError
    store = create_tce_store(mapper, table_reader)

    with pytest.raises(DataStoreError):
        store.get_by_id(1, 1)


def test_tce_store_get_by_id__data_not_found(mapper, table_reader, create_tce_store):
    """Test that `DataNotFoundError` is raised when no requested TCE was found."""
    table_reader.read.return_value = []
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(DataNotFoundError):
        store.get_by_id(1, 1)


def test_tce_store_get_by_id__parameters_schema_error(mapper, table_reader, create_tce_store):
    """Test that `KeyError` is raised when parameters schema is invalid (e.g. column is missing)."""
    table_reader.read.side_effect = MissingColumnError("column")
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(KeyError):
        store.get_by_id(1, 1)


TEST_TCE = [
    dict(tce_id=1, target_id=1, epoch=1.2, duration=2.3, period=3.4, name="tce1_1", label="PC"),
]


def test_tce_store_get_by_id__map_error(table_reader, create_tce_store, erroneous_mapper):
    """Test that `DataStoreError` is raised when map error occured."""
    table_reader.read.return_value = TEST_TCE
    store = create_tce_store(erroneous_mapper, table_reader)
    with pytest.raises(DataStoreError):
        store.get_by_id(1, 1)


def test_tce_store_get_by_id__return_correct_data(mapper, table_reader, create_tce_store):
    """Test that correct TCE is returned."""
    table_reader.read.return_value = TEST_TCE
    store = create_tce_store(mapper, table_reader)
    actual = store.get_by_id(1, 1)
    assert actual == TEST_TCE[0]


def test_tce_store_get_by_name__read_error(mapper, table_reader, create_tce_store):
    """Test that `DataStoreError` is raised when generic read error occured."""
    table_reader.read.side_effect = ReaderError
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(DataStoreError):
        store.get_by_name("tce1_1")


def test_tce_store_get_by_name__data_not_found(mapper, table_reader, create_tce_store):
    """Test that `DataNotFoundError` is raised when no requested TCE was found."""
    table_reader.read.return_value = []
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(DataNotFoundError):
        store.get_by_name("tce1_1")


def test_tce_store_get_by_name__parameters_schema_error(mapper, table_reader, create_tce_store):
    """Test that `KeyError` is raised when parameters schema is invalid (e.g. column is missing)."""
    table_reader.read.side_effect = MissingColumnError("column")
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(KeyError):
        store.get_by_name("tce1_1")


def test_tce_store_get_by_name__map_error(table_reader, create_tce_store, erroneous_mapper):
    """Test that `DataStoreError` is raised when map error occured."""
    table_reader.read.return_value = TEST_TCE
    store = create_tce_store(erroneous_mapper, table_reader)
    with pytest.raises(DataStoreError):
        store.get_by_name("tce1_1")


@pytest.mark.parametrize("name", [None, ""])
def test_tce_store_get_by_name__name_none_or_empty(name, mapper, table_reader, create_tce_store):
    """Test that correct TCE is returned."""
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(ValueError):
        store.get_by_name(name)


def test_tce_store_get_by_name__return_correct_data(mapper, table_reader, create_tce_store):
    """Test that correct TCE is returned."""
    table_reader.read.return_value = TEST_TCE
    store = create_tce_store(mapper, table_reader)
    actual = store.get_by_name("tce1_1")
    assert actual == TEST_TCE[0]


def test_tce_store_get_all_for_target__read_error(mapper, table_reader, create_tce_store):
    """Test that `DataStoreError` is raised when generic read error occured."""
    table_reader.read.side_effect = ReaderError
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(DataStoreError):
        store.get_all_for_target(1)


def test_tce_store_get_all_for_target__parameters_schema_error(
    mapper,
    table_reader,
    create_tce_store,
):
    """Test that `KeyError` is raised when parameters schema is invalid (e.g. column is missing)."""
    table_reader.read.side_effect = MissingColumnError("column")
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(KeyError):
        store.get_all_for_target(1)


def test_tce_store_get_all_for_target__map_error(table_reader, create_tce_store, erroneous_mapper):
    """Test that `DataStoreError` is raised when map error occured."""
    table_reader.read.return_value = TEST_TCE
    store = create_tce_store(erroneous_mapper, table_reader)
    with pytest.raises(DataStoreError):
        store.get_all_for_target(1)


@pytest.mark.parametrize(
    "read_tces,expected",
    [([], []), [TEST_TCE, TEST_TCE]],
    ids=["empty_list", "tces"],
)
def test_tce_store_get_all_for_target__return_correct_data(
    read_tces,
    expected,
    mapper,
    table_reader,
    create_tce_store,
):
    """Test that correct TCEs is returned."""
    table_reader.read.return_value = read_tces
    store = create_tce_store(mapper, table_reader)
    actual = store.get_all_for_target(1)
    assert actual == expected


def test_tce_store_tce_count__read_error(mapper, table_reader, create_tce_store):
    """Test that `DataStoreError` is raised when generic read error occured."""
    table_reader.read.side_effect = ReaderError
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(DataStoreError):
        store.tce_count


def test_tce_store_tce_count__parameters_schema_error(mapper, table_reader, create_tce_store):
    """Test that `KeyError` is raised when parameters schema is invalid (e.g. column is missing)."""
    table_reader.read.side_effect = MissingColumnError("columns")
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(KeyError):
        store.tce_count


@pytest.mark.parametrize(
    "read_tce_ids,expected",
    [
        ([], 0),
        ([dict(target_id=1), dict(target_id=2)], 2),
        ([dict(target_id=1), dict(target_id=1)], 2),
        ([dict(target_id=1), dict(target_id=1), dict(target_id=2)], 3),
    ],
    ids=["empty_list", "all_unique", "all_duplicates", "some_duplicates"],
)
def test_tce_store_tce_count__return_correct_data(
    read_tce_ids,
    expected,
    mapper,
    table_reader,
    create_tce_store,
):
    table_reader.read.return_value = read_tce_ids
    store = create_tce_store(mapper, table_reader)
    actual = store.tce_count
    assert actual == expected


def test_tce_store_unique_target_ids__read_error(mapper, table_reader, create_tce_store):
    """Test that `DataStoreError` is raised when generic read error occured."""
    table_reader.read.side_effect = ReaderError
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(DataStoreError):
        store.unique_target_ids


def test_tce_store_unique_target_ids__parameters_schema_error(
    mapper,
    table_reader,
    create_tce_store,
):
    """Test that `KeyError` is raised when parameters schema is invalid (e.g. column is missing)."""
    table_reader.read.side_effect = MissingColumnError("columns")
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(KeyError):
        store.unique_target_ids


@pytest.mark.parametrize(
    "read_tce_ids,expected",
    [
        ([], []),
        ([dict(target_id=1), dict(target_id=2)], [1, 2]),
        ([dict(target_id=1), dict(target_id=1)], [1]),
        ([dict(target_id=1), dict(target_id=1), dict(target_id=2)], [1, 2]),
    ],
    ids=["empty_list", "all_unique", "all_duplicates", "some_duplicates"],
)
def test_tce_store_unique_target_ids__return_correct_data(
    read_tce_ids,
    expected,
    mapper,
    table_reader,
    create_tce_store,
):
    table_reader.read.return_value = read_tce_ids
    store = create_tce_store(mapper, table_reader)
    actual = store.unique_target_ids
    assert actual == expected


def test_tce_store_events__read_error(mapper, table_reader, create_tce_store):
    """Test that `DataStoreError` is raised when generic read error occured."""
    table_reader.read.side_effect = ReaderError
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(DataStoreError):
        store.events


def test_tce_store_events__parameters_schema_error(mapper, table_reader, create_tce_store):
    """Test that `KeyError` is raised when parameters schema is invalid (e.g. column is missing)."""
    table_reader.read.side_effect = MissingColumnError("columns")
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(KeyError):
        store.events


@pytest.fixture
def tce_event_mapper():
    def mapper(tce):
        return TCE(
            name=tce["name"],
            id=tce["tce_id"],
            target_id=tce["target_id"],
            label=TceLabel.PC,
            event=PeriodicEvent(tce["epoch"], tce["duration"], tce["period"]),
        )

    return mapper


@pytest.mark.parametrize(
    "read_tce_ids,expected",
    [([], []), (TEST_TCE, [(1, 1, PeriodicEvent(epoch=1.2, duration=2.3, period=3.4))])],
    ids=["empty_list", "tces"],
)
def test_tce_store_events__return_correct_data(
    read_tce_ids,
    expected,
    tce_event_mapper,
    table_reader,
    create_tce_store,
):
    table_reader.read.return_value = read_tce_ids
    store = create_tce_store(tce_event_mapper, table_reader)
    actual = store.events
    assert actual == expected


def test_tce_store_labels_distribution__read_error(mapper, table_reader, create_tce_store):
    """Test that `DataStoreError` is raised when generic read error occured."""
    table_reader.read.side_effect = ReaderError
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(DataStoreError):
        store.labels_distribution


def test_tce_store_labels_distribution__parameters_schema_error(
    mapper,
    table_reader,
    create_tce_store,
):
    """Test that `KeyError` is raised when parameters schema is invalid (e.g. column is missing)."""
    table_reader.read.side_effect = MissingColumnError("columns")
    store = create_tce_store(mapper, table_reader)
    with pytest.raises(KeyError):
        store.labels_distribution


@pytest.mark.parametrize(
    "read_labels,expected",
    [
        (
            [],
            {TceLabel.PC: 0, TceLabel.AFP: 0, TceLabel.NTP: 0, TceLabel.FP: 0, TceLabel.UNKNOWN: 0},
        ),
        (
            [dict(label="PC")],
            {TceLabel.PC: 1, TceLabel.AFP: 0, TceLabel.NTP: 0, TceLabel.FP: 0, TceLabel.UNKNOWN: 0},
        ),
        (
            [
                dict(label="PC"),
                dict(label="AFP"),
                dict(label="NTP"),
                dict(label="FP"),
                dict(label="UNKNOWN"),
            ],
            {TceLabel.PC: 1, TceLabel.AFP: 1, TceLabel.NTP: 1, TceLabel.FP: 1, TceLabel.UNKNOWN: 1},
        ),
        (
            [dict(label="AFP"), dict(label="NTP")],
            {TceLabel.PC: 0, TceLabel.AFP: 1, TceLabel.NTP: 1, TceLabel.FP: 2, TceLabel.UNKNOWN: 0},
        ),
        (
            [dict(label="PC"), dict(label="PC")],
            {TceLabel.PC: 2, TceLabel.AFP: 0, TceLabel.NTP: 0, TceLabel.FP: 0, TceLabel.UNKNOWN: 0},
        ),
        (
            [dict(label="test"), dict(label="UNKNOWN")],
            {TceLabel.PC: 0, TceLabel.AFP: 0, TceLabel.NTP: 0, TceLabel.FP: 0, TceLabel.UNKNOWN: 2},
        ),
    ],
)
def test_tce_store_labels_distribution__return_correct_data(
    read_labels,
    expected,
    mapper,
    table_reader,
    create_tce_store,
):
    """Test that correct TCE labels distribution is returned and FP is computed if required."""
    table_reader.read.return_value = read_labels
    store = create_tce_store(mapper, table_reader)
    actual = store.labels_distribution
    assert actual == expected
