from unittest.mock import Mock

import numpy as np
import pytest

from gaia.data.mappers import MapperError
from gaia.data.models import StellarParameters, TimeSeries
from gaia.data.stores import (
    DataStoreError,
    MissingPeriodsError,
    StellarParametersStore,
    StellarParametersStoreParamsSchema,
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
    """Return a `Mapper` callable that returns the unchanged passed object."""
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


def test_time_series_store_get__map_error(time_series_reader, create_time_series_store):
    """Test that `DataStoreError` is raised when generic map error occured."""
    time_series_reader.read_by_id.return_value = TEST_TIME_SERIES

    def mapper(_):
        raise MapperError

    store = create_time_series_store(mapper, time_series_reader)
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
    """Factory function to create `return StellarParametersStore[StellarParameters]` object."""

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
    table_reader.read.side_effect = MissingColumnError
    store = create_stellar_store(mapper, table_reader)
    with pytest.raises(KeyError):
        store.get(1)


TEST_STELLAR_PARAMETERS = [{"id": 1}]


def test_stellar_parameters_store_get__map_error(table_reader, create_stellar_store):
    """Test that `DataStoreError` is raised when map error occured."""
    table_reader.read.return_value = TEST_STELLAR_PARAMETERS

    def mapper(_):
        raise MapperError

    store = create_stellar_store(mapper, table_reader)
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
