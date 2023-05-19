from unittest.mock import Mock

import numpy as np
import pytest

from gaia.data.mappers import MapperError
from gaia.data.models import TimeSeries
from gaia.data.stores import DataStoreError, MissingPeriodsError, TimeSeriesStore
from gaia.io import ByIdReader, DataNotFoundError, ReaderError
from tests.conftest import assert_dict_with_numpy_equal


@pytest.fixture
def time_series_reader():
    """Return a generic time series reader mock."""
    return Mock(spec=ByIdReader)


@pytest.fixture
def mapper():
    """Return a `Mapper` callable that returns the unchanged passed object."""
    return lambda source: source


class TimeSeriesTesting(TimeSeries):
    ...


@pytest.fixture
def create_time_series_store():
    """Factory function to create `TimeSeriesStore[TimeSeriesTesting]` test object."""

    def create(mapper, reader):
        return TimeSeriesStore[TimeSeriesTesting](mapper=mapper, reader=reader)

    return create


def test_time_series_store_get__data_not_found(
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
    TimeSeriesTesting(id=1, period=1, time=np.array([1.0, 2.0, 3.0])),
    TimeSeriesTesting(id=1, period=2, time=np.array([4.0, 5.0, 6.0])),
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
        ((1,), [TimeSeriesTesting(id=1, period=1, time=np.array([1.0, 2.0, 3.0]))]),
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
