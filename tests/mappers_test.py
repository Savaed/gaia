import numpy as np
import pytest

from gaia.data.mappers import MapperError, map_kepler_time_series
from gaia.data.models import KeplerTimeSeries, RawKeplerTimeSeries
from tests.conftest import assert_dict_with_numpy_equal


def test_map_kepler_time_series__data_mapped_correctly():
    """Test that `RawKeplerTimeSeries` is correctly mapped to `KeplerTimeSeries`."""
    raw_time_series = RawKeplerTimeSeries(
        id=1,
        period=1,
        time=[],
        pdcsap_flux=[4, 5, 6],
        mom_centr1=[7.0, 8.0, 9.0],
        mom_centr2=[10, 11, 12],
    )
    expected = KeplerTimeSeries(
        id=1,
        period=1,
        time=np.array([]),
        pdcsap_flux=np.array([4.0, 5.0, 6.0]),
        mom_centr1=np.array([7.0, 8.0, 9.0]),
        mom_centr2=np.array([10.0, 11.0, 12.0]),
    )
    actual = map_kepler_time_series(raw_time_series)
    assert_dict_with_numpy_equal(actual, expected)


def test_map_kepler_time_series__missing_key_in_source_data():
    """Test that `MapperError` is raised when the source dict has no required key(s)."""
    raw_time_series = dict(
        period=1,
        time=[1, 2, 3],
        pdcsap_flux=[4, 5, 6],
        mom_centr1=[7.0, 8.0, 9.0],
        mom_centr2=[10, 11, 12],
    )
    with pytest.raises(MapperError):
        map_kepler_time_series(raw_time_series)  # type: ignore
