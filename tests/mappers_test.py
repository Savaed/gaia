import numpy as np
import pytest

from gaia.data.mappers import MapperError, map_kepler_stallar_parameters, map_kepler_time_series
from gaia.data.models import (
    KeplerStellarParameters,
    KeplerTimeSeries,
    RawKeplerStellarParameter,
    RawKeplerTimeSeries,
)
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
    )  # Missing `id`
    with pytest.raises(MapperError):
        map_kepler_time_series(raw_time_series)  # type: ignore


def test_map_kepler_stallar_parameters__data_mapped_correctly():
    """Test that `RawKeplerTimeSeries` is correctly mapped to `KeplerTimeSeries`."""
    raw_stellar_params = RawKeplerStellarParameter(
        kepid=1,
        teff=2.0,
        dens=3.0,
        logg=4.0,
        feh=5.0,
        radius=6.0,
        mass=7.0,
    )
    expected = KeplerStellarParameters(
        id=1,
        effective_temperature=2.0,
        radius=6.0,
        mass=7.0,
        density=3.0,
        surface_gravity=4.0,
        metallicity=5.0,
    )
    actual = map_kepler_stallar_parameters(raw_stellar_params)
    assert actual == expected


def test_map_kepler_stallar_parameters__missing_key_in_source_data():
    """Test that `MapperError` is raised when the source dict has no required key(s)."""
    raw_stellar_params = dict(
        kepid=1,
        teff=2.0,
        dens=3.0,
        logg=4.0,
        feh=5.0,
        radius=6.0,  # Missing `mass`
    )
    with pytest.raises(MapperError):
        map_kepler_stallar_parameters(raw_stellar_params)  # type: ignore
