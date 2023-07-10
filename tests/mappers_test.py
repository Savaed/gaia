import numpy as np
import pytest

from gaia.data.mappers import (
    MapperError,
    map_kepler_stallar_parameters,
    map_kepler_tce,
    map_kepler_time_series,
    map_tce_label,
)
from gaia.data.models import (
    KeplerStellarParameters,
    KeplerTCE,
    KeplerTimeSeries,
    PeriodicEvent,
    RawKeplerStellarParameter,
    RawKeplerTce,
    RawKeplerTimeSeries,
    TceLabel,
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
    """Test that `RawKeplerStellarParameter` is correctly mapped to `KeplerStellarParameters`."""
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
        radius=6.0,  # Missing `mass`.
    )
    with pytest.raises(MapperError):
        map_kepler_stallar_parameters(raw_stellar_params)  # type: ignore


@pytest.mark.parametrize(
    "label,expected",
    [
        ("PC", TceLabel.PC),
        ("PLANET_CANDIDATE", TceLabel.PC),
        ("FP", TceLabel.FP),
        ("FALSE_POSITIVE", TceLabel.FP),
        ("AFP", TceLabel.AFP),
        ("ASTROPHYSICAL_FALSE_POSITIVE", TceLabel.AFP),
        ("NTP", TceLabel.NTP),
        ("NON_TRANSITING_PHENOMENA", TceLabel.NTP),
        ("UNKNOWN", TceLabel.UNKNOWN),
        ("TEST", TceLabel.UNKNOWN),
    ],
)
def test_map_tce_label__label_mapped_correctly(label, expected):
    """Test that TCE label is mapped correctly."""
    actual = map_tce_label(label)
    assert actual == expected


@pytest.mark.parametrize("kepler_name", ["kepler_name", None])
def test_map_kepler_tce__data_mapped_correctly(kepler_name):
    """Test that `RawKeplerStellarParameter` is correctly mapped to `KeplerStellarParameters`."""
    raw_tce = RawKeplerTce(
        kepid=1,
        tce_plnt_num=2,
        tce_cap_stat=2.0,
        tce_hap_stat=3.0,
        boot_fap=4.0,
        tce_rb_tcount0=5.0,
        tce_prad=6.0,
        tcet_period=7.0,
        tce_depth=8.0,
        tce_time0bk=9.0,
        tce_duration=24.0,
        tce_period=10.0,
        kepler_name=kepler_name,
        label=TceLabel.PC.name,  # Tce label mapping tested separately.
    )
    expected = KeplerTCE(
        id=2,
        target_id=1,
        name=kepler_name,
        label=TceLabel.PC,
        event=PeriodicEvent(duration=1.0, epoch=9.0, period=10.0),
        opt_ghost_core_aperture_corr=2.0,
        opt_ghost_halo_aperture_corr=3.0,
        bootstrap_false_alarm_proba=4.0,
        rolling_band_fgt=5.0,
        radius=6.0,
        fitted_period=7.0,
        transit_depth=8.0,
    )
    actual = map_kepler_tce(raw_tce)
    assert actual == expected


def test_map_kepler_tce__missing_key_in_source_data():
    """Test that `MapperError` is raised when the source dict has no required key(s)."""
    raw_tce = dict(
        kepid=1,
        tce_cap_stat=2.0,
        tce_hap_stat=3.0,
        boot_fap=4.0,
        tce_rb_tcount0=5.0,
        tce_prad=6.0,
        tcet_period=7.0,
        tce_depth=8.0,
        tce_time0bk=9.0,
        tce_duration=24.0,
        tce_period=10.0,
        kepler_name="kepler_name",
        label=TceLabel.PC.name,
    )  # Missing `tce_plnt_num`.
    with pytest.raises(MapperError):
        map_kepler_tce(raw_tce)  # type: ignore
