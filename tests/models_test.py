import pytest

from gaia.data.models import KeplerTCE, PeriodicEvent


@pytest.mark.parametrize(
    "normalize,duration,expected",
    [
        (True, 24, 1),
        (False, 24, 24),
    ],
)
def test_kepler_tce__post_init(normalize, duration, expected):
    """Test that ."""
    tce = KeplerTCE(
        tce_id=1,
        target_id=1,
        name="tce",
        label=None,
        epoch=1.2,
        period=1.2,
        opt_ghost_core_aperture_corr=1.2,
        opt_ghost_halo_aperture_corr=1.2,
        bootstrap_false_alarm_proba=1.2,
        rolling_band_fgt=1.2,
        radius=1.2,
        fitted_period=1.2,
        transit_depth=1.2,
        duration=duration,
        _normalize_duration=normalize,
    )

    assert tce.duration == expected


def test_kepler_tce__event():
    """Test that ."""
    epoch = 1.1
    period = 2.2
    duration = 24
    tce = KeplerTCE(
        tce_id=1,
        target_id=1,
        name="tce",
        label=None,
        epoch=epoch,
        period=period,
        duration=duration,
        opt_ghost_core_aperture_corr=1.2,
        opt_ghost_halo_aperture_corr=1.2,
        bootstrap_false_alarm_proba=1.2,
        rolling_band_fgt=1.2,
        radius=1.2,
        fitted_period=1.2,
        transit_depth=1.2,
        _normalize_duration=False,
    )
    excepted = PeriodicEvent(epoch=epoch, duration=duration, period=period)
    assert tce.event == excepted
