import pytest

from gaia.data.models import TCE, KeplerTCE, PeriodicEvent, TceLabel


def test_tce_event__return_correct_value():
    """Test that ."""
    epoch = 1.1
    period = 2.2
    duration = 24
    tce = TCE(
        tce_id=1,
        target_id=1,
        name="tce",
        label_text=None,
        epoch=epoch,
        period=period,
        duration=duration,
    )
    expected = PeriodicEvent(epoch=epoch, duration=duration, period=period)
    assert tce.event == expected


@pytest.mark.parametrize(
    "normalize,duration,expected",
    [
        (True, 24, 1),
        (False, 24, 24),
    ],
    ids=[
        "normalize",
        "dont_normalize",
    ],
)
def test_kepler_tce_post_init__normalize_duration(normalize, duration, expected):
    """Test that transit duration for TCE is normalized when required."""
    tce = KeplerTCE(
        tce_id=1,
        target_id=1,
        name="tce",
        label_text=None,
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


@pytest.mark.parametrize(
    "label_text,expected",
    [
        (None, TceLabel.UNKNOWN),
        ("", TceLabel.UNKNOWN),
        ("invalid_label", TceLabel.UNKNOWN),
        ("PC", TceLabel.PC),
        ("PLANET_CANDIDATE", TceLabel.PC),
    ],
    ids=[
        "from_None",
        "from_empty_string",
        "from_invalid_label",
        "from_enum_name",
        "from_enum_value",
    ],
)
def test_kepler_tce_label__return_correct_value(label_text, expected):
    """Test that label is correctly determined based on its text representation."""
    tce = KeplerTCE(
        tce_id=1,
        target_id=1,
        name="tce",
        label_text=label_text,
        epoch=1.2,
        period=1.2,
        opt_ghost_core_aperture_corr=1.2,
        opt_ghost_halo_aperture_corr=1.2,
        bootstrap_false_alarm_proba=1.2,
        rolling_band_fgt=1.2,
        radius=1.2,
        fitted_period=1.2,
        transit_depth=1.2,
        duration=12,
    )

    assert tce.label == expected
