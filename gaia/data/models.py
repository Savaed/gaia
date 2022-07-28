"""DTO for Kepler time series stellar parameters and TCE scalar features."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodEvent:
    t0: float
    """Epoch - a time of first observed transit-like event."""
    duration: float
    period: float


@dataclass(frozen=True)
class TCE:
    """
    Threshold-Crossing Event.

    A sequence of transit-like features in the flux time series of a given target that
    resembles the signature of a transiting planet to a sufficient degree that the target is
    passed on for further analysis.
    """

    kepid: int
    tce_num: int
    """The number of observed TCE in the flux time series of a given target."""
    event: PeriodEvent
    opt_ghost_core_aperture_corr_stat: float
    opt_ghost_halo_aperture_corr_stat: float
    bootstrap_false_alarm_proba: float
    rolling_band_fgt: float
    radius: float
    fitted_period: float


@dataclass(frozen=True)
class StellarParameters:
    kepid: int
    effective_temperature: float
    radius: float
    mass: float
    density: float
    surface_gravity: float  # log(g)
    metallicity: float


class TimeSeries:
    """
    Features of the Kepler time series.

    This may or may not include all of the fields available in the FITS file for the target.
    """

    def __init__(self, kepid: int, data: dict[str, str], include: dict[str, str]) -> None:
        if not include:
            raise ValueError(
                "Unable to create an empty KeplerTimeSeries objects. "
                "Specify at least one mapping field in the 'include' parameter"
            )
        self.kepid = kepid

        try:
            for fits_field, prop_name in include.items():
                setattr(TimeSeries, prop_name, data[fits_field])
        except KeyError as ex:
            raise ValueError(f"Key {ex} not found in 'data'")

        self.available_fields = list(include.values())


@dataclass(frozen=True)
class KeplerData:
    time_series: TimeSeries
    stellar_params: StellarParameters
    tces: list[TCE]

    def __post_init__(self):
        self.kepid = self.time_series.kepid
