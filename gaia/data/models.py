import abc
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, TypedDict
from uuid import UUID

import numpy as np
import numpy.typing as npt


Id: TypeAlias = str | int | UUID


@dataclass
class PeriodicEvent:
    epoch: float
    """Time of first observed transit-like event."""
    duration: float
    period: float


class TceLabel(Enum):
    PC = "PLANET_CANDIDATE"
    NTP = "NON_TRANSITING_PHENOMENA"
    AFP = "ASTROPHYSICAL_FALSE_POSITIVE"
    UNKNOWN = "UNKNOWN"
    FP = "FALSE_POSITIVE"
    """Generic non-PC label. Prefer AFP and NTP.

    This should only be used when cannot determine whether TCE is AFP or NTP.
    """


@dataclass
class TCE(abc.ABC):
    """Threshold-Crossing Event.

    A sequence of transit-like features in the flux time series of a given target that resembles
    the signature of a transiting planet to a sufficient degree that the target is passed on for
    further analysis.
    """

    id: Id
    target_id: Id
    epoch: float
    duration: float
    period: float
    name: str | None
    label_text: str | None
    """Should be `TceLabel.name|value`.

    The main purpose of this field is to facilitate the initialization of `TCE` from files and
    databases where TCE labels are stored mostly as plain text, not an enum. For all other uses,
    use the `TCE.label` property.
    """

    @property
    def label(self) -> TceLabel:
        """A label that identifies whether a TCE is a planet candidate (PC) or not.

        This is derived from `TCE.label_text`. If `TCE.label_text` is not a valid
        `TceLabel.name|value` or is None or an empty string, this returns `TceLabel.UNKNOWN`.
        """
        if not self.label_text:
            return TceLabel.UNKNOWN

        try:
            return TceLabel(self.label_text)
        except ValueError:
            try:
                return TceLabel[self.label_text]
            except KeyError:
                return TceLabel.UNKNOWN

    @property
    def event(self) -> PeriodicEvent:
        return PeriodicEvent(self.epoch, self.duration, self.period)


@dataclass
class KeplerTCE(TCE):
    opt_ghost_core_aperture_corr: float
    opt_ghost_halo_aperture_corr: float
    bootstrap_false_alarm_proba: float
    rolling_band_fgt: float
    radius: float
    fitted_period: float
    transit_depth: float


@dataclass
class StellarParameters(abc.ABC):
    """Physical properties of the target star or binary/multiple system."""

    id: Id


@dataclass
class KeplerStellarParameters(StellarParameters):
    effective_temperature: float
    radius: float
    mass: float
    density: float
    surface_gravity: float
    metallicity: float


Series: TypeAlias = npt.NDArray[np.float_]


class TimeSeriesBase(TypedDict):
    """Minimum time series representation."""

    id: Id
    time: Series


class PeriodicTimeSeries(TimeSeriesBase):
    """Time series for a single observation period."""

    period: int


class TimeSeries(TimeSeriesBase):
    """Basic time series for multiple observation periods"""

    periods_mask: Series


class KeplerTimeSeries(TimeSeries):
    """Kepler time series for multiple observation periods."""

    pdcsap_flux: Series
    mom_centr1: Series
    mom_centr2: Series
