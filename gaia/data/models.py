import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, MutableMapping, TypeAlias, TypedDict
from uuid import UUID

import numpy as np
import numpy.typing as npt


Id: TypeAlias = str | int | UUID

Series: TypeAlias = npt.NDArray[np.float_]
IntSeries: TypeAlias = npt.NDArray[np.int_]
AnySeries: TypeAlias = npt.NDArray[np.object_]

IterableOfSeries: TypeAlias = Iterable[Series]
ListOfSeries: TypeAlias = list[Series]

BooleanArray: TypeAlias = npt.NDArray[np.bool_]


def flatten_dict(dct: MutableMapping[str, Any]) -> dict[str, Any]:
    items: list[Any] = []
    for k, v in dct.items():
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)


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


class RawKeplerTce(TypedDict):
    kepid: int
    tce_plnt_num: int
    tce_cap_stat: float
    tce_hap_stat: float
    boot_fap: float
    tce_rb_tcount0: float
    tce_prad: float
    tcet_period: float
    tce_depth: float
    tce_time0bk: float
    tce_duration: float
    tce_period: float
    kepler_name: str | None
    label: str


@dataclass
class TCE(abc.ABC):
    """Threshold-Crossing Event.

    A sequence of transit-like features in the flux time series of a given target that resembles
    the signature of a transiting planet to a sufficient degree that the target is passed on for
    further analysis.
    """

    id: Id
    target_id: Id
    name: str | None
    label: TceLabel
    event: PeriodicEvent


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


class RawKeplerStellarParameter(TypedDict):
    kepid: int
    teff: float
    dens: float
    logg: float
    feh: float
    radius: float
    mass: float


@dataclass
class KeplerStellarParameters(StellarParameters):
    effective_temperature: float
    radius: float
    mass: float
    density: float
    surface_gravity: float
    metallicity: float


class RawKeplerTimeSeries(TypedDict):
    id: int
    period: int
    time: list[float]
    pdcsap_flux: list[float]
    mom_centr1: list[float]
    mom_centr2: list[float]


class TimeSeries(TypedDict):
    id: Id
    period: int | str
    time: Series


class KeplerTimeSeries(TimeSeries):
    pdcsap_flux: Series
    mom_centr1: Series
    mom_centr2: Series
