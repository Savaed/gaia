import abc
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeAlias, TypedDict
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
    duration: float
    period: float
    secondary_phase: float | None
    """Secondary phase for secondary transit should be set to `None`."""


class TceLabel(Enum):
    PC = "PLANET_CANDIDATE"
    NTP = "NON_TRANSITING_PHENOMENA"
    AFP = "ASTROPHYSICAL_FALSE_POSITIVE"
    UNKNOWN = "UNKNOWN"
    FP = "FALSE_POSITIVE"
    """Generic non-PC label. Prefer AFP and NTP.

    This should only be used when cannot determine whether TCE is AFP or NTP.
    """


# Think about better mapping
class LabeledTce(TypedDict):
    label: TceLabel | int


def change_tce_label_to_int(tce: LabeledTce) -> dict[str, Any]:
    tce["label"] = 1 if tce["label"] == TceLabel.PC else 0
    return tce  # type: ignore


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
    tce_maxmesd: float
    wst_depth: float


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
    secondary_transit_depth: float


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
    KEPLERID: int
    QUARTER: int
    TIME: list[float]
    PDCSAP_FLUX: list[float]
    MOM_CENTR1: list[float]
    MOM_CENTR2: list[float]


class TimeSeries(TypedDict):
    id: Id
    period: int | str
    time: Series


class KeplerTimeSeries(TimeSeries):
    pdcsap_flux: Series
    mom_centr1: Series
    mom_centr2: Series


NasaTableFormat = Literal["ascii", "pipe-delimited", "xml", "json", "csv"]


@dataclass
class NasaTableRequest:
    """NASA API table request with optional filtering, selecting, ordering, etc.

    See: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
    """

    name: str
    columns: list[str] = field(default_factory=list)
    query: str = ""
    order: str = ""
    format: NasaTableFormat = "csv"

    def __post_init__(self) -> None:
        self.order = self.order.strip()
        self.query = self.query.strip()
        self.name = self.name.strip()
        self.columns = list(map(str.strip, self.columns))

    @property
    def query_string(self) -> str:
        """URL query string that represents this request."""

        url_parts = {
            "table": self.name,
            "format": self.format,
            "where": self.query,
            "order": self.order,
            "select": ",".join(self.columns),
        }
        return "&".join(f"{k}={v}" for k, v in url_parts.items() if v)
