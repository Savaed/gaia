import abc
from abc import ABC
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Iterable, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt


class FromDictMixin(ABC):
    """Abstract class that provides a method to map flat dictionary to the dataclass object."""

    # TODO: Use '() -> Self' after upgrading to python 3.11
    @classmethod
    def from_flat_dict(cls, data: dict[str, Any], mapping: dict[str, str] | None = None) -> Any:
        """Construct a dataclass from a flat (not nested) dictionary.

        Args:
            data (dict[str, Any]): Dictionary to map from
            mapping (dict[str, str] | None, optional): Dictionary keys to class fields names map in
                the format: 'dict_key' -> 'dataclass_field_name'. If `None` then assume that field
                and keys have the the same names. Defaults to None.

        Raises:
            TypeError: Class is not dataclass
            KeyError: Some required class fields are missing in dictionary data. It may be due to
                invalid names mapping

        Returns:
            Self: An instance of the dataclass with data mapped
        """
        if not is_dataclass(cls):
            raise TypeError(f"Class '{cls}' must be a dataclass")

        mapping = mapping or {}
        class_fields_names = [field.name for field in fields(cls)]

        if mapping:
            # Map all data-dict fields to new class names and filter out non-class keys.
            new_data = {mapping.get(k, k): v for k, v in data.items()}
            new_data = {k: v for k, v in new_data.items() if k in class_fields_names}
        else:
            new_data = {k: v for k, v in data.items() if k in class_fields_names}

        try:
            return cls(**new_data)
        except TypeError as ex:
            missing_keys = str(ex).split(":")[-1].strip()
            raise KeyError(
                f"Required __init__ parameters: {missing_keys} not found in data dictionary",
            )


@dataclass
class PeriodicEvent:
    epoch: float
    """Time of first observed transit-like event."""
    duration: float
    period: float


ID: TypeAlias = int | str


TCE_NO_NAME_FLAG = "NO_NAME"


@dataclass
class TargetRelatedObject:
    """Object related to the target star, binary or multiple system."""

    target_id: ID


class TceLabel(Enum):
    PC = "PLANET_CANDIDATE"
    NTP = "NON_TRANSITING_PHENOMENA"
    AFP = "ASTROPHYSICAL_FALSE_POSITIVE"
    UNKNOWN = "UNKNOWN"
    FP = "FALSE_POSITIVE"
    """Generic non-PC label. Prefer AFP and NTP.

    This should only be used when there is no method to determine whether TCE is AFP or NTP.
    """


@dataclass
class TCE(FromDictMixin, TargetRelatedObject):
    """Threshold-Crossing Event.

    A sequence of transit-like features in the flux time series of a given target that resembles
    the signature of a transiting planet to a sufficient degree that the target is passed on for
    further analysis.
    """

    tce_id: ID
    name: str | None
    label: TceLabel

    @property
    @abc.abstractmethod
    def event(self) -> PeriodicEvent:
        ...

    @classmethod
    def from_flat_dict(cls, data: dict[str, Any], mapping: dict[str, str] | None = None) -> Any:
        tce: TCE = super().from_flat_dict(data, mapping)

        # Label is saved as `TceLabel.name` so it must be casted to the actual `TceLabel` enum
        tce.label = TceLabel[tce.label]  # type: ignore
        return tce


@dataclass(unsafe_hash=True)
class KeplerTCE(TCE):
    opt_ghost_core_aperture_corr: float
    opt_ghost_halo_aperture_corr: float
    bootstrap_false_alarm_proba: float
    rolling_band_fgt: float
    radius: float
    fitted_period: float
    transit_depth: float
    epoch: float
    duration: float
    period: float
    _normalize_duration: bool = True

    def __post_init__(self) -> None:  # pragma: no cover
        self.target_id = int(self.target_id)
        self.tce_id = int(self.tce_id)

        if self.name == TCE_NO_NAME_FLAG:  # This is for easier bool(tce)
            self.name = None

        if self._normalize_duration:
            self.duration = round(self.duration / 24, 4)  # For Kepler 'duration' is in hours

    @property
    def event(self) -> PeriodicEvent:  # pragma: no cover
        return PeriodicEvent(self.epoch, self.duration, self.period)


@dataclass
class StellarParameters(FromDictMixin, TargetRelatedObject):
    """Physical properties of the target star, binary or multiple system."""


@dataclass
class KeplerStellarParameters(StellarParameters):
    effective_temperature: float
    radius: float
    mass: float
    density: float
    surface_gravity: float
    """log(g)"""
    metallicity: float


Series: TypeAlias = npt.NDArray[np.float_]


class TimeSeries(TypedDict):
    TIME: Series


class KeplerTimeSeries(TimeSeries):
    MOM_CENTR1: Series
    MOM_CENTR2: Series
    PDCSAP_FLUX: Series


PeriodicTimeSeries: TypeAlias = dict[str, TimeSeries]

PeriodicData: TypeAlias = dict[str, Iterable[float]]
