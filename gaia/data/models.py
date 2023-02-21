from abc import ABC
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt


class FromDictMixin(ABC):
    """Abstract class that provides a method to map flat dictionary to the dataclass."""

    @classmethod
    def from_dict(cls, data: dict[str, Any], mapping: dict[str, str] | None = None):  # type: ignore
        """Construct a dataclass from a flat (not nested) dictionary.

        Args:
            data (dict[str, Any]): Dictionary to map from
            mapping (dict[str, str] | None, optional): Dictionary keys to class fields names map in
                the format: 'dict_key' -> 'class_field_name'. If `None` then assume that field and
                keys have the the same names. Defaults to None.

        Raises:
            TypeError: Class is not dataclass
            KeyError: Some required class fields are missing in dictionary data. It may be due to
                invalid names mapping

        Returns:
            Self: An instance of the dataclass with data mapped
        """
        if not is_dataclass(cls):
            raise TypeError(f"{cls=} must be a dataclass")

        mapping = mapping or {}
        class_fields_names = [f.name for f in fields(cls)]

        if mapping:
            # Map all data-dict fields to new class names
            new_data = {mapping.get(k, k): v for k, v in data.items()}
            new_data = {k: v for k, v in new_data.items() if k in class_fields_names}
        else:
            new_data = {k: v for k, v in data.items() if k in class_fields_names}

        try:
            return cls(**new_data)
        except TypeError as ex:
            missing_keys = str(ex).split(":")[-1].strip()
            raise KeyError(f"The following keys are not present in passed data: {missing_keys}")


@dataclass
class PeriodicEvent:
    epoch: float
    """Time of first observed transit-like event."""
    duration: float
    period: float


@dataclass
class KeplerTCE:
    """Threshold-Crossing Event (TCE) for Kepler observations.

    A sequence of transit-like features in the flux time series of a given target that resembles
    the signature of a transiting planet to a sufficient degree that the target is passed on for
    further analysis.
    """

    kepid: int
    label: str
    tce_num: int
    """The number of observed TCE in the flux time series of a given target."""
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

    @property
    def event(self) -> PeriodicEvent:
        """Transit-like periodic event."""
        return PeriodicEvent(self.epoch, self.duration, self.period)


@dataclass
class KeplerStellarParameters:
    kepid: int
    effective_temperature: float
    radius: float
    mass: float
    density: float
    surface_gravity: float
    """log(g)"""
    metallicity: float


Segments = list[npt.NDArray[np.float_]] | npt.NDArray[np.float_]


class KeplerTimeSeries(TypedDict):
    TIME: Segments
    MOM_CENTR1: Segments
    MOM_CENTR2: Segments
    PDCSAP_FLUX: Segments


KeplerQuarterlyTimeSeries: TypeAlias = dict[str, KeplerTimeSeries]
