"""DTO for Kepler time series, stellar parameters, events and TCE scalar features."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PeriodEvent:
    """An event that begins in an epoch has a duration and a period."""

    epoch: float
    """Time of first observed transit-like event."""
    duration: float
    period: float


@dataclass(frozen=True)
class TCE:
    """Threshold-Crossing Event.

    A sequence of transit-like features in the flux time series of a given target that
    resembles the signature of a transiting planet to a sufficient degree that the target is
    passed on for further analysis.
    """

    kepid: int
    tce_num: int
    """The number of observed TCE in the flux time series of a given target."""
    event: PeriodEvent
    opt_ghost_core_aperture_corr: float
    opt_ghost_halo_aperture_corr: float
    bootstrap_false_alarm_proba: float
    rolling_band_fgt: float
    radius: float
    fitted_period: float

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TCE":
        """Create a TCE object from a dict (e.g. from pandas Series using `to_dict` method).

        Parameters
        ----------
        data : dict[str, Any]
            Dict with keys as desribed in NASA API docs:
            https://exoplanetarchive.ipac.caltech.edu/docs/API_tce_columns.html

        Returns
        -------
        TCE
            TCE object based on dict data
        """
        event = PeriodEvent(
            epoch=data["tce_time0bk"], duration=data["tce_duration"], period=data["tce_period"]
        )
        return TCE(
            kepid=data["kepid"],
            tce_num=data["tce_plnt_num"],
            event=event,
            opt_ghost_core_aperture_corr=data["tce_cap_stat"],
            opt_ghost_halo_aperture_corr=data["tce_hap_stat"],
            bootstrap_false_alarm_proba=data["boot_fap"],
            rolling_band_fgt=data["tce_rb_tcount0"],
            radius=data["tce_prad"],
            fitted_period=data["tcet_period"],
        )


@dataclass(frozen=True)
class StellarParameters:
    """Parameters for target observed by Kepler to find transiting planets."""

    kepid: int
    effective_temperature: float
    radius: float
    mass: float
    density: float
    surface_gravity: float  # log(g)
    metallicity: float

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "StellarParameters":
        """Create a StellarParameters object from a dict
        (e.g. from pandas Series using `to_dict` method).

        Parameters
        ----------
        data : dict[str, Any]
            Dict with keys as desribed in NASA API docs:
            https://exoplanetarchive.ipac.caltech.edu/docs/API_keplerstellar_columns.html

        Returns
        -------
        StellarParameters
            StellarParameters object based on dict data
        """
        return StellarParameters(
            kepid=data["kepid"],
            effective_temperature=data["teff"],
            radius=data["radius"],
            mass=data["mass"],
            density=data["dens"],
            surface_gravity=data["logg"],
            metallicity=data["feh"],
        )


@dataclass
class TimeSeries:
    """Features of the Kepler time series.

    This may or may not include all of the fields available in the FITS file for the target.
    """

    def __init__(self, kepid: int, data: dict[str, Any]) -> None:
        self.kepid = kepid

        for attr_name, data_array in data.items():
            setattr(TimeSeries, attr_name, data_array)


@dataclass(frozen=True)
class KeplerData:
    """Complete data of Kepler target.

    This includes a target's time series, stellar features, and a list of TCE.
    """

    kepid: int = field(init=False)
    time_series: dict[str, np.ndarray]
    stellar_params: StellarParameters
    tces: list[TCE]

    def __post_init__(self) -> None:
        self.kepid = self.stellar_params.kepid  # type: ignore
