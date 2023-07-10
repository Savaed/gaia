from typing import Callable, TypeAlias, TypeVar

import numpy as np

from gaia.data.models import (
    KeplerStellarParameters,
    KeplerTimeSeries,
    RawKeplerStellarParameter,
    RawKeplerTimeSeries,
)


T = TypeVar("T")

Mapper: TypeAlias = Callable[..., T]


class MapperError(Exception):
    """Raised when cannot map the source object to the target."""


def map_kepler_time_series(source: RawKeplerTimeSeries) -> KeplerTimeSeries:
    try:
        return KeplerTimeSeries(
            id=source["id"],
            time=np.array(source["time"]),
            mom_centr1=np.array(source["mom_centr1"]),
            mom_centr2=np.array(source["mom_centr2"]),
            pdcsap_flux=np.array(source["pdcsap_flux"]),
            period=source["period"],
        )
    except KeyError as ex:
        raise MapperError(f"Key '{ex}' not found in the source RawKeplerTimeSeries object")


def map_kepler_stallar_parameters(source: RawKeplerStellarParameter) -> KeplerStellarParameters:
    try:
        return KeplerStellarParameters(
            id=source["kepid"],
            effective_temperature=source["teff"],
            radius=source["radius"],
            mass=source["mass"],
            density=source["dens"],
            surface_gravity=source["logg"],
            metallicity=source["feh"],
        )
    except KeyError as ex:
        raise MapperError(f"Key '{ex}' not found in the source RawKeplerStellarParameter object")
