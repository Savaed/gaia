from collections.abc import Callable, Iterable
from dataclasses import asdict
from typing import Any, TypeAlias, TypeVar

import numpy as np

from gaia.data.models import (
    KeplerStellarParameters,
    KeplerTCE,
    KeplerTimeSeries,
    PeriodicEvent,
    RawKeplerStellarParameter,
    RawKeplerTce,
    RawKeplerTimeSeries,
    TceLabel,
    flatten_dict,
)


T = TypeVar("T")

Mapper: TypeAlias = Callable[..., T]


class MapperError(Exception):
    """Raised when cannot map the source object to the target."""


def map_kepler_time_series(source: RawKeplerTimeSeries) -> KeplerTimeSeries:
    try:
        return KeplerTimeSeries(
            id=source["KEPLERID"],
            time=np.array(source["TIME"]),
            mom_centr1=np.array(source["MOM_CENTR1"]),
            mom_centr2=np.array(source["MOM_CENTR2"]),
            pdcsap_flux=np.array(source["PDCSAP_FLUX"]),
            period=source["QUARTER"],
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


def map_tce_label(label: str) -> TceLabel:
    """Map text TCE label to `TceLabel` enumeration.

    Args:
        label (str): Text TCE label

    Returns:
        TceLabel: Enum if its `value` or `name` matches `label`, `TceLabel.UNKNOWN` otherwise.
    """
    try:
        return TceLabel[label]
    except KeyError:
        try:
            return TceLabel(label)
        except ValueError:
            return TceLabel.UNKNOWN


def map_kepler_tce(source: RawKeplerTce) -> KeplerTCE:
    try:
        return KeplerTCE(
            id=source["tce_plnt_num"],
            target_id=source["kepid"],
            name=source["kepler_name"],
            label=map_tce_label(source["label"]),
            event=PeriodicEvent(
                epoch=source["tce_time0bk"],
                duration=source["tce_duration"] / 24,  # Convert hours to days
                period=source["tce_period"],
                secondary_phase=source["tce_maxmesd"],
            ),
            opt_ghost_core_aperture_corr=source["tce_cap_stat"],
            opt_ghost_halo_aperture_corr=source["tce_hap_stat"],
            bootstrap_false_alarm_proba=source["boot_fap"],
            fitted_period=source["tcet_period"],
            radius=source["tce_prad"],
            rolling_band_fgt=source["tce_rb_tcount0"],
            transit_depth=source["tce_depth"],
            secondary_transit_depth=source["wst_depth"],
        )
    except KeyError as ex:
        raise MapperError(f"Key '{ex}' not found in the source RawKeplerTce object")


def map_dataclass_to_flat_dict(source: Any, exclude: Iterable[str] | None = None) -> dict[str, Any]:
    exclude = exclude or []
    try:
        dct = asdict(source)
    except TypeError:
        raise MapperError("Expected 'source' to be a dataclass instance")
    flat_dict = flatten_dict(dct)
    clean_dict = {k: v for k, v in flat_dict.items() if k not in exclude}
    return clean_dict
