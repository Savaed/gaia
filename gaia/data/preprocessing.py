import numpy as np
import numpy.typing as npt

from gaia.data.models import TCE, Series
from gaia.log import logger


def compute_euclidean_distance(series: Series) -> Series:
    ndim = series.ndim
    if ndim != 2:
        raise ValueError(f"Expected 'series' to be 2D, but got {ndim}D")

    return np.linalg.norm(series, axis=0)  # type: ignore


def normalize_median(series: Series) -> Series:
    return series / np.nanmedian(series)


def phase_fold_time(period: float, epoch: float, time: Series) -> Series:
    half_period = period / 2
    return np.mod(time + (half_period - epoch), period) - half_period


# TODO: Change to `transit_strategy` later
def compute_transits(
    tces: list[TCE],
    time: Series,
    default: str = "no detected",
) -> npt.NDArray[np.object_]:
    transits_mask = [default] * len(time)

    logger.error(f"{time.min(), time.max()}")

    for tce in tces:
        tce_name = tce.name or str(tce.id)
        folded_time = phase_fold_time(tce.event.period, tce.event.epoch, time)
        transits_mask = [
            tce_name if np.abs(current_time) <= tce.event.duration else transit_marker
            for current_time, transit_marker in zip(folded_time, transits_mask)
        ]

    return np.array(transits_mask)
