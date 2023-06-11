import numpy as np
import numpy.typing as npt

from gaia.data.models import TCE, Series


def compute_euclidean_distance(series: Series) -> Series:
    ndim = series.ndim
    if ndim != 2:
        raise ValueError(f"Expected 'series' to be 2D, but got {ndim}D")

    return np.linalg.norm(series, axis=0)  # type: ignore


def normalize_median(series: Series) -> Series:
    return series / np.nanmedian(series)


def phase_fold_time(time: Series, *, epoch: float, period: float) -> Series:
    """Create a phase-folded time vector.

    Args:
        time (Series): 1D array of time values
        epoch (float): First transit occurrence. This value is mapped to zero
        period (float): Period to fold over

    Raises:
        ValueError: `period` <= 0 OR `time` is not 1D

    Returns:
        Series: 1D numpy array folded around a `period` with time values within
            `[-period / 2, period / 2]`
    """
    if period <= 0:
        raise ValueError(f"Expected 'period' > 0, but got {period=}")
    if time.ndim != 1:
        raise ValueError(f"Expected 'time' to be 1D, but got {time.ndim}D")

    half_period = period / 2
    folded_time = np.mod(time + (half_period - epoch), period)
    folded_time -= half_period
    return folded_time


# TODO: Change to `transit_strategy` later
def compute_transits(
    tces: list[TCE],
    time: Series,
    default: str = "no detected",
) -> npt.NDArray[np.object_]:
    transits_mask = [default] * len(time)

    for tce in tces:
        tce_name = tce.name or str(tce.id)
        folded_time = phase_fold_time(time, epoch=tce.event.epoch, period=tce.event.period)
        transits_mask = [
            tce_name if np.abs(current_time) <= tce.event.duration else transit_marker
            for current_time, transit_marker in zip(folded_time, transits_mask)
        ]

    return np.array(transits_mask)
