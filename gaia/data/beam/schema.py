"""Schema of elements processed in the Apache Beam pipeline."""

from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from gaia.data.models import TCE, StellarParameters, TimeSeries


@dataclass
class KeplerFullData:
    """Represents complete Kepler data: time series, TCEs, stellar parameters."""

    kepid: int
    time_series: TimeSeries
    stellar_params: StellarParameters
    tces: list[TCE]


@dataclass
class TimeSeriesBase:
    """Represents time series for a specific target."""

    kepid: int
    time: list[np.ndarray]


@dataclass
class TimeSeriesFeature(TimeSeriesBase):
    """Represents a time series feature for a specific target."""

    series: list[np.ndarray]
    tces: list[TCE]


@dataclass
class CentroidXY(TimeSeriesBase):
    """Represents the X (row) and Y (column) centroid positions over time."""

    tces: list[TCE]
    centroid_x: list[np.ndarray]
    centroid_y: list[np.ndarray]


@dataclass
class NormTimeSeries(TimeSeriesBase):
    """Represents normalized time series feature."""

    series: np.ndarray
    tce: TCE


@dataclass
class Views:
    """Represents local and global time series views for the specified TCE."""

    koi_tce_id: str
    local_view: np.ndarray
    global_view: np.ndarray

@dataclass
class ErrorEvent:
    """Represents an error in pipeline processing."""

    transform: str
    error: Union[str, Exception]
    data: Any
    timestamp: Union[int, float, str]
