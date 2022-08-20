"""Schema of elements processed in the Apache Beam pipeline."""

from dataclasses import dataclass

import numpy as np

from gaia.data.models import TCE, StellarParameters, TimeSeries


@dataclass
class KeplerDataElement:
    """Represents complete Kepler data: time series, TCEs, stellar parameters."""

    kepid: int
    time_series: TimeSeries
    stellar_params: StellarParameters
    tces: list[TCE]


@dataclass
class TimeSeriesElement:
    """Represents time series for a specific target."""

    kepid: int
    time: list[np.ndarray]


@dataclass
class TimeSeriesFeatureElement(TimeSeriesElement):
    """Represents a time series feature for a specific target."""

    series: list[np.ndarray]
    tces: list[TCE]


@dataclass
class CentroidXYElement(TimeSeriesElement):
    """Represents the X (row) and Y (column) centroid positions over time."""

    tces: list[TCE]
    centroid_x: list[np.ndarray]
    centroid_y: list[np.ndarray]


@dataclass
class NormTimeSeriesElement(TimeSeriesElement):
    """Represents normalized time series feature."""

    time_series: np.ndarray
    tce: TCE


@dataclass
class ViewsElement:
    """Represents local and global time series views for the specified TCE."""

    koi_tce_id: str
    local_view: np.ndarray
    global_view: np.ndarray
