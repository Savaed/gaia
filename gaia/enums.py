"""Enums used throughout the project."""

from enum import Enum, auto


class Cadence(Enum):
    """Frequency of time series observations."""

    LONG = "llc"
    """Observations with an interval of ~29.4 minutes."""
    SHORT = "slc"
    """Observations with an interval of 1 minute."""


class TceLabel(Enum):
    PLANET_CANDIDATE = auto()
    FALSE_POSITIVE = auto()


class TceSpecificLabel(Enum):
    PLANET_CANDIDATE = auto()
    ASTROPHYSICAL_FALSE_POSITIVE = auto()
    NON_TRANSIT_PHENOMENON = auto()
