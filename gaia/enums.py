"""Enums used throughout the project."""

import enum


class Cadence(enum.Enum):
    """Frequency of time series observations."""

    LONG = "llc"
    """Observations with an interval of ~29.4 minutes."""
    SHORT = "slc"
    """Observations with an interval of 1 minute."""


Add kepler data read functions

Add functions for read time series, events, TCEs scalar data
and stellar parameters along with nesessery data models and enums.
Add unit test for those functions.