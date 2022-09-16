"""Shared pytest fixtures and utils."""

from typing import Any

import pandas as pd


def create_df(records: list[tuple[Any, ...]]) -> pd.DataFrame:
    """Create pandas `DataFrame` from records.

    Parameters
    ----------
    records : list[tuple[Any, ...]]
        A list of tuples contained test data. The first tuple must be a columns names os type `str`

    Returns
    -------
    pd.DataFrame
        DataFrame object representing passed records
    """
    return pd.DataFrame.from_records(columns=records[0], data=records[1:])
