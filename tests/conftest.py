import pandas as pd


def create_df(records: list[tuple]) -> pd.DataFrame:
    """Create `DataFrame` from records.

    Parameters
    ----------
    records : list[Any]
        A list of tuples contained test data. The first tuple must be a columns names
    Returns
    -------
    pd.DataFrame
        DataFrame object representing passed records
    """
    return pd.DataFrame.from_records(columns=records[0], data=records[1:])
