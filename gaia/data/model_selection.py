"""Function to split TCEs into train, tests and validation data sets."""

from typing import Optional

import numpy as np


def train_test_val_split(
    x: np.ndarray,
    *,
    test_size: float = 0.1,
    validation_size: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split KOIs into train, test and validation set.

    This guarantees that TCEs for the same target are placed only in one of the sets.

    Parameters
    ----------
    x : list[int]
        KOI IDs
    test_size : float, optional
        The size of the test set, by default 0.1
    validation_size : float, optional
        The size of the validation set, by default 0.1
    shuffle : bool, optional
        Whether to randomly shuffle `x` values, by default True
    seed : Optional[int], optional
        The state of pseudo-random values generator. Setting this parameter to some value,
        make all future splits deterministic, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Train, test and validation set
    """
    if shuffle:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(x)

    kepids, num_koi_tces = np.unique(x, return_counts=True, axis=0)
    num_all_tces = np.cumsum(num_koi_tces)
    test_split_index = np.argwhere(num_all_tces >= test_size * x.size).min()
    val_split_index = np.argwhere(num_all_tces >= (test_size + validation_size) * x.size).min()
    test_set, val_set, train_set = np.array_split(kepids, [test_split_index, val_split_index])
    return train_set, test_set, val_set
