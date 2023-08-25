from typing import Sequence

import numpy as np

from gaia.data.models import Id


def train_test_val_split(
    x: Sequence[int | float | str],
    *,
    test_size: float = 0.1,
    validation_size: float = 0.1,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[set[Id], set[Id], set[Id]]:
    """Split targets into train, test and validation set.

    This guarantees that TCEs for the same target are placed only in one of the sets.

    Args:
        x (Iterable[TId]): Target IDs
        test_size (float, optional): The size of the test set. Defaults to 0.1.
        validation_size (float, optional): The size of the validation set. Defaults to 0.1.
        shuffle (bool, optional):  Whether to randomly shuffle `x` values. Defaults to True.
        seed (int | None, optional): The state of pseudo-random values generator. Setting this
            parameter to some value, make all future splits deterministic. Defaults to None.

    Returns:
        tuple[set[TId], set[TId], set[TId]]: Train, test and validation set
    """

    if test_size + validation_size > 1:
        raise ValueError("'test_size' + 'validation_size' cannot be < 1")
    if not (0 <= test_size <= 1.0):
        raise ValueError(f"Expected 'test_size' to be in range [0, 1], but got f{test_size=}")
    if not (0 <= validation_size <= 1.0):
        raise ValueError(
            f"Expected 'validation_size' to be in range [0, 1], but got f{validation_size=}",
        )
    if not x:
        raise ValueError("'x' cannot be an empty iterable")

    if shuffle:
        np.random.default_rng(seed=seed).shuffle(list(x))  # Make mypy happy

    target_ids, tces = np.unique(x, return_counts=True)
    num_all_tces = np.cumsum(tces)
    x_len = len(x)
    test_split_index = np.argwhere(num_all_tces >= test_size * x_len).min()
    val_split_index = np.argwhere(num_all_tces >= (test_size + validation_size) * x_len).min()
    test_set, val_set, train_set = np.array_split(target_ids, [test_split_index, val_split_index])
    return set(train_set), set(test_set), set(val_set)
