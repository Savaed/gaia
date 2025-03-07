from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeAlias
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from gaia.data.models import AnySeries, IntSeries, Series


@pytest.fixture
def http_response(request):
    """Return a mock of `aiohttp.ClientSession` HTTP method with status and response body."""
    response_body, status, method = request.param
    response_mock = MagicMock(**{"read": AsyncMock(return_value=response_body), "status": status})
    return MagicMock(**{f"{method}.return_value.__aenter__.return_value": response_mock})


def assert_dict_with_numpy_equal(a: Mapping[Any, Any], b: Mapping[Any, Any]) -> None:
    """Assert that two dictionaries with NumPy arrays as several of their values are equal.

    Equality means keys and values are the same, but not necessarily in the same order of keys.
    """
    sorted_a = dict(sorted(a.items()))
    sorted_b = dict(sorted(b.items()))
    is_keys_equal = sorted_a.keys() == sorted_b.keys()
    assert all(
        [np.array_equal(left, right) for left, right in zip(sorted_a.values(), sorted_b.values())]
        + [is_keys_equal],
    )


DictWithListValues: TypeAlias = Mapping[Any, list[Any]]


def assert_dict_with_list_equal_no_order(d1: DictWithListValues, d2: DictWithListValues) -> None:
    """Assert that two dictionaries with lists as their values are equal.

    Equality: values in lists are the same, but not necessarily in the same order.
    """
    tmp_d1 = {k: set(v) for k, v in d1.items()}
    tmp_d2 = {k: set(v) for k, v in d2.items()}
    assert tmp_d1 == tmp_d2


def create_df(data: tuple[Iterable[Any], ...]) -> pd.DataFrame:
    """Create table with specified columns and rows.

    Args:
        data (tuple[Iterable[Any], ...]): Data in form of columns and rows

    Returns:
        pd.DataFrame: Table with passed columns and rows
    """
    return pd.DataFrame(data=data[1:], columns=data[0])


@pytest.fixture
def save_to_file(tmp_path: Path) -> Callable[[str], Path]:
    def _save(data: str) -> Path:
        filepath = tmp_path / "test_file"
        filepath.write_text(data)
        return filepath

    return _save


IterableOfArrays: TypeAlias = Iterable[AnySeries | IntSeries | Series]


def assert_iterable_of_arrays_equal(
    arrays_1: IterableOfArrays,
    arrays_2: IterableOfArrays,
    equal_nan: bool = True,
) -> None:
    """Assert that all numpy arrays in two iterables `arrays_1` and `arrays_2` are equal as follows:
    `arrays_1[i] == arrays_2[i]`.

    Args:
        arrays_1 (Iterable[AnySeries]): First iterable of numpy arrays
        arrays_2 (Iterable[AnySeries]): Second iterable of numpy arrays
        equal_nan (bool, optional): Whether `np.nan` values are considered equal. Defaults to True.
    """
    is_equal = [
        np.array_equal(array_1, array_2, equal_nan=equal_nan)
        for array_1, array_2 in zip(arrays_1, arrays_2)
    ]
    assert all(is_equal)


def assert_iterable_of_arrays_almost_equal(
    arrays_1: IterableOfArrays,
    arrays_2: IterableOfArrays,
    equal_nan: bool = True,
    relative_tolerance: float = 1.0e-5,
    absolute_tolerance: float = 1.0e-8,
) -> None:
    """Assert that all numpy arrays in two iterables `arrays_1` and `arrays_2` are equal within a
    tolerance as follows: `absolute(arrays_1[i] - arrays_2[i]) <= (absolute_tolerance +
    relative_tolerance * absolute(arrays_2[i]))`.

    Args:
        arrays_1 (Iterable[AnySeries]): First iterable of numpy arrays
        arrays_2 (Iterable[AnySeries]): Second iterable of numpy arrays
        equal_nan (bool, optional): Whether `np.nan` values are considered equal. Defaults to True.
        relative_tolerance (float, optional): Relative tolerance. Defaults to 1.e-5
        absolute_tolerance (float, optional): Absolute tolerance. Defaults to 1.e-8
    """
    is_equal = [
        np.allclose(
            array_1,
            array_2,
            equal_nan=equal_nan,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        for array_1, array_2 in zip(arrays_1, arrays_2)
    ]
    assert all(is_equal)
