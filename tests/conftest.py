import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Mapping, TypeAlias
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest


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


@pytest.fixture(scope="session")
def event_loop():
    """Setup pytest-asyncio loop to be the main one.

    If there is no running event loop, create one and set as the current one.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop


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
