import shutil
from collections.abc import Iterator
from enum import Enum
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from pyparsing import Any

from gaia.log import configure_logging


TEST_LOG_DIR = "./test_logs/"


@pytest.fixture
def response(request: Any) -> MagicMock:
    """Mock an `aiohttp.ClientSession` HTTP method with status and returned response body."""
    response_body, status, method = request.param
    response_mock = MagicMock(**{"read": AsyncMock(return_value=response_body), "status": status})
    return MagicMock(**{f"{method}.return_value.__aenter__.return_value": response_mock})


@pytest.fixture(scope="session")
def setup_logging() -> Iterator[None]:
    try:
        configure_logging(TEST_LOG_DIR)
        yield
    finally:
        shutil.rmtree(TEST_LOG_DIR)


def enum_short_id(value: Enum) -> str:
    return value.name


def prefix_id(value: Any, prefix: str) -> str:
    return f"{prefix}-{str(value)}"


def assert_dict_with_numpy_equal(a: dict[Any, Any], b: dict[Any, Any]) -> None:
    """
    Assert that two dictionaries with NumPy arrays as their values are equal
    (have the same keys and value but not necessarily in the same order of keys).
    """
    sorted_a = dict(sorted(a.items()))
    sorted_b = dict(sorted(b.items()))
    assert sorted_a.keys() == sorted_b.keys()
    assert all([np.array_equal(r, e) for r, e in zip(sorted_a.values(), sorted_b.values())])


def assert_dict_with_list_equal_no_order(
    d1: dict[Any, list[Any]],
    d2: dict[Any, list[Any]],
) -> None:
    """
    Assert that two dictionaries with values ​​as lists have the same values ​​regardless of
    their order.
    """
    tmp_d1 = {k: set(v) for k, v in d1.items()}
    tmp_d2 = {k: set(v) for k, v in d2.items()}
    assert tmp_d1 == tmp_d2
