import shutil
from collections.abc import Iterator
from enum import Enum
from unittest.mock import AsyncMock, MagicMock

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
