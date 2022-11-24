from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def response(request):
    """Mock an `aiohttp.ClientSession` HTTP method with status and returned response body."""
    response_body, status, method = request.param
    response_mock = MagicMock(**{"read": AsyncMock(return_value=response_body), "status": status})
    return MagicMock(**{f"{method}.return_value.__aenter__.return_value": response_mock})
