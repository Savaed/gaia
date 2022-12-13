from enum import Enum
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from gaia.http import ApiError, download


TEST_URL = "https://www.test-api.com"


class HttpMethod(Enum):
    GET = "get"


def aiohttp_error(error_type, status):
    """Return an `aiohttp` error object based on the specified error type."""
    error_msg = "test error"
    match error_type:
        case aiohttp.ClientResponseError:
            request_info = aiohttp.RequestInfo(TEST_URL, HttpMethod.GET.value, None, None)
            return aiohttp.ClientResponseError(request_info, None, message=error_msg, status=status)
        case aiohttp.ClientOSError:  # pragma: no cover
            return aiohttp.ClientOSError(status, error_msg)


@pytest.fixture(
    params=[
        (aiohttp_error(aiohttp.ClientOSError, 999), HttpMethod.GET),
        (aiohttp_error(aiohttp.ClientResponseError, 400), HttpMethod.GET),
        (aiohttp_error(aiohttp.ClientResponseError, 401), HttpMethod.GET),
        (aiohttp_error(aiohttp.ClientResponseError, 403), HttpMethod.GET),
        (aiohttp_error(aiohttp.ClientResponseError, 404), HttpMethod.GET),
    ],
    ids=[
        "generic_os_error",
        "http_400_bad_request",
        "http_401_unauthorized",
        "http_403_forbidden",
        "http_404_not_found",
    ],
)
def error_response(request, mocker):
    """Mock `aiohttp.ClientSession` HTTP method to raise an error (OS or client)."""
    error, http_method = request.param
    http_method_mock = MagicMock(**{"return_value.__aenter__.side_effect": error})
    mocker.patch.object(aiohttp.ClientSession, http_method.value, http_method_mock)


@pytest.mark.asyncio
@pytest.mark.usefixtures("error_response")
async def test_download__http_or_os_error():
    """Test check whether ApiError is raised for OS error or malformed request."""
    with pytest.raises(ApiError):
        await download(TEST_URL, aiohttp.ClientSession())


@pytest.mark.asyncio
async def test_download__return_correct_response():
    """Test check whether the correct response is returned."""
    expected = b"c1,c2\n1,2"
    response_mock = MagicMock(**{"read": AsyncMock(return_value=expected), "status": 200})
    session_mock = MagicMock(**{"get.return_value.__aenter__.return_value": response_mock})
    result = await download(TEST_URL, session_mock)
    assert result == expected
