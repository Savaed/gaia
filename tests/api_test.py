from enum import Enum
from unittest.mock import MagicMock

import aiohttp
import pytest

from gaia.api import ApiError, download


TEST_URL = "https://www.test.com"


class HttpMethod(Enum):
    GET = "get"


def aiohttp_error(error_type, status):
    """Return an aiohttp error based on the specified error type"""
    error_msg = "test error"
    match error_type:
        case aiohttp.ClientResponseError:
            request_info = aiohttp.RequestInfo(TEST_URL, HttpMethod.GET.value, None, None)
            return aiohttp.ClientResponseError(request_info, None, message=error_msg, status=status)
        case aiohttp.ClientOSError:  # pragma: no cover
            return aiohttp.ClientOSError(-1, error_msg)


@pytest.fixture
def http_response(request):
    """
    Mock http method on `aiohttp.ClientSession()` so it
    will return specified response or raise an error.
    """
    response, error, method, status = request.param
    mock = aiohttp.ClientSession
    method_mock = MagicMock()

    if error:
        method_mock.return_value.__aenter__.return_value.read.side_effect = [error]

    method_mock.return_value.__aenter__.return_value.status = status or 200
    method_mock.return_value.__aenter__.return_value.read.return_value = response
    setattr(mock, method.value, method_mock)
    return mock


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "http_response",
    [
        (None, aiohttp.ClientOSError(1, "error"), HttpMethod.GET, 200),
        (None, aiohttp_error(aiohttp.ClientResponseError, 400), HttpMethod.GET, 400),
        (None, aiohttp_error(aiohttp.ClientResponseError, 401), HttpMethod.GET, 401),
        (None, aiohttp_error(aiohttp.ClientResponseError, 403), HttpMethod.GET, 403),
        (None, aiohttp_error(aiohttp.ClientResponseError, 404), HttpMethod.GET, 404),
    ],
    indirect=True,
    ids=[
        "generic_os_error",
        "http_400_bad_request",
        "http_401_unauthorized",
        "http_403_forbidden",
        "http_404_not_found",
    ],
)
async def test_download__invalid_request(http_response):
    """Test check whether ApiError is raised when the request is malformed."""
    with pytest.raises(ApiError):
        await download(TEST_URL, aiohttp.ClientSession())


@pytest.mark.asyncio
@pytest.mark.parametrize("http_response", [(b"test", None, HttpMethod.GET, 200)], indirect=True)
async def test_download__return_correct_response(http_response):
    """Test check whether the correct response is returned."""
    result = await download(TEST_URL, aiohttp.ClientSession())
    assert result == b"test"
