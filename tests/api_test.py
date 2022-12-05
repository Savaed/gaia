from enum import Enum
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from gaia.api import ApiError, NasaApi, create_mast_urls, download
from gaia.enums import Cadence


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
def error_response(request):
    error, http_method = request.param
    return MagicMock(**{f"{http_method.value}.return_value.__aenter__.side_effect": error})


@pytest.mark.asyncio
async def test_download__http_or_os_error(error_response):
    """Test check whether ApiError is raised for OS error or malformed request."""
    with pytest.raises(ApiError):
        await download(TEST_URL, error_response)


@pytest.mark.asyncio
async def test_download__return_correct_response():
    """Test check whether the correct response is returned."""
    expected = b"c1,c2\n1,2"
    response_mock = MagicMock(**{"read": AsyncMock(return_value=expected), "status": 200})
    session_mock = MagicMock(**{"get.return_value.__aenter__.return_value": response_mock})

    result = await download(TEST_URL, session_mock)

    assert result == expected


class TestNasaApi:
    """Unit tests for `gaia.api.NasaApi` class."""

    @pytest.mark.asyncio
    async def test_download__http_or_os_error(self, mocker):
        """Test check whether ApiError is raised for any underlying error (HTTP or OS-related)."""
        error = ApiError(message="test", status=400, url=TEST_URL)
        mocker.patch("gaia.api.download", side_effect=error)

        async with aiohttp.ClientSession() as session:
            with pytest.raises(ApiError):
                await NasaApi(TEST_URL, session).download("table1", {"col1", "col2"})

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response_body,expected",
        [
            (
                b"ERROR<br>\nError Type: UserError - 'table' parameter<br>\nMessage:    'x' is not a valid table.",  # noqa
                "'x' is not a valid table",
            ),
            (
                b"ERROR<br>\nError Type: SystemError<br>\nMessage:    Error 907: HY000 :[Oracle][ODBC][Ora]ORA-00907: missing right parenthesis . FAIL",  # noqa
                ".*ORA-00907.*",
            ),
        ],
        ids=["table_not_found", "invalid_columns_query"],
    )
    async def test_download__error_response_http_200(self, response_body, expected, mocker):
        """Test check whether ApiError is raised for HTTP 200 OK error response."""
        mocker.patch("gaia.api.download", return_value=response_body)

        with pytest.raises(ApiError, match=expected):
            async with aiohttp.ClientSession() as session:
                await NasaApi(TEST_URL, session).download("table1", {"c1", "c2"})

    @pytest.mark.asyncio
    async def test_download__return_table(self, mocker):
        """Test check whether the correct table content is returned."""
        table_content = b"col1,col2\n1,2"
        table = "table"
        mocker.patch("gaia.api.download", return_value=table_content)

        async with aiohttp.ClientSession() as session:
            table_name, result = await NasaApi(TEST_URL, session).download(table, {"col1", "col2"})

        assert result == table_content
        assert table_name == table


@pytest.mark.parametrize("cadence", [Cadence.LONG, Cadence.SHORT])
@pytest.mark.parametrize("kepid", [-1, 0, 1_000_000_000])
def test_create_mast_urls__invalid_kepid(kepid, cadence):
    with pytest.raises(ValueError):
        next(create_mast_urls(kepid, cadence, base_url="sdfsf"))


@pytest.mark.parametrize("cadence", [Cadence.LONG, Cadence.SHORT])
def test_create_mast_urls__empty_base_url(cadence):
    with pytest.raises(ValueError, match="'base_url' cannot be empty"):
        next(create_mast_urls(123, cadence, base_url=""))


# TODO:Jakies takie to nie do konca mi sie podoba, ale działa


def x():
    return [
        (1, Cadence.LONG, "base", ("1",), ["base/0000/000000001//kplr000000001-1_llc.fits"]),
        (
            1,
            Cadence.LONG,
            "base",
            ("1", "2"),
            [
                "base/0000/000000001//kplr000000001-1_llc.fits",
                "base/0000/000000001//kplr000000001-2_llc.fits",
            ],
        ),
        (1, Cadence.SHORT, "base", ("1",), ["base/0000/000000001//kplr000000001-1_slc.fits"]),
        (
            1,
            Cadence.SHORT,
            "base",
            ("1", "2"),
            [
                "base/0000/000000001//kplr000000001-1_slc.fits",
                "base/0000/000000001//kplr000000001-2_slc.fits",
            ],
        ),
    ]


@pytest.mark.parametrize("kepid,cadence,base_url,prefixes,expected", x())
def test_create_mast_urls__return_correct_urls(
    kepid,
    cadence,
    base_url,
    prefixes,
    expected,
    mocker,
):
    mocker.patch("gaia.api.get_quarter_prefixes", return_value=prefixes)
    result = list(create_mast_urls(kepid, cadence, base_url))
    assert result == expected
