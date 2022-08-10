""""REST API for MAST and NASA data archives."""

import re
from dataclasses import dataclass
from typing import AnyStr, Generator, Optional, Protocol

import aiohttp

from gaia.constants import get_quarter_prefixes
from gaia.enums import Cadence


def create_nasa_url(
    base_url: str,
    table: str,
    data_fmt: str = "csv",
    select: Optional[list[str]] = None,
    query: Optional[str] = None,
) -> str:
    """
    Create NASA HTTP URL for scalar data.

    Parameters
    ----------
    base_url : str
        Base URL shared by each API request
    table : str
        Table name to download
    data_fmt : str, optional
        Format of downloaded tabel, by default "csv"
    select : Optional[str], optional
        Table columns to include, by default None
    query : Optional[str], optional
        SQL-like query to filter the data, by default None

    Returns
    -------
    str
        URL based on a table name, data format, selected columns and filter statement
    """
    if not base_url or not table or not data_fmt:
        raise ValueError("Parameters 'base_url', 'table', 'data_fmt' cannot be empty")

    url = f"{base_url}?table={table}&format={data_fmt}"

    if select:
        url += f"&select={','.join(select)}"

    if query:
        url += f"&where={query}"

    return url


def get_mast_urls(base_url: str, kepid: int, cadence: Cadence) -> Generator[None, None, str]:
    """
    Create MAST HTTP URLs for time series.

    Parameters
    ----------
    base_url : str
        Base URL shared by each API request
    kepid : int
        Id of target Kepler Object of Interest (KOI)
    cadence : Cadence
        Observation frequency

    Yields
    ------
    Generator[None, None, str]
        URL for specific KOI  that can be use to retrive time series from the MAST archive
    """
    if not base_url:
        raise ValueError("'base_url' cannot be empty")

    if not 0 < kepid < 1_000_000_000:
        raise ValueError(f"'kepid' must be in range 1 to 999 999 999 inclusive, but got {kepid=}")

    fmt_kepid = f"{kepid:09}"
    url = (
        f"{base_url}?uri=mast:Kepler/url/missions/kepler/lightcurves/"
        f"{fmt_kepid[:4]}/{fmt_kepid}//kplr{fmt_kepid}"
    )
    yield from (f"{url}-{prefix}_{cadence.value}.fits" for prefix in get_quarter_prefixes(cadence))


@dataclass(frozen=True)
class InvalidRequestOrResponse(Exception):
    """Raised when the request or response is invalid."""

    error: str
    msg: str


class ApiError(Exception):
    """
    Raised when there is a serious API communication problem
    other than just an invalid request or response.
    """


class BaseApi(Protocol):
    async def fetch(self, url: str) -> AnyStr:
        ...


class MastApi:
    async def fetch(self, url: str) -> AnyStr:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, ssl=False, raise_for_status=True) as resp:
                    response_body = await resp.read()
                    return response_body
        except (aiohttp.ClientResponseError, aiohttp.ClientPayloadError, aiohttp.InvalidURL) as ex:
            # Mainly HTTP 404 NotFound - Kepler data is unavailable for a few quarters.
            raise InvalidRequestOrResponse(f"HTTP {ex.status}", ex.message) from ex
        except aiohttp.ServerConnectionError as ex:
            raise ApiError(f"Unable to connect to {url=}. {ex}") from ex


class NasaApi:
    async def fetch(self, url: str) -> tuple[str, AnyStr]:
        base_url, query_params = self._split_url(url)
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(base_url, params=query_params) as resp:
                    table = await resp.read()
        except Exception as ex:
            raise ApiError(f"Cannot fetch table from {url=}. {ex}") from ex
        self._validate_response(table)
        return query_params.get("table"), table.decode("utf-8")

    def _split_url(self, url: str) -> tuple[str, dict[str, str]]:
        base_url, query = url.split("?")
        params = {(p := param.split("="))[0]: p[1] for param in query.split("&")}
        return base_url, params

    def _validate_response(self, response: AnyStr) -> None:
        if not response:
            raise InvalidRequestOrResponse(error=None, msg="There is no response")

        if isinstance(response, bytes):
            response = response.decode("utf-8")

        # Format of error response is 'ERROR<br>...'.
        if response.startswith("ERROR<br>"):
            error = (
                re.search(r"(?<=Error Type: UserError - )(.+)", response)
                .group(0)
                .replace("<br>", "")
                .strip()
            )
            error_msg = re.search(r"(?<=Message:)(.+)", response).group(0).strip()
            raise InvalidRequestOrResponse(error=error, msg=error_msg)
