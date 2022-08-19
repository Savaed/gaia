""""REST API for MAST and NASA data archives."""

import re
from dataclasses import dataclass
from typing import AnyStr, Generator, Optional

import aiohttp

from gaia.constants import get_quarter_prefixes
from gaia.enums import Cadence
from gaia.io.kepler import check_kepid


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


def get_mast_urls(base_url: str, kepid: int, cadence: Cadence) -> Generator[str, None, None]:
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

    check_kepid(kepid)

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


class MastApi:
    """REST API for fetching Kepler time series from Mikulski Archive for Space Telescopes."""

    async def fetch(self, url: str) -> bytes:
        """Fetch Kepler time series from Mikulski Archive for Space Telescopes (MAST).

        See: https://archive.stsci.edu/missions-and-data/kepler/kepler-bulk-downloads

        Parameters
        ----------
        url : str
            HTTP url to the source time series file

        Returns
        -------
        any
            Fits file with time series as a binary file

        Raises
        ------
        InvalidRequestOrResponse
            In most cases, it indicates that is no file for a specified observation quarter
        ApiError
            Any communication error like server disconnection
        """
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, ssl=False, raise_for_status=True) as resp:
                    response_body = await resp.read()
                    return response_body
        except (aiohttp.ClientResponseError, aiohttp.ClientPayloadError, aiohttp.InvalidURL) as ex:
            # Mainly HTTP 404 NotFound - Kepler time series is unavailable for a few quarters.
            raise InvalidRequestOrResponse(f"HTTP {ex.status}", ex.message) from ex
        except aiohttp.ServerConnectionError as ex:
            raise ApiError(f"Unable to connect to {url=}. {ex}") from ex


class NasaApi:
    """REST API for fetching data tables from the official NASA archive."""

    async def fetch(self, url: str) -> tuple[str, str]:
        """Fetch data tables from NASA archives using the official REST API.

        See: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html

        Parameters
        ----------
        url : str
            HTTP url to the source table

        Returns
        -------
        tuple[str, AnyStr]
            Table name, table decoded as string

        Raises
        ------
        ApiError
            Any communication error like server disconnection
        """
        base_url, query_params = self._split_url(url)
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(base_url, params=query_params) as resp:
                    table = await resp.read()
        except Exception as ex:
            raise ApiError(f"Cannot fetch table from {url=}. {ex}") from ex

        # Validate the response body because NASA API send HTTP 200 response even for errors
        self._validate_response(table)
        return query_params.get("table"), table.decode("utf-8")

    def _split_url(self, url: str) -> tuple[str, dict[str, str]]:
        """Split url into base url and query parameters.

        Example
        -------
            >>> self._split_url("http://www.url.com?x=1&y=2")
            >>> ("http://www.url.com",  {"x": 1, "y":2})
        """
        base_url, query = url.split("?")
        # Split query params into format of {"name": value}
        # e.g. http://www.url.com?x=1&y=2 >> {"x": 1, "y":2}
        params = {(param_part := param.split("="))[0]: param_part[1] for param in query.split("&")}
        return base_url, params

    def _validate_response(self, response: AnyStr) -> None:
        """Validate NASA API response body."""
        if not response:
            raise InvalidRequestOrResponse(error="NO_RESPONSE", msg="There is no response")

        if isinstance(response, bytes):
            response = response.decode("utf-8")

        # Format of error response is 'ERROR<br>ErrorType:...<br>Message:...'
        if response.startswith("ERROR<br>"):
            error = (
                re.search(r"(?<=Error Type: UserError - )(.+)", response)
                .group(0)
                .replace("<br>", "")
                .strip()
            )
            error_msg = re.search(r"(?<=Message:)(.+)", response).group(0).strip()
            raise InvalidRequestOrResponse(error=error, msg=error_msg)
