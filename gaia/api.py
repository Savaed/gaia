"""
REST API for NASA tabular data (available via NASA official API) and light curves for star targets.

NASA API: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
Kepler light curves: https://archive.stsci.edu/missions-and-data/kepler/kepler-bulk-downloads
"""

import re
from dataclasses import dataclass

import aiohttp


@dataclass
class ApiError(Exception):
    """Raised when the HTTP request to the NASA API is incorrect."""

    message: str
    status: int
    url: str | None

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.status}: {self.url} {self.message}"


async def download(url: str, session: aiohttp.ClientSession) -> bytes:
    """Download data from `URL` via REST API.

    Args:
        url (str): The source URL
        session (aiohttp.ClientSession): HTTP session

    Raises:
        ApiError: Underlying OS error or malformed request (e.g. HTTP 404 Not Found)

    Returns:
        bytes: Response data
    """
    try:
        async with session.get(url, raise_for_status=True) as resp:
            response: bytes = await resp.read()
    except aiohttp.ClientOSError as ex:
        raise ApiError(message=ex.strerror, status=ex.errno, url=url)
    except aiohttp.ClientResponseError as ex:
        raise ApiError(message=ex.message, status=ex.status, url=url)

    return response


class NasaApi:
    """NASA REST API. This allows to download tabular data from the official NASA archive.

    See: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
    """

    def __init__(self, base_url: str, session: aiohttp.ClientSession) -> None:
        self._base_url = base_url
        self._session = session

    async def download(self, table: str, columns: set[str], data_format: str = "csv") -> bytes:
        """Download a NASA table.

        Note that not all tables from the NASA archive are available via API.

        Args:
            table (str): Table name. See NASA API docs for supported tables.
            columns (set[str]): Columns to select from the table. If not specified, all columns will
                be downloaded
            data_format (str, optional): Format of returned data. See NASA API docs for supported
                tables. Some formats are supported only for several tables. Defaults to "csv".

        Raises:
            ApiError: Underlying OS error or malformed request (e.g. HTTP 404 Not Found)

        Returns:
            bytes: Table content
        """
        data = await download(self._create_url(table, columns, data_format), self._session)
        self._raise_for_error(data)
        return data

    def _raise_for_error(self, response: bytes) -> None:
        if b"ERROR" in response:
            error_msg = self._to_error_msg(response)
            raise ApiError(message=error_msg, status=400, url=None)

    def _to_error_msg(self, error: bytes) -> str:
        # Error from NASA API is of format:
        # ERROR<br>
        # Error Type: {error_type}<br>
        # Message:    {error_message}.
        error_str = error.decode("utf-8")
        error_msg_list = re.findall(r"(?<=Message:).*", error_str)
        return error_msg_list[0].strip(" .") if error_msg_list else ""  # type: ignore

    def _create_url(
        self,
        table: str,
        columns: set[str],
        data_format: str = "csv",
        query: str | None = None,
    ) -> str:
        columns_str = ",".join(columns)
        url = f"{self._base_url}?table={table}{f'&select={columns_str}' or ''}&format={data_format}"
        return f"{url}&where={query:r}" if query else url
