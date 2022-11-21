"""
REST API for NASA tabular data (available via NASA official API) and light curves for star targets.

NASA API: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
Kepler light curves: https://archive.stsci.edu/missions-and-data/kepler/kepler-bulk-downloads
"""

from dataclasses import dataclass

import aiohttp


@dataclass
class ApiError(Exception):
    """Raised when the HTTP request to the NASA API is incorrect."""

    message: str
    status: int
    url: str | None

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.status}: {self.url} {self.message.lower()}"


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
