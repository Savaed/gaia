from dataclasses import dataclass

import aiohttp


@dataclass
class ApiError(Exception):
    """Raised when the HTTP request to the API is incorrect."""

    message: str
    status: int
    url: str | None = None

    def __str__(self) -> str:
        return f"{self.status}: {self.message} {self.url or ''}"


async def download(url: str, session: aiohttp.ClientSession) -> bytes:
    """Download data from online resources via REST API.

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
            return await resp.read()  # type: ignore[no-any-return]
    except aiohttp.ClientOSError as ex:
        raise ApiError(message=ex.strerror, status=ex.errno, url=url)
    except aiohttp.ClientResponseError as ex:
        raise ApiError(message=ex.message, status=ex.status, url=url)
