"""
Class and functions to conveniently download all kinds of time series and tabular data
e.g. NASA data tables or Kepler light curves.

Downloading from NASA's official REST API and MAST archive is supported.
For more information on what you can download, see:
   https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html - NASA
   https://archive.stsci.edu/missions-and-data/kepler/kepler-bulk-downloads - MAST
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeAlias

import aiohttp
import structlog

from gaia.http import ApiError, download
from gaia.io import Saver


log = structlog.stdlib.get_logger()

DownloadResult: TypeAlias = tuple[str, bytes]


@dataclass
class TableRequest:
    name: str
    query: str | None = None
    columns: set[str] = field(default_factory=set)


class Downloader(Protocol):
    async def download_tables(self, requests: Iterable[TableRequest]) -> None:
        ...

    async def download_time_series(self, ids: Iterable[int]) -> None:
        ...


class KeplerDownloader:
    def __init__(self, saver: Saver, nasa_base_url: str, mast_base_url: str) -> None:
        self._saver = saver
        self._nasa_base_url = nasa_base_url.rstrip("/")
        self._mast_base_url = mast_base_url.rstrip("/")
        home = Path().home()
        self._tables_meta_path = home / f"{self.__class__.__name__}_tables.txt"
        self._series_meta_path = home / f"{self.__class__.__name__}_series.txt"

    async def download_tables(self, requests: Iterable[TableRequest]) -> None:
        async with aiohttp.ClientSession() as sess:
            for request in requests:
                try:
                    table = await download(self._create_table_url(request), sess)
                    self._raise_for_nasa_error(table)
                except ApiError as ex:
                    log.warning("Cannot download NASA table", reason=ex)
                    continue

                try:
                    self._saver.save_table(request.name, table)
                except Exception as ex:
                    log.warning("Cannot save NASA table", reason=ex)

    def _raise_for_nasa_error(self, response_body: bytes) -> None:
        # Error from NASA API is of format:
        # ERROR<br>
        # Error Type: {error_type}<br>
        # Message:    {error_message}.
        if response_body.startswith(b"ERROR<br>"):
            try:
                error_msg = re.search(rb"(?<=Message:).*", response_body).group()  # type: ignore
            except AttributeError:
                error_msg = b"unknown error"
            msg = error_msg.strip(b" .").decode("utf-8")
            raise ApiError(message=msg, status=200, url=None)

    def _create_table_url(self, request: TableRequest) -> str:
        table = f"table={request.name}"
        select = f"&select={','.join(request.columns)}" if request.columns else ""
        query = f"&where={request.query}" if request.query else ""
        return f"{self._nasa_base_url}?{table}{select}{query}&format=csv"
