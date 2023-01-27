"""
Class and functions to conveniently download all kinds of time series and tabular data
e.g. NASA data tables or Kepler light curves.

Downloading from NASA's official REST API and MAST archive is supported.
For more information on what you can download, see:
   https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html - NASA
   https://archive.stsci.edu/missions-and-data/kepler/kepler-bulk-downloads - MAST
"""

import asyncio
import concurrent.futures
import functools
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeAlias

import aiohttp
import structlog

from gaia.enums import Cadence
from gaia.http import ApiError, download
from gaia.io import Saver
from gaia.quarters import get_quarter_prefixes
from gaia.utils import retry


log = structlog.stdlib.get_logger()


@dataclass
class TableRequest:
    name: str
    query: str | None = None
    columns: list[str] = field(default_factory=list)


class Downloader(Protocol):
    async def download_tables(self, requests: Iterable[TableRequest]) -> None:
        ...

    async def download_time_series(self, ids: Iterable[int]) -> None:
        ...


class _MissingFileError(Exception):
    """Internal error reported when HTTP 404 is received while downloading time series files."""


_TimeSeriesQueue: TypeAlias = asyncio.Queue[tuple[str, bytes]]
_SavingTasks: TypeAlias = list[tuple[str, asyncio.Future[None]]]


class KeplerDownloader:
    """Downloader for fetching and saving Kepler table and time series files.

    Tables in CSV format and time series in FITS format are fetched asynchronously via REST API
    from the official archives of NASA and the Mikulski Archive for Space Telescopes (MAST) and
    then saved to the local or external destination.
    """

    def __init__(
        self,
        saver: Saver,
        cadence: Cadence,
        nasa_base_url: str,
        mast_base_url: str,
        num_tasks: int = 40,
    ) -> None:
        self._saver = saver
        self._nasa_base_url = nasa_base_url.rstrip("/")
        self._mast_base_url = mast_base_url.rstrip("/")
        self._urls_meta_path = Path().home() / f"{self.__class__.__name__}_series.txt"
        self._cadence = cadence
        self._num_tasks = num_tasks
        self._missing_file_urls: list[str] = []
        self._saved_file_urls: list[str] = []

    async def download_tables(self, requests: Iterable[TableRequest]) -> None:
        """Download tables in CSV files from the NASA archive.

        Downloaded files are saved to the local or external file system e.g. HDFS, AWS etc.

        Args:
            requests (Iterable [TableRequest]): Requests specifying which tables to download
        """
        async with aiohttp.ClientSession() as sess:
            for request in requests:
                try:
                    table = await download(self._create_table_url(request), sess)
                    self._raise_for_nasa_error(table)
                except ApiError as ex:
                    log.warning("Cannot download NASA table", reason=ex, table=request.name)
                    continue
                try:
                    self._saver.save_table(request.name, table)
                except Exception as ex:
                    log.warning("Cannot save NASA table", reason=ex, table=request.name)

    async def download_time_series(self, ids: Iterable[int]) -> None:
        """Download time series in FITS files from the MAST archive.

        Downloaded files are saved in a local or external file system, e.g. HDFS, AWS, etc.
        Fetching and saving are done asynchronously with the retry strategy for download errors.
        When all retries have failed, the original error is re-raised to the caller. Any save error
        is immediately re-raised.

        This method implements a mechanism to prevent re-downloading of already downloaded files.
        URLs that have been processed are saved locally in the metadata text file.

        Args:
            ids (Iterable[int]): The IDs of the targets for which the time series are downloaded
        """
        queue: _TimeSeriesQueue = asyncio.Queue(self._num_tasks)
        downloading_task = self._fetch_time_series(ids, queue)
        saving_task = asyncio.create_task(self._save_time_series(queue))
        await downloading_task
        await queue.join()
        saving_task.cancel()
        await saving_task

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
        return f"{self._nasa_base_url}?{table}{select}{query}&format=csv"  # Only csv for now

    async def _save_time_series(self, queue: _TimeSeriesQueue) -> None:
        saving_tasks_len = self._num_tasks * 2
        saving_tasks: _SavingTasks = []
        loop = asyncio.get_running_loop()

        with concurrent.futures.ThreadPoolExecutor(self._num_tasks) as pool:
            try:
                while True:
                    url, data = await self._get_queue_item(queue)
                    self._add_save_task(saving_tasks, loop, pool, url, data)

                    if len(saving_tasks) == saving_tasks_len:
                        await self._await_save(saving_tasks)  # pragma: no cover
            except asyncio.CancelledError:
                # After cancelling, start save tasks for any files not yet saved
                for _ in range(queue.qsize()):  # pragma: no cover
                    self._add_save_task(saving_tasks, loop, pool, *queue.get_nowait())
                if saving_tasks:
                    await self._await_save(saving_tasks)
            finally:
                self._save_meta(self._saved_file_urls)

    def _add_save_task(
        self,
        tasks: _SavingTasks,
        loop: asyncio.AbstractEventLoop,
        pool: concurrent.futures.ThreadPoolExecutor,
        url: str,
        data: bytes,
    ) -> None:
        filename = url.split("/")[-1]
        item = filename, loop.run_in_executor(pool, self._saver.save_time_series, filename, data)
        tasks.append(item)

    async def _get_queue_item(self, queue: _TimeSeriesQueue) -> tuple[str, bytes]:
        url, data = await queue.get()
        queue.task_done()
        return url, data

    async def _await_save(self, tasks: _SavingTasks) -> None:
        urls, save_tasks = list(zip(*tasks))
        results = await asyncio.gather(*save_tasks, return_exceptions=True)
        tasks.clear()
        for url, result in zip(urls, results):
            self._handle_saving_result(result, url)

    @functools.singledispatchmethod
    def _handle_saving_result(self, result: None, url: str) -> None:
        log.info("FITS file save sucessful", url=url)
        self._saved_file_urls.append(url)

    @_handle_saving_result.register
    def _(self, result: Exception, url: str) -> None:
        # Is this log even necessary?
        log.error("Cannot save FITS files", reason=result, url=url)
        raise result

    async def _fetch_time_series(self, ids: Iterable[int], queue: _TimeSeriesQueue) -> None:
        log.info("Start downloading FITS files")
        self._urls_meta_path.touch(exist_ok=True)
        urls = self._filter_urls(list(self._create_mast_urls(ids)))

        try:
            for url_batch in self._url_batches(urls):
                tasks = [self._fetch(url) for url in url_batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for url, result in zip(url_batch, results):
                    await self._handle_fetch_result(result, url, queue)
        finally:
            self._save_meta(self._missing_file_urls)

    @functools.singledispatchmethod
    async def _handle_fetch_result(self, result: bytes, url: str, queue: _TimeSeriesQueue) -> None:
        log.info("FITS file download successful", url=url)
        await queue.put((url, result))

    @_handle_fetch_result.register
    async def _(self, result: Exception, url: str, queue: _TimeSeriesQueue) -> None:
        log.error("Cannot download FITS file", reason=result)
        raise result

    @_handle_fetch_result.register
    async def _(self, result: _MissingFileError, url: str, queue: _TimeSeriesQueue) -> None:
        log.info("Missing FITS file", url=url)
        self._missing_file_urls.append(url)

    @retry(on=ApiError)
    async def _fetch(self, url: str) -> bytes:
        async with aiohttp.ClientSession() as sess:
            try:
                return await download(url, sess)
            except ApiError as ex:
                if ex.status == 404:
                    raise _MissingFileError
                raise
            except asyncio.TimeoutError:
                # In case of aiohttp session timeout (300s by default)
                raise ApiError("HTTP request timeout", 408, url)

    def _create_mast_urls(self, ids: Iterable[int]) -> Iterator[str]:
        log.info("Creating MAST HTTP URLs")
        # This produces urls like e.g:
        # https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:Kepler/url/missions/kepler/lightcurves/0008/000892376//kplr000892376-2009166043257_llc.fits
        for id_ in ids:
            id_str = f"{id_:09d}"
            url = f"{self._mast_base_url}/{id_str[:4]}/{id_str}//kplr{id_str}"
            yield from (
                f"{url}-{pref}_{self._cadence.value}.fits"
                for pref in get_quarter_prefixes(self._cadence)
            )

    def _filter_urls(self, urls: Iterable[str]) -> list[str]:
        processed_urls = set(self._urls_meta_path.read_text().splitlines())
        if processed_urls:
            meta_path = self._urls_meta_path.as_posix()
            log.info("Metadata file detected. Skip URLs from this file", path=meta_path)
        return list(set(urls) - processed_urls)

    def _save_meta(self, urls: list[str]) -> None:
        if urls:
            text = "\n".join(urls) + "\n"
            filepath = self._urls_meta_path.as_posix()
            with open(filepath, mode="a") as f:
                f.write(text)
            log.info("Metadata file saved", number_of_urls=len(urls), path=filepath)

    def _url_batches(self, urls: list[str]) -> Iterator[Iterable[str]]:
        log.info("Creating URLs batches", batch_size=self._num_tasks)
        yield from (urls[n : n + self._num_tasks] for n in range(0, len(urls), self._num_tasks))
