import asyncio
import concurrent.futures
import functools
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol, TypeAlias

import aiohttp

from gaia.data.models import Id, NasaTableRequest
from gaia.enums import Cadence
from gaia.http import ApiError, download
from gaia.io import Saver
from gaia.log import logger
from gaia.progress import ProgressBar
from gaia.quarters import get_quarter_prefixes
from gaia.utils import get_chunks, retry


class RestDownloader(Protocol):
    async def download_tables(self, requests: Iterable[NasaTableRequest]) -> None:
        ...

    async def download_time_series(self, ids: Iterable[Id]) -> None:
        ...


class _MissingFileError(Exception):
    """Internal error raised when HTTP 404 is received while downloading time series files."""


_TimeSeriesQueue: TypeAlias = asyncio.Queue[tuple[str, bytes]]
_SavingTasks: TypeAlias = list[tuple[str, asyncio.Future[None]]]


class KeplerDownloader:
    """Downloader for Kepler table and time series files.

    Data is download via REST API from the official archives of NASA and the Mikulski Archive for
    Space Telescopes (MAST).
    """

    MAST_BASE_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:Kepler/url/missions/kepler/lightcurves"
    NASA_BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"

    def __init__(self, saver: Saver, cadence: Cadence, num_async_requests: int = 25) -> None:
        self._saver = saver
        self._checkpoint_filepath = Path.cwd() / f"{self.__class__.__name__}_checkpoint.txt"
        self._cadence = cadence
        self._num_async_requests = num_async_requests
        self._missing_file_urls: list[str] = []
        self._saved_file_urls: list[str] = []

    async def download_tables(self, requests: Iterable[NasaTableRequest]) -> None:
        """Download tables in CSV files from the NASA archive via REST API.

        Downloaded files are saved to the local or external file system e.g. HDFS, AWS etc.

        Args:
            requests (Iterable [NasaTableRequest]): Which tables to download
        """
        async with aiohttp.ClientSession() as sess:
            for request in requests:
                log = logger.bind(table=request.name)
                url = f"{self.NASA_BASE_URL}?{request.query_string}"
                try:
                    table = await download(url, sess)
                    self._raise_for_nasa_error(table)
                except ApiError as ex:
                    log.bind(reason=ex).warning("Cannot download NASA table")
                    continue
                try:
                    self._saver.save_table(f"{request.name}.{request.format}", table)
                except Exception as ex:
                    log.bind(reason=ex).warning("Cannot save NASA table")

    async def download_time_series(self, ids: Iterable[Id]) -> None:
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
        queue: _TimeSeriesQueue = asyncio.Queue(self._num_async_requests)
        downloading_task = self._fetch_time_series(ids, queue)
        saving_task = asyncio.create_task(self._save_time_series(queue))
        await downloading_task
        await queue.join()
        saving_task.cancel()

        try:
            await saving_task
        except asyncio.CancelledError:
            # HACK: When there are no download urls, the `saving_task` is immediately canceled and
            # raise a asyncio.CancelledError
            ...

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

    async def _save_time_series(self, queue: _TimeSeriesQueue) -> None:
        saving_tasks_len = self._num_async_requests * 2
        saving_tasks: _SavingTasks = []
        loop = asyncio.get_running_loop()

        with concurrent.futures.ThreadPoolExecutor(self._num_async_requests) as pool:
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
                self._save_downloaded_urls_checkpoint()

    def _save_downloaded_urls_checkpoint(self) -> None:
        if self._saved_file_urls:
            self._save_meta(self._saved_file_urls)
            logger.info(
                "Downloaded time series urls saved",
                urls=len(self._saved_file_urls),
                path=self._checkpoint_filepath.as_posix(),
            )

    def _add_save_task(
        self,
        saving_tasks: _SavingTasks,
        loop: asyncio.AbstractEventLoop,
        pool: concurrent.futures.ThreadPoolExecutor,
        url: str,
        data: bytes,
    ) -> None:
        _, _, filename = url.rpartition("/")
        task = loop.run_in_executor(pool, self._saver.save_time_series, filename, data)
        saving_tasks.append((url, task))

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
    def _handle_saving_result(self, _: None, url: str) -> None:
        self._saved_file_urls.append(url)
        logger.bind(url=url).debug("FITS file saved")

    @_handle_saving_result.register
    def _(self, result: Exception, url: str) -> None:
        logger.bind(reason=result, url=url).error("Cannot save FITS files")
        raise result

    async def _fetch_time_series(self, ids: Iterable[Id], queue: _TimeSeriesQueue) -> None:
        try:
            tasks = None
            self._checkpoint_filepath.touch()
            urls = self._filter_urls(list(self._create_mast_urls(ids)))

            if not urls:
                logger.info("No URLs to download")
                return

            with ProgressBar() as bar:
                task_id = bar.add_task("Downloading FITS files", total=len(urls))

                for url_batch in get_chunks(urls, self._num_async_requests):
                    tasks = [self._fetch(url) for url in url_batch]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for url, result in zip(url_batch, results):
                        await self._handle_fetch_result(result, url, queue)

                    bar.advance(task_id, len(results))
        except asyncio.CancelledError:
            if tasks:  # pragma: no cover
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._save_missing_urls_checkpoint()

    def _save_missing_urls_checkpoint(self) -> None:
        if self._missing_file_urls:
            self._save_meta(self._missing_file_urls)
            logger.info(
                "Missing time series urls saved",
                urls=len(self._missing_file_urls),
                path=self._checkpoint_filepath.as_posix(),
            )

    @functools.singledispatchmethod
    async def _handle_fetch_result(self, result: bytes, url: str, queue: _TimeSeriesQueue) -> None:
        logger.debug("FITS file download successful", url=url)
        await queue.put((url, result))

    @_handle_fetch_result.register
    async def _(self, result: Exception, url: str, queue: _TimeSeriesQueue) -> None:
        logger.error("Cannot download FITS file", reason=result)
        raise result

    @_handle_fetch_result.register
    async def _(self, result: _MissingFileError, url: str, queue: _TimeSeriesQueue) -> None:
        logger.debug("Missing FITS file", url=url)
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

    def _create_mast_urls(self, ids: Iterable[Id]) -> Iterator[str]:
        logger.bind(unique_ids=len(set(ids))).info("Creating MAST HTTP URLs")
        # This produces urls like e.g:
        # https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:Kepler/url/missions/kepler/lightcurves/0008/000892376//kplr000892376-2009166043257_llc.fits
        for id_ in ids:
            target_id_str = f"{str(id_):0>9}"
            url = f"{self.MAST_BASE_URL}/{target_id_str[:4]}/{target_id_str}//kplr{target_id_str}"
            yield from (
                f"{url}-{prefix}_{self._cadence.value}.fits"
                for prefix in get_quarter_prefixes(self._cadence)
            )

    def _filter_urls(self, urls: Iterable[str]) -> list[str]:
        logger.info("Filtering URLs")
        processed_urls = set(self._checkpoint_filepath.read_text().splitlines())

        if processed_urls:
            logger.info(
                "Checkpoint file detected",
                path=self._checkpoint_filepath.as_posix(),
                processed_urls=len(processed_urls),
            )

        return list(set(urls) - processed_urls)

    def _save_meta(self, urls: list[str]) -> None:
        text = "\n".join(urls) + "\n"
        with open(self._checkpoint_filepath, mode="a") as f:
            f.write(text)
