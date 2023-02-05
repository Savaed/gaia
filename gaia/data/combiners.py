"""
Classes to combine data from multiple objects (files, SQL tables, documents, etc.) into fewer
output objects for easier future processing.

This can be useful for reducing the number of sources you need to read next pre-processing stage.
One example is reducing the number of FITS files in the Kepler time-series dataset, where the data
is split into many files for each observation quarter for every single KOI.
"""

import asyncio
import concurrent.futures
import functools
import re
from collections import defaultdict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Generic, Protocol, TypeAlias, TypeVar

import structlog
import tensorflow as tf
from rich.progress import track

from gaia.io import read_fits_table


log = structlog.stdlib.get_logger()

T = TypeVar("T", bound=dict[Any, Any])


class CombinerError(Exception):
    """Raised when the combining process cannot be fully completed."""


class Combiner(Protocol[T]):  # type: ignore
    async def combine(self, fields: set[str] | None = None) -> AsyncIterator[tuple[str, T]]:
        ...


# This is in the format: kepid -> list[tuple[quarter, filepath]]
_Groups: TypeAlias = dict[str, list[tuple[str, str]]]


class KeplerFitsCombiner(Generic[T]):
    """File combiner to combine multiple time series FITS files for a single target (KOI)."""

    def __init__(
        self,
        data_dir: str,
        id_pattern: str,
        quarter_pattern: str,
        hdu_header: str = "LIGHTCURVE",
    ) -> None:
        self._data_dir = data_dir
        self._kepid_pattern = re.compile(id_pattern)
        self._quarter_pattern = re.compile(quarter_pattern)
        self._header = hdu_header
        self._meta_path = Path().home() / f"{self.__class__.__name__}_metadata.txt"

    async def combine(self, fields: set[str] | None = None) -> AsyncIterator[tuple[str, T]]:
        """Combine multiple FITS files into one dictionary for each KOI.

        This function combines multiple files for one KOI into dictionaries in the format:
        {'quarter_prefix': 'T'}', where 'T' is data format in FITS files. Moreover it checks
        files already combined (from the metadata text file) and does not process them again.

        Args:
            fields (set[str] | None, optional): Fields to keep in the output dictionaries.
                If None, all available fields will be kept. Default None.

        Raises:
            CombinerError: The fields passed to the `fields` parameter and the fields from the FITS
                files do not match OR the file HDU extension is missing OR the file paths cannot be
                grouped by KOI and/or quarter prefixes.

        Yields:
            Iterator[AsyncIterator[tuple[str, T]]: A tuple with KOI ID and dict with combined data
                for each KOI.
        """
        self._meta_path.touch(exist_ok=True)
        loop = asyncio.get_running_loop()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            for kepid, paths_group in track(self._group_paths().items(), "Combining FITS files"):
                output: T = defaultdict(dict)  # type: ignore
                quarter_prefixes, paths = list(zip(*paths_group))
                tasks = self._start_read_tasks(fields, loop, paths, pool)
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for prefix, result in zip(quarter_prefixes, results):
                    self._handle_read_result(result, prefix, output)

                log.debug("File combine succeeded", kepid=kepid)
                yield kepid, output
                self._save_meta(kepid)

    def _save_meta(self, kepid: str) -> None:
        with open(self._meta_path, mode="a") as f:
            f.write(f"{kepid}\n")
        log.debug("Metadata saved", kepid=kepid)

    @functools.singledispatchmethod
    def _handle_read_result(
        self,
        result: T,
        quarter_prefix: str,
        output: dict[Any, Any],
    ) -> None:
        output[quarter_prefix] = result

    @_handle_read_result.register
    def _(self, result: Exception, quarter_prefix: str, output: dict[Any, Any]) -> None:
        raise CombinerError(result)

    def _start_read_tasks(
        self,
        fields: set[str] | None,
        loop: asyncio.AbstractEventLoop,
        paths: list[str],
        pool: concurrent.futures.ThreadPoolExecutor,
    ) -> list[asyncio.Future[dict[str, Any]]]:
        return [loop.run_in_executor(pool, read_fits_table, p, self._header, fields) for p in paths]

    def _group_paths(self) -> _Groups:
        if not tf.io.gfile.exists(self._data_dir):
            raise CombinerError(f"Root data directory {self._data_dir!r} not found")

        groups: _Groups = defaultdict(list)
        processed_kepids = self._read_meta()

        log.info("Grouping paths based on kepid", data_dir=self._data_dir)
        for path in tf.io.gfile.listdir(self._data_dir):
            kepid = self._substract_from_path(path, self._kepid_pattern)

            if kepid in processed_kepids:
                continue

            quarter_prefix = self._substract_from_path(path, self._quarter_pattern)
            # `tf.io.gfile.listdir()` returns only filenames, but we need full paths
            path = f"{self._data_dir}/{path}"
            groups[kepid].append((quarter_prefix, path))

        log.info(f"Paths splited into {len(groups)} group(s)")
        return groups

    def _read_meta(self) -> set[str]:
        processed_kepids = set(self._meta_path.read_text().splitlines())
        if processed_kepids:
            log.info(
                "Metadata file detected. Skip processed files",
                path=str(self._meta_path),
                kepids_len=len(processed_kepids),
            )
        return processed_kepids

    def _substract_from_path(self, path: str, pattern: re.Pattern[str]) -> str:
        try:
            return pattern.search(path).group()  # type: ignore[union-attr]
        except AttributeError:
            raise CombinerError(
                f"Cannot substract quarter prefix or kepid from {path=!r} using {pattern=!r}",
            )
