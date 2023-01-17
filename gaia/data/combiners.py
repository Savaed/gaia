"""
Classes to combine data from multiple objects (files, SQL tables, documents, etc.) into fewer
output objects for easier future processing.

This can be useful for reducing the number of sources you need to read next pre-processing stage.
One example is reducing the number of FITS files in the Kepler time-series dataset, where the data
is split into many files for each observation quarter for every single KOI.
"""

import asyncio
import concurrent.futures
import re
from collections import defaultdict
from collections.abc import Iterable
from functools import singledispatchmethod
from typing import Any, Generic, Protocol, TypeAlias, TypedDict, TypeVar

import numpy as np
import structlog
import tensorflow as tf

from gaia.io import read_fits_table


log = structlog.stdlib.get_logger()

T = TypeVar("T", bound=dict[str, Any])

# Represents time series observed at specific periods, e.g. quarters for Kepler time series
IntervalTimeSeries: TypeAlias = dict[str, dict[str, T]]


class KeplerTimeSeries(TypedDict):
    TIME: list[np.ndarray]
    SAP_FLUX: list[np.ndarray]
    PDCSAP_FLUX: list[np.ndarray]


class CombinerError(Exception):
    """Raised when cannot process object(s) in combining."""


class FilesCombiner(Protocol[T]):
    async def combine(self, paths: Iterable[str]) -> IntervalTimeSeries[T]:
        ...


class FitsCombiner(Generic[T]):
    """
    Provides a convenient way to combine tabular data from multiple FITS files into a single output
    the dictionary uses the `astropy` module to read
    """

    def __init__(
        self,
        quarter_pattern: str,
        hdu_header: str = "LIGHTCURVE",
        fields: Iterable[str] | None = None,
    ) -> None:
        super().__init__()
        self._fields = set(fields) if fields else None
        self._quarter_pattern = re.compile(quarter_pattern)
        self._hdu_header = hdu_header

    async def combine(self, paths: list[str]) -> dict[str, T]:
        """Combine tables from multiple FITS files read from `paths` into a single dictionary.

        Args:
            paths (Iterable[str]): Paths to FITS files. It should be all paths for a single object

        Raises:
            CombinerError: FITS file reading error, connection error (for external resources like
                GCS) or ID/quarter extraction error

        Returns:
            dict[str, T]: Dictionary with combined data from all files
        """
        loop = asyncio.get_running_loop()
        quarters_prefixes, filtered_paths = self._extract_quarter_prefixes(list(paths))

        with concurrent.futures.ThreadPoolExecutor() as pool:
            tasks = [
                loop.run_in_executor(pool, read_fits_table, path, self._hdu_header, self._fields)
                for path in filtered_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            out: dict[str, T] = {}
            for prefix, data in zip(quarters_prefixes, results):
                self._handle_result(data, prefix, out)

            if not out:
                raise CombinerError("File reading failed for all paths")

            return out

    @singledispatchmethod
    def _handle_result(self, result: T, prefix: str, out: dict[str, T]) -> None:
        out[prefix] = result

    @_handle_result.register
    def _(self, result: Exception, prefix: str, out: dict[str, T]) -> None:
        log.warning(str(result))

    def _extract_quarter_prefixes(self, paths: list[str]) -> tuple[list[str], list[str]]:
        prefixes: list[str] = []
        filtered_paths = paths.copy()

        for path in paths:
            try:
                prefixes.append(self._quarter_pattern.search(path).group())  # type: ignore
            except AttributeError:
                msg = (
                    f"Cannot retrieve quarter prefix from {path!r} using "
                    f"pattern {self._quarter_pattern.pattern!r}"
                )
                log.warning(msg)
                filtered_paths.remove(path)

        if not any(prefixes):
            raise CombinerError("Quarter prefix extraction failed for all paths")

        return prefixes, filtered_paths


def group_kepler_paths(data_dir: str, kepid_pattern: str) -> dict[str, list[str]]:
    """Group Kepler time series FITS file paths into the dictionary where KIC (kepid) is the key.

    Args:
        data_dir (str): Root data directory path
        kepid_pattern (str): Regex pattern to extract kepid from file paths

    Returns:
        dict[str, list[str]]: Dictionary 'kepid' -> ['file_path1', 'file_path2', ...]
    """
    log.info("Grouping paths based on kepid", data_dir=data_dir)
    groups = defaultdict(list)
    paths = tf.io.gfile.glob(f"{data_dir.rstrip('/')}/*.fits")
    pattern = re.compile(kepid_pattern)
    log.info(f"Found {len(paths)} paths")

    for path in paths:
        try:
            kepid = pattern.search(path).group()  # type: ignore[union-attr]
        except AttributeError:
            log.warning(f"Cannot retrieve kepid from {path!r} using a pattern {kepid_pattern!r}")
        else:
            groups[kepid].append(path)

    log.info(f"Paths grouped into {len(groups)} groups")
    return groups
