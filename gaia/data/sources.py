"""Data sources that provide data regardless of its format or source location.

They provide a convenient way to retrieve data such as time series or various scalar values
without worrying about what the data format is or where it is located.
"""

import numbers
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from gaia.data.models import ID, TCE, FromDictMixin
from gaia.io import Reader, ReaderError


class DataSourceError(Exception):
    """Raised when there is an error with the data resource itself.

    Mainly this error can be encountered when the data resources are not found, the connection to
    the resources is broken, the user does not have permission to the resources, etc.
    """


class DataNotFoundError(Exception):
    """Raised when requested data is not found."""

    def __init__(self, *ids: ID) -> None:
        self._ids = ids

    def __str__(self) -> str:  # pragma: no cover
        return f"No data found for IDs={self._ids}"


@dataclass
class MissingDataPartsError(Exception):
    """Raised when some required parts of data are missing."""

    parts: Iterable[str]

    def __str__(self) -> str:  # pragma: no cover
        missing_parts = ", ".join(self.parts)
        return f"Following requested data parts are missing: {missing_parts}"


# TODO: Add proper type hints in the future
TTimeSeries = TypeVar("TTimeSeries")  # Any dict[str, Sequence[numbers.Number]].


class TimeSeriesSource(Generic[TTimeSeries]):
    """Data source for time series like e.g. light curves.

    The data dictionary from the specified reader must have keys matching the `TTimeSeries` keys.
    """

    def __init__(self, reader: Reader[dict[str, TTimeSeries]]) -> None:
        self._reader = reader

    def get(
        self,
        id_: ID,
        squeeze: bool = False,
        parts: Iterable[str] | None = None,
    ) -> dict[str, TTimeSeries]:
        """Retrieve time series for the particular observation target.

        Args:
            id_ (ID): The id of the target e.g. the id of KOI for Kepler dataset. `str` or `int`.
            squeeze (bool, optional): Whether to flatten series for all series parts into a single
                one. Defaults to False.
            parts (Iterable[str] | None, optional): Which parts (e.g. observation quarters for
                Kepler dataset) of the time series to return. If None then all available parts are
                returned. Defaults to None.

        Raises:
            DataNotFoundError: No requested time series found
            DataSourceError: The data could not be read due to a data resource problem
            MissingDataPartsError: Requested data parts are unavailable

        Returns:
            dict[str, TTimeSeries]: Data in the format 'series_part' -> 'TTimeSeries'.
                If `squeeze` is set to True then all partial time series (separate series for each
                part of observations) are combined together and returned as a dictionary with only
                one key: 'all' and values as combined time series in `TTimeSeries` format.
        """
        try:
            series_dict = self._reader.read(str(id_))[0]
        except KeyError:
            raise DataNotFoundError(id_)
        except ReaderError as ex:
            raise DataSourceError(ex)

        missing_parts = set(parts) - set(series_dict) if parts else None
        if missing_parts:
            raise MissingDataPartsError(missing_parts)

        series = (
            {part: value for part, value in series_dict.items() if part in parts}
            if parts
            else series_dict
        )
        return self._squeeze(series) if squeeze else series

    def _squeeze(self, data: dict[str, TTimeSeries]) -> dict[str, TTimeSeries]:
        output: dict[str, list[npt.ArrayLike]] = defaultdict(list)
        for data_parts in data.values():
            for key, series in data_parts.items():  # type: ignore
                output[key].append(series)

        flat_series: dict[str, npt.NDArray[np.float_]] = {
            k: np.concatenate(v) for k, v in output.items()
        }
        return {"all": flat_series}  # type: ignore


TTCE = TypeVar("TTCE", bound=TCE)
SimpleDict: TypeAlias = dict[str, str | numbers.Number]


class TceSource(Generic[TTCE]):
    """Data source for TCEs.

    The data dictionary from the specified reader must have keys matching the `TTCE` keys.
    """

    def __init__(self, reader: Reader[SimpleDict]) -> None:
        self._reader = reader

    def get_all_for_target(self, target_id: ID) -> list[TTCE]:
        """Retrieve all TCEs for a particular observation target if any.

        Args:
            target_id (ID): The id of the target e.g. the id of KOI for Kepler dataset.
                `str` or `int`.

        Raises:
            DataNotFoundError: No requested target found
            DataSourceError: The data could not be read due to a data resource or decompression
                error encountered

        Returns:
            list[TTCE]: All TCEs for the target. Can be an empty list
        """
        return self._get(target_id)

    def get(self, target_id: ID, tce_id: ID) -> TTCE:
        """Retrieve Threshold-Crossing Event (TCE) for a particular observation target.

        Args:
            target_id (ID): The id of the target e.g. the id of KOI for Kepler dataset.
                `str` or `int`.
            tce_id (ID): The id of the TCE. Usually TCE number. `str` or `int`

        Raises:
            DataNotFoundError: No requested target or TCE found
            DataSourceError: The data could not be read due to a data resource or decompression
                error encountered

        Returns:
            TTCE: Requested TCE
        """
        tces = self._get(target_id)
        try:
            return [tce for tce in tces if str(tce.tce_id) == tce_id][0]
        except IndexError:
            raise DataNotFoundError(target_id, tce_id)

    def _get(self, target_id: ID) -> list[TTCE]:
        # HACK: This is a `TTCE` real type at runtime. See: https://stackoverflow.com/a/63318205
        generic_type: type[TTCE] = self.__orig_class__.__args__[0]  # type: ignore
        try:
            result = self._reader.read(str(target_id))
        except KeyError:
            raise DataNotFoundError(target_id)
        except ReaderError as ex:
            raise DataSourceError(ex)

        try:
            return [generic_type.from_flat_dict(row) for row in result]
        except KeyError as ex:
            raise DataSourceError(
                f"Unable to create an instance of '{generic_type.__name__}'. {ex}",
            )


TParams = TypeVar("TParams", bound=FromDictMixin)


class StellarParametersSource(Generic[TParams]):
    """Data source for physical properties of observation star or binary/multiple system target.

    The data dictionary from the specified reader must have keys matching the `TParams` keys.
    """

    def __init__(self, reader: Reader[SimpleDict]) -> None:
        self._reader = reader

    def get(self, id_: ID) -> TParams:
        """Retrieve physical properties for a particular observation target.

        Args:
            _id (ID): The id of the target e.g. the id of KOI for Kepler dataset. `str` or `int`.

        Raises:
            DataNotFoundError: No requested target found
            DataSourceError: The data could not be read due to a data resource or decompression
                error encountered

        Returns:
            TTCE: Requested physical properties of star or binary/multiple system
        """
        try:
            params_dict = self._reader.read(str(id_))[0]
        except KeyError:
            raise DataNotFoundError(id_)
        except ReaderError as ex:
            raise DataSourceError(ex)

        # HACK: This is a `TParams` real type at runtime. See: https://stackoverflow.com/a/63318205
        generic_type: type[TParams] = self.__orig_class__.__args__[0]  # type: ignore
        try:
            return generic_type.from_flat_dict(params_dict)  # type: ignore
        except KeyError as ex:
            raise DataSourceError(
                f"Unable to create an instance of '{generic_type.__name__}'. {ex}",
            )
