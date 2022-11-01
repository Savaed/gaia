"""Kepler data models shared in the project."""

from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass
from typing import Any

import numpy as np


class DictConvertibleObject(metaclass=ABCMeta):
    """
    Abstract class for objects created from dict-like data with optional key-attribute names
    mapping.
    """

    @classmethod
    def from_dict(
        cls, data_dict: dict[str, Any], field_map: dict[str, str] | None = None
    ) -> DictConvertibleObject:
        """Create an object from a dict.

        Parameters
        ----------
        data_dict : dict[str, Any]
            Data which initialize an object
        field_map : Optional[dict[str, str]], optional
            Fields mapping. Keys are object attributes, and values are `data_dict` keys. If it is
            empty or None keys in `data_dict` must match keyword arguments of the created object,
            by default None

        Returns
        -------
        DictConvertibleObject
            The object initializes with provided `data_dict` values
        """
        data = (
            {field: data_dict[key] for field, key in field_map.items()} if field_map else data_dict
        )
        try:
            return cls(**data)
        except TypeError as ex:
            bad_key = str(ex).split()[-1]
            raise ValueError(
                f"Unexpected key {bad_key} provided for {cls.__name__}.__init__()"
            ) from ex


@dataclass
class TimeSeries:
    """Single feature of the Kepler time series."""

    time: list[np.ndarray]
    values: list[np.ndarray]
