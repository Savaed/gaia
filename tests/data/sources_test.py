from dataclasses import dataclass
from typing import TypedDict
from unittest.mock import Mock

import numpy as np
import pytest

from gaia.data.models import TCE, FromDictMixin
from gaia.data.sources import (
    DataNotFoundError,
    DataSourceError,
    MissingDataPartsError,
    StellarParametersSource,
    TceSource,
    TimeSeriesSource,
)
from gaia.io import CsvTableReader, Reader, ReaderError
from tests.conftest import assert_dict_with_numpy_equal


class TimeSeriesTestDict(TypedDict):
    a: list[int]
    b: list[int]


TEST_TIME_SERIES_DICT = {
    "quarter1": TimeSeriesTestDict(a=[1], b=[2]),
    "quarter2": TimeSeriesTestDict(a=[3], b=[4]),
}


@pytest.fixture
def time_series_reader():
    """Return/Mock/Prepare."""
    reader = Mock(spec=Reader[dict[str, TimeSeriesTestDict]])
    reader.read.return_value = [TEST_TIME_SERIES_DICT]
    return reader


@pytest.fixture
def error_time_series_reader():
    """Return/Mock/Prepare."""
    reader = Mock(spec=Reader[dict[str, TimeSeriesTestDict]])
    reader.read.side_effect = ReaderError("Cannot read underlying data")
    return reader


@pytest.fixture
def id_not_found_time_series_reader():
    """Return/Mock/Prepare."""
    reader = Mock(spec=Reader[dict[str, TimeSeriesTestDict]])
    reader.read.side_effect = KeyError(999)
    return reader


def test_time_series_source_get__reader_error(error_time_series_reader):
    """Test that DataSourceError is raised when there is any error with reading the data."""
    source = TimeSeriesSource[TimeSeriesTestDict](error_time_series_reader)
    with pytest.raises(DataSourceError):
        source.get("1")


def test_time_series_source_get__time_series_not_found(id_not_found_time_series_reader):
    """Test that DataNotFoundError is raised when no ID matching time series found."""
    source = TimeSeriesSource[TimeSeriesTestDict](id_not_found_time_series_reader)
    with pytest.raises(DataNotFoundError):
        source.get("999")


@pytest.mark.parametrize(
    "parts,expected",
    [
        (None, TEST_TIME_SERIES_DICT),
        (["quarter1"], {"quarter1": {"a": [1], "b": [2]}}),
        (["quarter1", "quarter2"], TEST_TIME_SERIES_DICT),
        (["quarter1", "quarter2", "quarter2"], TEST_TIME_SERIES_DICT),
    ],
    ids=[
        "all_by_default",
        "few_specified",
        "all_by_specified",
        "all_by_specified_some_parts_repeat",
    ],
)
def test_time_series_source_get__get_specified_parts(parts, expected, time_series_reader):
    """Test that only specified data parts are returned."""
    source = TimeSeriesSource[TimeSeriesTestDict](time_series_reader)
    result = source.get("1", parts=parts)
    assert result == expected


@pytest.mark.parametrize(
    "parts",
    [
        ["quarter99", "quarter999"],
        ["quarter1", "quarter999"],
    ],
    ids=["all_are_missing", "one_is_missing"],
)
def test_time_series_source_get__missing_parts(parts, time_series_reader):
    """Test that MissingDataPartsError is raised when some of specifed parts are missing."""
    source = TimeSeriesSource[TimeSeriesTestDict](time_series_reader)
    with pytest.raises(MissingDataPartsError):
        source.get("1", parts=parts)


@pytest.mark.parametrize(
    "parts,expected",
    [
        (["quarter1"], {"a": np.array([1]), "b": np.array([2])}),
        (["quarter1", "quarter2"], {"a": np.array([1, 3]), "b": np.array([2, 4])}),
    ],
    ids=["one_part", "multiple_parts"],
)
def test_time_series_source_get__squeeze_output(parts, expected, time_series_reader):
    """Test that ."""
    source = TimeSeriesSource[TimeSeriesTestDict](time_series_reader)
    result = source.get("1", parts=parts, squeeze=True)
    assert_dict_with_numpy_equal(result["all"], expected)  # type: ignore


class CsvTestDict(TypedDict):
    target_id: int
    tce_id: int
    tce_test: str
    params_test: float


@dataclass
class TCETestClass(TCE):
    tce_test: str


@dataclass
class StellarParamsTestClass(FromDictMixin):
    params_test: float


TEST_CSV_DICT = CsvTestDict(target_id=1, tce_id=1, tce_test="tce", params_test=1.2)


@pytest.fixture
def valid_csv_reader():
    """Return/Mock/Prepare."""
    reader = Mock(spec=CsvTableReader[CsvTestDict])
    reader.read.return_value = [TEST_CSV_DICT]
    return reader


@pytest.fixture
def error_csv_reader():
    """Return/Mock/Prepare."""
    reader = Mock(spec=CsvTableReader[CsvTestDict])
    reader.read.side_effect = ReaderError("Cannot read source CSV test file")
    return reader


@pytest.fixture
def id_not_found_csv_reader():
    """Return/Mock/Prepare."""
    reader = Mock(spec=CsvTableReader[CsvTestDict])
    reader.read.side_effect = KeyError(999)
    return reader


def test_stellar_parameters_source_get__reader_error(error_csv_reader):
    """Test that DataSourceError is raised when there is any error with reading the data."""
    source = StellarParametersSource[StellarParamsTestClass](error_csv_reader)
    with pytest.raises(DataSourceError):
        source.get("1")


def test_stellar_parameters_source_get__stellar_params_not_found(id_not_found_csv_reader):
    """Test that DataNotFoundError is raised when no ID matching object found."""
    source = StellarParametersSource[StellarParamsTestClass](id_not_found_csv_reader)
    with pytest.raises(DataNotFoundError):
        source.get("999")


def test_stellar_parameters_source_get__object_from_dict_create_error():
    """Test that DataSourceError is raised when cannot create the object from dictionary data.

    This may happen when a dictionary read via reader has different keys than class field names.
    """

    class _InvalidTestDict(TypedDict):
        x: int

    reader_mock = Mock(spec=Reader[_InvalidTestDict])
    # Return {"x": int}, but {"params_test": float} is required
    reader_mock.read.return_value = [_InvalidTestDict(x=123)]
    source = StellarParametersSource[StellarParamsTestClass](reader_mock)
    with pytest.raises(DataSourceError):
        source.get("1")


def test_stellar_parameters_source_get__return_correct_data(valid_csv_reader):
    """Test that ."""
    source = StellarParametersSource[StellarParamsTestClass](valid_csv_reader)
    result = source.get("1")
    assert result == StellarParamsTestClass(params_test=1.2)


def test_tce_source_get_all_for_target__reader_error(error_csv_reader):
    """Test that ."""
    source = TceSource[TCETestClass](error_csv_reader)
    with pytest.raises(DataSourceError):
        source.get_all_for_target("1")


def test_tce_source_get_all_for_target__target_not_found(
    id_not_found_csv_reader,
):
    """Test that DataNotFoundError is raised when no ID matching object found."""
    source = TceSource[TCETestClass](id_not_found_csv_reader)
    with pytest.raises(DataNotFoundError):
        source.get_all_for_target("999")


def test_tce_source_get_all_for_target__object_from_dict_create_error():
    """Test that DataSourceError is raised when cannot create the object from dictionary data.

    This may happen when a dictionary read via reader has different keys than class field names.
    """

    class _InvalidTestDict(TypedDict):
        x: int

    reader_mock = Mock(spec=Reader[_InvalidTestDict])
    # Return {"x": int}, but {"tce_test": str, target_id: int | str} is required
    reader_mock.read.return_value = [_InvalidTestDict(x=123)]
    source = TceSource[TCETestClass](reader_mock)
    with pytest.raises(DataSourceError):
        source.get_all_for_target("1")


@pytest.mark.parametrize(
    "reader_output,expected",
    [
        (
            [CsvTestDict(target_id=1, tce_id=1, tce_test="tce", params_test=1.2)],
            [TCETestClass(target_id=1, tce_id=1, tce_test="tce")],
        ),
        (
            [
                CsvTestDict(target_id=1, tce_id=1, tce_test="tce", params_test=1.2),
                CsvTestDict(target_id=1, tce_id=2, tce_test="tce2", params_test=1.2),
            ],
            [
                TCETestClass(target_id=1, tce_id=1, tce_test="tce"),
                TCETestClass(target_id=1, tce_id=2, tce_test="tce2"),
            ],
        ),
    ],
    ids=["one_tce", "many_tces"],
)
def test_tce_source_get_all_for_target__return_correct_data(reader_output, expected):
    """Test that ."""
    reader_mock = Mock(spec=CsvTableReader[CsvTestDict])
    reader_mock.read.return_value = reader_output
    source = TceSource[TCETestClass](reader_mock)
    result = source.get_all_for_target("1")
    assert result == expected


def test_tce_source_get__reader_error(error_csv_reader):
    """Test that ."""
    source = TceSource[TCETestClass](error_csv_reader)
    with pytest.raises(DataSourceError):
        source.get("1", "2")


# 2 przypadki: nie ma tagetu, jest target, ale nie ma tce
@pytest.mark.parametrize(
    "tce_id,reader_output",
    [
        ("999", KeyError()),
        ("999", [CsvTestDict(target_id=1, tce_id=1, tce_test="tce", params_test=1.2)]),
    ],
    ids=["target_not_found", "target_found_tce_not_found"],
)
def test_tce_source_get__tce_not_found(tce_id, reader_output):
    """Test that DataNotFoundError is raised when no ID matching object found."""
    reader_mock = Mock(spec=Reader[CsvTestDict])
    reader_mock.read.side_effect = [reader_output]
    source = TceSource[TCETestClass](reader_mock)
    with pytest.raises(DataNotFoundError):
        source.get("1", tce_id=tce_id)


def test_tce_source_get__object_from_dict_create_error():
    """Test that DataSourceError is raised when cannot create the object from dictionary data.

    This may happen when a dictionary read via reader has different keys than class field names.
    """

    class _InvalidTestDict(TypedDict):
        x: int

    reader_mock = Mock(spec=Reader[_InvalidTestDict])
    # Return {"x": int}, but {"tce_test": str, target_id: int | str} is required
    reader_mock.read.return_value = [_InvalidTestDict(x=123)]
    source = TceSource[TCETestClass](reader_mock)
    with pytest.raises(DataSourceError):
        source.get("1", "2")


def test_tce_source_get__return_correct_data(valid_csv_reader):
    """Test that ."""
    source = TceSource[TCETestClass](valid_csv_reader)
    result = source.get("1", "1")
    assert result == TCETestClass(target_id=1, tce_id=1, tce_test="tce")
