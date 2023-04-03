from dataclasses import dataclass
from typing import TypedDict
from unittest.mock import Mock

import numpy as np
import pytest

from gaia.data.models import TCE, PeriodicEvent, StellarParameters
from gaia.data.sources import (
    DataNotFoundError,
    DataSourceError,
    MissingPeriodsError,
    StellarParametersSource,
    TceSource,
    TimeSeriesSource,
)
from gaia.io import CsvTableReader, ReaderError, TimeSeriesReader
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
    """Mock an instance of `TimeSeriesReader` to return `TEST_TIME_SERIES_DICT` and return it."""
    reader = Mock(spec=TimeSeriesReader[dict[str, TimeSeriesTestDict]])
    reader.read.return_value = TEST_TIME_SERIES_DICT
    return reader


@pytest.fixture
def error_time_series_reader():
    """Mock an instance of `TimeSeriesReader` to raise `ReaderError` and return it."""
    reader = Mock(spec=TimeSeriesReader[dict[str, TimeSeriesTestDict]])
    reader.read.side_effect = ReaderError("Cannot read underlying data")
    return reader


@pytest.fixture
def id_not_found_time_series_reader():
    """Mock an instance of `TimeSeriesReader` to return `KeyError(999)` and return it."""
    reader = Mock(spec=TimeSeriesReader[dict[str, TimeSeriesTestDict]])
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
    "periods,expected",
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
        "all_by_specified_some_periods_repeat",
    ],
)
def test_time_series_source_get__get_specified_periods(periods, expected, time_series_reader):
    """Test that only specified time series periods are returned."""
    source = TimeSeriesSource[TimeSeriesTestDict](time_series_reader)
    result = source.get("1", periods=periods)
    assert result == expected


@pytest.mark.parametrize(
    "periods",
    [
        ["quarter99", "quarter999"],
        ["quarter1", "quarter999"],
    ],
    ids=["all_are_missing", "one_is_missing"],
)
def test_time_series_source_get__missing_periods(periods, time_series_reader):
    """Test that MissingDataPartsError is raised when some of specifed periods are missing."""
    source = TimeSeriesSource[TimeSeriesTestDict](time_series_reader)
    with pytest.raises(MissingPeriodsError):
        source.get("1", periods=periods)


@pytest.mark.parametrize(
    "periods,expected",
    [
        (["quarter1"], {"a": np.array([1]), "b": np.array([2])}),
        (["quarter1", "quarter2"], {"a": np.array([1, 3]), "b": np.array([2, 4])}),
    ],
    ids=["one_period", "multiple_periods"],
)
def test_time_series_source_get__squeeze_output(periods, expected, time_series_reader):
    """Test that time series for each period are combined together."""
    source = TimeSeriesSource[TimeSeriesTestDict](time_series_reader)
    result = source.get("1", periods=periods, squeeze=True)
    assert_dict_with_numpy_equal(result["all"], expected)  # type: ignore


class CsvTestDict(TypedDict):
    target_id: int
    tce_id: int
    name: str
    params_test: float
    label: str


@dataclass(unsafe_hash=True)
class TCETestClass(TCE):
    @property
    def event(self) -> PeriodicEvent:  # pragma: no cover
        return PeriodicEvent(1, 2, 3)


@dataclass
class StellarParamsTestClass(StellarParameters):
    params_test: float


TEST_CSV_DICT = CsvTestDict(target_id=1, tce_id=1, name="tce", params_test=1.2, label="test_label")


@pytest.fixture
def valid_csv_reader():
    """Mock an instance of `CsvTableReader` to return `[TEST_CSV_DICT]` and return it."""
    reader = Mock(spec=CsvTableReader)
    reader.read.return_value = [TEST_CSV_DICT]
    return reader


@pytest.fixture
def error_csv_reader():
    """Mock an instance of `CsvTableReader` to raise `ReaderError` and return it."""
    reader = Mock(spec=CsvTableReader)
    reader.read.side_effect = ReaderError("Cannot read source CSV test file")
    return reader


def test_stellar_parameters_source_get__reader_error(error_csv_reader):
    """Test that DataSourceError is raised when there is any error with reading the data."""
    source = StellarParametersSource[StellarParamsTestClass](error_csv_reader)
    with pytest.raises(DataSourceError):
        source.get(1)


def test_stellar_parameters_source_get__stellar_params_not_found(valid_csv_reader):
    """Test that DataNotFoundError is raised when no ID matching object found."""
    source = StellarParametersSource[StellarParamsTestClass](valid_csv_reader)
    with pytest.raises(DataNotFoundError):
        source.get(999)


def test_stellar_parameters_source_get__object_from_dict_create_error():
    """Test that DataSourceError is raised when cannot create the object from dictionary data.

    This may happen when a dictionary read via reader has different keys than class field names.
    """
    reader_mock = Mock(spec=CsvTableReader)
    # No key 'params_test' required for StellarParamsTestClass
    reader_mock.read.return_value = [dict(target_id=1)]
    source = StellarParametersSource[StellarParamsTestClass](reader_mock)
    with pytest.raises(DataSourceError):
        source.get(1)


def test_stellar_parameters_source_get__return_correct_data(valid_csv_reader):
    """Test that correct data is returned."""
    target_id = 1
    source = StellarParametersSource[StellarParamsTestClass](valid_csv_reader)
    result = source.get(target_id)
    assert result == StellarParamsTestClass(target_id=target_id, params_test=1.2)


def test_tce_source_get_all_for_target__reader_error(error_csv_reader):
    """Test that DataSourceError is raised when there is any internal reader error."""
    source = TceSource[TCETestClass](error_csv_reader)
    with pytest.raises(DataSourceError):
        source.get_all_for_target(1)


def test_tce_source_get_all_for_target__target_not_found(valid_csv_reader):
    """Test that DataNotFoundError is raised when no ID matching object found."""
    source = TceSource[TCETestClass](valid_csv_reader)
    with pytest.raises(DataNotFoundError):
        source.get_all_for_target(999)


def test_tce_source_get_all_for_target__object_from_dict_create_error():
    """Test that DataSourceError is raised when cannot create the object from dictionary data.

    This may happen when a dictionary read via reader has different keys than class field names.
    """
    reader_mock = Mock(spec=CsvTableReader)
    # No key 'name' required for TCETestClass
    reader_mock.read.return_value = [dict(target_id=1, tce_id=1)]
    source = TceSource[TCETestClass](reader_mock)
    with pytest.raises(DataSourceError):
        source.get_all_for_target(1)


@pytest.mark.parametrize(
    "reader_output,expected",
    [
        (
            [CsvTestDict(target_id=1, tce_id=1, name="tce", params_test=1.2, label="test_label")],
            [TCETestClass(target_id=1, tce_id=1, name="tce", label="test_label")],
        ),
        (
            [
                CsvTestDict(target_id=1, tce_id=1, name="tce", params_test=1.2, label="test_label"),
                CsvTestDict(
                    target_id=1,
                    tce_id=2,
                    name="tce2",
                    params_test=1.2,
                    label="test_label",
                ),
            ],
            [
                TCETestClass(target_id=1, tce_id=1, name="tce", label="test_label"),
                TCETestClass(target_id=1, tce_id=2, name="tce2", label="test_label"),
            ],
        ),
    ],
    ids=["one_tce", "many_tces"],
)
def test_tce_source_get_all_for_target__return_correct_data(reader_output, expected):
    """Test that correct TCEs are returned."""
    reader_mock = Mock(spec=CsvTableReader)
    reader_mock.read.return_value = reader_output
    source = TceSource[TCETestClass](reader_mock)
    result = source.get_all_for_target(1)
    assert result == expected


def test_tce_source_get_by_id__reader_error(error_csv_reader):
    """Test that DataSourceError is raised when there is any internal reader error."""
    source = TceSource[TCETestClass](error_csv_reader)
    with pytest.raises(DataSourceError):
        source.get_by_id(1, 2)


@pytest.mark.parametrize(
    "target_id,tce_id",
    [(999, 1), (1, 999)],
    ids=["target_not_found", "target_found_tce_not_found"],
)
def test_tce_source_get_by_id__tce_not_found(target_id, tce_id, valid_csv_reader):
    source = TceSource[TCETestClass](valid_csv_reader)
    with pytest.raises(DataNotFoundError):
        source.get_by_id(target_id, tce_id)


def test_tce_source_get_by_id__object_from_dict_create_error():
    """Test that DataSourceError is raised when cannot create the object from dictionary data.

    This may happen when a dictionary read via reader has different keys than class field names.
    """
    reader_mock = Mock(spec=CsvTableReader)
    # No key 'name' required for TCETestClass
    reader_mock.read.return_value = [dict(target_id=1, tce_id=1)]
    source = TceSource[TCETestClass](reader_mock)
    with pytest.raises(DataSourceError):
        source.get_by_id(1, 1)


def test_tce_source_get_by_id__return_correct_data(valid_csv_reader):
    """Test that correct TCE are returned."""
    source = TceSource[TCETestClass](valid_csv_reader)
    result = source.get_by_id(1, 1)
    assert result == TCETestClass(target_id=1, tce_id=1, name="tce", label="test_label")


def test_tce_source_get_by_name__reader_error(error_csv_reader):
    """Test that DataSourceError is raised when there is any internal reader error."""
    source = TceSource[TCETestClass](error_csv_reader)
    with pytest.raises(DataSourceError):
        source.get_by_name("tce")


def test_tce_source_get_by_name__tce_not_found(valid_csv_reader):
    source = TceSource[TCETestClass](valid_csv_reader)
    with pytest.raises(DataNotFoundError):
        source.get_by_name("sdfs")


def test_tce_source_get_by_name__object_from_dict_create_error():
    """Test that DataSourceError is raised when cannot create the object from dictionary data.

    This may happen when a dictionary read via reader has different keys than class field names.
    """
    reader_mock = Mock(spec=CsvTableReader)
    # No key 'name' required for TCETestClass
    reader_mock.read.return_value = [dict(target_id=1, tce_id=1)]
    source = TceSource[TCETestClass](reader_mock)
    with pytest.raises(DataSourceError):
        source.get_by_name("tce")


def test_tce_source_get_by_name__return_correct_data(valid_csv_reader):
    """Test that correct TCE are returned."""
    source = TceSource[TCETestClass](valid_csv_reader)
    result = source.get_by_name("tce")
    assert result == TCETestClass(target_id=1, tce_id=1, name="tce", label="test_label")


@pytest.fixture(params=["single_target", "two_targets", "empty"])
def fixture_name(request):
    """Return a list of two dicts with two TCEs for one or two targets."""
    if request.param == "single_target":
        return (
            [
                CsvTestDict(
                    target_id=1,
                    tce_id=1,
                    name="target1_tce1",
                    params_test=0.1,
                    label="test_label",
                ),
                CsvTestDict(
                    target_id=1,
                    tce_id=2,
                    name="target1_tce2",
                    params_test=0.2,
                    label="test_label",
                ),
            ],
            {1},
            2,
        )

    if request.param == "two_targets":
        return (
            [
                CsvTestDict(
                    target_id=1,
                    tce_id=1,
                    name="target1_tce1",
                    params_test=0.1,
                    label="test_label",
                ),
                CsvTestDict(
                    target_id=2,
                    tce_id=2,
                    name="target2_tce2",
                    params_test=0.2,
                    label="test_label",
                ),
            ],
            {1, 2},
            2,
        )

    return [], set(), 0


def test_tce_source_tce_count__return_correct_count(fixture_name):
    """Test that the correct count of TCEs is returned."""
    reader_mock = Mock(CsvTableReader)
    dct, _, expected = fixture_name
    reader_mock.read.return_value = dct
    tce_count = TceSource[TCETestClass](reader_mock).tce_count
    assert tce_count == expected


def test_tce_source_target_unique_ids__return_correct_count(fixture_name):
    """Test that the correct unique targets IDs are returned."""
    reader_mock = Mock(CsvTableReader)
    dct, expected, _ = fixture_name
    reader_mock.read.return_value = dct
    target_count = TceSource[TCETestClass](reader_mock).target_unique_ids
    assert target_count == expected
