"""Unit tests for `gaia.data.sources` module."""

from unittest.mock import Mock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from gaia.data.sources import FITSKeplerTimeSeriesSource, InvalidColumnError, MissingKoiError
from gaia.enums import Cadence
from gaia.io import FITSTimeSeriesReader


class TestFITSKeplerTimeSeriesSource:
    """Unit tests for `FITSKeplerTimeSeriesSource` class."""

    @pytest.fixture
    def time_series(self) -> dict[str, np.ndarray]:
        """Return time series data read from FITS file(s)."""
        return {"A": np.array([1, 2, 3, 4, 5]), "B": np.array([6, 7, 8, 9, 10])}

    def test_get__missing_koi(self, mocker: MockerFixture) -> None:
        """Test check whether MissingKoiError is raised when no data found for `kepid`."""
        mocker.patch("gaia.data.sources.get_kepler_fits_paths", return_value=["path1"])
        mock_read = Mock(spec=FITSTimeSeriesReader, **{"read.side_effect": FileNotFoundError})
        data_source = FITSKeplerTimeSeriesSource(mock_read, "test/dir", Cadence.LONG)

        with pytest.raises(MissingKoiError):
            data_source.get(123, "column1")

    @pytest.fixture
    def missing_column_time_series(
        self, time_series: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], str]:
        """Return time series and one column which is not included in the time series."""
        missing_column = "".join(time_series)
        return time_series, missing_column

    def test_get__invalid_fits_column(
        self, missing_column_time_series: tuple[dict[str, np.ndarray], str], mocker: MockerFixture
    ) -> None:
        """Test check whether InvalidColumnError is raised when no specified column is found in FITS file."""
        time_series, missing_column = missing_column_time_series
        mocker.patch("gaia.data.sources.get_kepler_fits_paths", return_value=["path1"])
        mock_read = Mock(spec=FITSTimeSeriesReader, **{"read.return_value": time_series})
        data_source = FITSKeplerTimeSeriesSource(mock_read, "test/dir", Cadence.LONG)

        with pytest.raises(InvalidColumnError):
            data_source.get(132, missing_column)

    def test_get__return_correct_time_series(
        self, time_series: dict[str, np.ndarray], mocker: MockerFixture
    ) -> None:
        """Test check whether a correct time series is returned."""
        mocker.patch("gaia.data.sources.get_kepler_fits_paths", return_value=["path1"])
        mock_read = Mock(spec=FITSTimeSeriesReader, **{"read.return_value": time_series})
        time_column, field = time_series
        data_source = FITSKeplerTimeSeriesSource(
            mock_read, "test/dir", Cadence.LONG, time_field=time_column
        )

        result = data_source.get(123, field)

        assert result.time == [time_series[time_column]] and result.values == [time_series[field]]

    def test_get_for_quarters__missing_koi(self, mocker: MockerFixture) -> None:
        """Test check whether MissingKoiError is raised when no data found for `kepid`."""
        mocker.patch("gaia.data.sources.get_kepler_fits_paths", return_value=["path1"])
        mock_read = Mock(spec=FITSTimeSeriesReader, **{"read.side_effect": FileNotFoundError})
        data_source = FITSKeplerTimeSeriesSource(mock_read, "test/dir", Cadence.LONG)

        with pytest.raises(MissingKoiError):
            data_source.get_for_quarters(123, "column1", ("1", "2"))

    def test_get_for_quarters__invalid_fits_column(
        self, missing_column_time_series: tuple[dict[str, np.ndarray], str], mocker: MockerFixture
    ) -> None:
        """
        Test check whether InvalidColumnError is raised when no required column is found.
        """
        time_series, missing_column = missing_column_time_series
        mocker.patch("gaia.data.sources.get_kepler_fits_paths", return_value=["path1"])
        mock_read = Mock(spec=FITSTimeSeriesReader, **{"read.return_value": time_series})
        data_source = FITSKeplerTimeSeriesSource(mock_read, "test/dir", Cadence.LONG)

        with pytest.raises(InvalidColumnError):
            data_source.get_for_quarters(132, missing_column, ("1", "2"))
