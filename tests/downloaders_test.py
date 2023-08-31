import asyncio
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from gaia.downloaders import KeplerDownloader
from gaia.enums import Cadence
from gaia.http import ApiError
from gaia.io import FileSaver


NASA_BASE_URL = "https://www.nasa.com"
MAST_BASE_URL = "https://www.mast.com"


@pytest.fixture
def downloader(tmp_path: Path) -> KeplerDownloader:
    """Return a test instance of `KeplerDownloader` class.

    The object has mocked `FileSaver`, test metadata path and `Cadence.LONG` as cadence.
    """
    saver = MagicMock(spec=FileSaver)
    instance = KeplerDownloader(
        saver=saver,
        mast_base_url=MAST_BASE_URL,
        cadence=Cadence.LONG,
    )
    # Make sure that metadata is temporal for each test
    instance._checkpoint_filepath = tmp_path / "test_meta.txt"
    return instance


@pytest.mark.asyncio
async def test_download_tables__no_failure_on_single_table_download_error(
    downloader: KeplerDownloader,
) -> None:
    """Test that a single table downloading error does not interrupt the entire process."""
    requests = [
        (f"{NASA_BASE_URL}?table=tab1", "tab1.csv"),
        (f"{NASA_BASE_URL}?table=tab1", "tab1.csv"),
    ]
    # Error on second tables downloading
    with patch(
        "gaia.downloaders.download",
        side_effect=[ApiError("test error", 500), b"test data"],
    ):
        await downloader.download_tables(requests)


@pytest.mark.asyncio
async def test_download_tables__no_failure_on_single_table_saving_error(
    downloader: KeplerDownloader,
) -> None:
    """Test that a single table saving error does not interrupt the entire process."""
    requests = [
        (f"{NASA_BASE_URL}?table=tab1", "tab1.csv"),
        (f"{NASA_BASE_URL}?table=tab1", "tab1.csv"),
    ]
    downloader._saver.save_table.side_effect = [Exception("test error"), None]  # type: ignore
    with patch("gaia.downloaders.download", side_effect=[b"test data", b"test data"]):
        # Error on second table saving
        await downloader.download_tables(requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table_request",
    [
        (f"{NASA_BASE_URL}?table=table1&format=csv", ""),
        (f"{NASA_BASE_URL}?table=table1&select=col1,col2&format=csv", ""),
        (f"{NASA_BASE_URL}?table=table1&where=col1 is null&format=csv", ""),
        (
            f"{NASA_BASE_URL}?table=table1&select=col1,col2&where=col1 is null&format=csv",
            "",
        ),
    ],
    ids=[
        "table",
        "table_columns",
        "table_query",
        "table_columns_query",
    ],
)
async def test_download_tables__request_correct_url(
    table_request: tuple[str, str],
    downloader: KeplerDownloader,
) -> None:
    """Test that download function is called with the correct URL."""
    url, _ = table_request
    with patch("gaia.downloaders.download", return_value=b"test data") as download_mock:
        await downloader.download_tables([table_request])
        download_mock.assert_called_with(url, ANY)


@pytest.mark.asyncio
async def test_download_tables__treat_http_200_error_body_as_error(
    downloader: KeplerDownloader,
) -> None:
    """Test that the response beginning with 'ERROR<br>' is treated as an error and not saved."""
    with patch("gaia.downloaders.download", return_value=b"ERROR<br>"):
        await downloader.download_tables(
            [(f"{NASA_BASE_URL}?table=table1&format=csv", "table1.csv")],
        )
        downloader._saver.save_table.assert_not_called()  # type: ignore


URL_A_PREF = "https://www.mast.com/0000/000000001//kplr000000001-a_pref_llc.fits"
URL_B_PREF = "https://www.mast.com/0000/000000001//kplr000000001-b_pref_llc.fits"


@pytest.fixture
def downloader_with_test_meta(downloader: KeplerDownloader) -> KeplerDownloader:
    """
    Return `downloader` fixture result but with test URL
    https://www.mast.com/0000/000000001//kplr000000001-a_pref_llc.fits saved in the metadata file.
    """
    downloaded_url = f"{URL_A_PREF}\n"
    downloader._checkpoint_filepath.write_text(downloaded_url)
    return downloader


TEST_ID = (1,)


@pytest.mark.asyncio
async def test_download_time_series__skip_already_downloaded_files(
    downloader_with_test_meta: KeplerDownloader,
) -> None:
    """Test that already downloaded FITS files are not being re-downloaded."""
    with (
        patch("gaia.downloaders.get_quarter_prefixes", return_value=("a_pref", "b_pref")),
        patch("gaia.downloaders.download", side_effect=[b"test", b"test"]) as download_mock,
    ):
        await downloader_with_test_meta.download_time_series(TEST_ID)
        download_mock.assert_called_once_with(URL_B_PREF, ANY)


@pytest.mark.asyncio
async def test_download_time_series__skip_missing_file(
    downloader_with_test_meta: KeplerDownloader,
) -> None:
    """Test that missing FITS files are skipped (no errors raised and data saved)."""
    with patch("gaia.downloaders.get_quarter_prefixes", return_value=("a_pref", "b_pref")), patch(
        "gaia.downloaders.download",
        return_value=b"test",
    ):
        await downloader_with_test_meta.download_time_series(TEST_ID)
        downloader_with_test_meta._saver.save_time_series.assert_called_once_with(  # type: ignore
            "kplr000000001-b_pref_llc.fits",
            b"test",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cadence", [Cadence.LONG, Cadence.SHORT])
async def test_download_time_series__request_correct_url(
    cadence: Cadence,
    downloader: KeplerDownloader,
) -> None:
    """Test that download function is called with the correct URL."""
    downloader._cadence = cadence
    with (
        patch("gaia.downloaders.get_quarter_prefixes", return_value=("a_pref", "b_pref")),
        patch("gaia.downloaders.download", side_effect=[b"test", b"test"]) as download_mock,
    ):
        download_calls = [
            call(
                f"https://www.mast.com/0000/000000001//kplr000000001-a_pref_{cadence.value}.fits",
                ANY,
            ),
            call(
                f"https://www.mast.com/0000/000000001//kplr000000001-b_pref_{cadence.value}.fits",
                ANY,
            ),
        ]
        await downloader.download_time_series(TEST_ID)
        download_mock.assert_has_calls(download_calls, any_order=True)


@pytest.mark.asyncio
async def test_download_time_series__save_downloaded_files(downloader: KeplerDownloader) -> None:
    """Test that downloaded FITS files are saved (saving method called with correct args)."""
    test_data = b"test"
    save_calls = [
        call("kplr000000001-a_pref_llc.fits", test_data),
        call("kplr000000001-b_pref_llc.fits", test_data),
    ]
    with patch("gaia.downloaders.get_quarter_prefixes", return_value=("a_pref", "b_pref")), patch(
        "gaia.downloaders.download",
        side_effect=[test_data, test_data],
    ):
        await downloader.download_time_series(TEST_ID)
        downloader._saver.save_time_series.assert_has_calls(save_calls, any_order=True)  # type: ignore # noqa


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "http_responses",
    [
        (b"test", b"test"),
        (ApiError("missing file", 404), ApiError("missing file", 404)),
        (ApiError("missing file", 404), b"test"),
        (b"test", ApiError("missing file", 404)),
    ],
    ids=["two_files", "two_missing_files", "missing_and_file", "file_and_missing"],
)
async def test_download_time_series__save_meta(
    http_responses: ApiError | bytes,
    downloader: KeplerDownloader,
) -> None:
    """Test that URLs are properly saved to the metadata file."""
    expected = set([URL_A_PREF, URL_B_PREF])
    with patch("gaia.downloaders.get_quarter_prefixes", return_value=("a_pref", "b_pref")), patch(
        "gaia.downloaders.download",
        side_effect=http_responses,
    ):
        await downloader.download_time_series(TEST_ID)
        saved_meta = set(downloader._checkpoint_filepath.read_text().splitlines())
        assert saved_meta == expected


@pytest.fixture
def disable_retry_sleep():
    """Mock `asyncio.sleep` in `gaia.utils module`."""
    with patch("gaia.utils.asyncio.sleep"):
        yield


@pytest.mark.asyncio
@pytest.mark.usefixtures("disable_retry_sleep")
async def test_download_time_series__retry_failed_requests(downloader: KeplerDownloader) -> None:
    """
    Test that the method doesn't break when a single download fails but the download is retry the
    specified number of times.
    """
    # NOTE: This test asumes that retries number == 5 (default) and retry on ApiError for _fetch()
    download_results = [
        ApiError("test error", 503),
        ApiError("test error", 503),
        asyncio.TimeoutError(),
        asyncio.TimeoutError(),
        ApiError("test error", 503),
        b"ok",
    ]
    # 1 normal call + 5 retries
    with (
        patch("gaia.downloaders.get_quarter_prefixes", return_value=("a_pref",)),
        patch("gaia.downloaders.download", side_effect=download_results) as download_mock,
    ):
        await downloader.download_time_series(TEST_ID)
        assert download_mock.await_count == 6


@pytest.mark.asyncio
@pytest.mark.usefixtures("disable_retry_sleep")
async def test_download_time_series__raise_when_ratries_limit_reached(
    downloader: KeplerDownloader,
) -> None:
    """Test that an error is reported after exceeding the download retry limit."""
    # NOTE: This test asumes that retries number == 5 (default) and retry on ApiError for _fetch()
    # 1 normal call + 5 retries end up with an error
    download_results = [ApiError("test error", 503)] * 6
    with patch("gaia.downloaders.get_quarter_prefixes", return_value=("a_pref",)), patch(
        "gaia.downloaders.download",
        side_effect=download_results,
    ):
        with pytest.raises(ApiError, match="test error"):
            await downloader.download_time_series(TEST_ID)


@pytest.mark.asyncio
async def test_download_time_series__raise_on_file_saving_error(
    downloader: KeplerDownloader,
) -> None:
    """Test that any error raised by time series saving method is re-raised to the caller."""
    saving_error_msg = "saving time series file error"
    with patch("gaia.downloaders.get_quarter_prefixes", return_value=("a_pref",)), patch(
        "gaia.downloaders.download",
        side_effect=b"ok",
    ):
        downloader._saver.save_time_series.side_effect = PermissionError(saving_error_msg)  # type: ignore # noqa
        with pytest.raises(PermissionError, match=saving_error_msg):
            await downloader.download_time_series(TEST_ID)


@pytest.mark.asyncio
async def test_download_time_series__return_immiedietly_if_no_urls(
    downloader_with_test_meta: KeplerDownloader,
) -> None:
    """Test that no download function is called when there is no URLs."""
    with (
        patch("gaia.downloaders.get_quarter_prefixes", return_value=()),
        patch("gaia.downloaders.download") as http_download_mock,
    ):
        await downloader_with_test_meta.download_time_series(TEST_ID)
        http_download_mock.assert_not_called()
