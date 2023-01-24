from unittest.mock import ANY, MagicMock

import pytest

from gaia.downloaders import KeplerDownloader, TableRequest
from gaia.http import ApiError
from gaia.io import FileSaver


NASA_BASE_URL = "https://www.nasa.com"
MAST_BASE_URL = "https://www.mast.com"


@pytest.fixture
def downloader():
    """Return an instance of `KeplerDownloader` with mocked `FileSaver`."""
    saver = MagicMock(spec=FileSaver)
    return KeplerDownloader(saver=saver, nasa_base_url=NASA_BASE_URL, mast_base_url=MAST_BASE_URL)


@pytest.mark.asyncio
async def test_download_tables__no_failure_on_single_table_download_error(mocker, downloader):
    """Test that a single table downloading error doesn't interrupt the entire process."""
    requests = [TableRequest("tab1"), TableRequest("tab2")]
    # Error on second tables downloading.
    mocker.patch(
        "gaia.downloaders.download",
        side_effect=[ApiError("test error", 500), b"test data"],
    )
    await downloader.download_tables(requests)


@pytest.mark.asyncio
async def test_download_tables__no_failure_on_single_table_saving_error(mocker, downloader):
    """Test that a single table saving error doesn't interrupt the entire process."""
    requests = [TableRequest("tab1"), TableRequest("tab2")]
    mocker.patch("gaia.downloaders.download", side_effect=[b"test data", b"test data"])
    # Error on second table saving.
    downloader._saver.save_table.side_effect = [Exception("test error"), None]
    await downloader.download_tables(requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table_request,url",
    [
        (TableRequest("table1"), f"{NASA_BASE_URL}?table=table1&format=csv"),
        (
            TableRequest("table1", columns={"col1", "col2"}),
            f"{NASA_BASE_URL}?table=table1&select=col1,col2&format=csv",
        ),
        (
            TableRequest("table1", query="col1 is null"),
            f"{NASA_BASE_URL}?table=table1&where=col1 is null&format=csv",
        ),
        (
            TableRequest("table1", columns={"col1", "col2"}, query="col1 is null"),
            f"{NASA_BASE_URL}?table=table1&select=col1,col2&where=col1 is null&format=csv",
        ),
    ],
    ids=[
        "table",
        "table_columns",
        "table_query",
        "table_columns_query",
    ],
)
async def test_download_tables__request_correct_url(table_request, url, mocker, downloader):
    """Test check whether ."""
    download_mock = mocker.patch("gaia.downloaders.download", return_value=b"test data")
    await downloader.download_tables([table_request])
    download_mock.assert_called_with(url, ANY)


@pytest.mark.asyncio
@pytest.mark.usefixtures("setup_logging")
async def test_download_tables__treat_http_200_error_body_as_error(mocker, caplog, downloader):
    """Test check whether ."""
    mocker.patch("gaia.downloaders.download", return_value=b"ERROR<br>")
    await downloader.download_tables([TableRequest("tab1")])
    assert "Cannot download NASA table" in caplog.text
