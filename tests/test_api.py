"""Unit test for module `gaia.api.py`."""

import re
from typing import Any

import pytest

from gaia.api import create_nasa_url, get_mast_urls
from gaia.enums import Cadence


@pytest.fixture(name="nasa_request")
def fixture_nasa_request(request):
    data = {
        "basic": (
            "https://www.url",
            "sample_table",
            "csv",
            None,
            None,
            "https://www.url?table=sample_table&format=csv",
        ),
        "empty_url": ("", "sample_table", "csv", None, None, None),
        "empty_table": ("https://www.url", "", "csv", None, None, None),
        "empty_data_fmt": (
            "https://www.url",
            "sample_table",
            "",
            None,
            None,
            "https://www.url?table=sample_table",
        ),
        "select": (
            "https://www.url",
            "sample_table",
            "csv",
            ["x", "y"],
            None,
            "https://www.url?table=sample_table&format=csv&select=x,y",
        ),
        "query": (
            "https://www.url",
            "sample_table",
            "csv",
            None,
            "col > 1",
            "https://www.url?table=sample_table&format=csv&where=col > 1",
        ),
        "full": (
            "https://www.url",
            "sample_table",
            "csv",
            ["x", "y"],
            "col like 'abc'",
            "https://www.url?table=sample_table&format=csv&select=x,y&where=col like 'abc'",
        ),
    }
    return data[request.param]


@pytest.mark.parametrize(
    "nasa_request", ["empty_url", "empty_table", "empty_data_fmt"], indirect=True
)
def test_create_nasa_url__empty_input(nasa_request):
    """If any of inputs is empty it should raise ValueError."""
    url, table, data_fmt, _, _, _ = nasa_request

    with pytest.raises(ValueError):
        create_nasa_url(url, table, data_fmt)


@pytest.mark.parametrize("nasa_request", ["basic", "select", "query", "full"], indirect=True)
def test_create_nasa_url__valid_inputs(nasa_request):
    """If all parameters are valid it should return proper URL."""
    url, table, data_fmt, select, query, expected = nasa_request
    result = create_nasa_url(url, table, data_fmt, select, query)
    assert result == expected


@pytest.mark.parametrize("url", [None, ""])
def test_get_mast_urls__empty_input(url):
    """If `base_url` is empty then it should raise ValueError."""
    with pytest.raises(ValueError):
        get_mast_urls(url, 1, None)


@pytest.mark.parametrize("kepid", [-999, 0, 1_000_000_000])
def test_get_mast_urls__invalid_kepid(kepid: int):
    """If `kepid` is less than 0 or greater than 999 999 999 then it should raise ValueError."""
    with pytest.raises(ValueError):
        get_mast_urls("base_url", kepid, None)


@pytest.mark.parametrize("cadence", [Cadence.LONG, Cadence.SHORT])
def test_get_mast_urls__valid_inputs(cadence: Cadence):
    """If inputs are valid then it should return proper urls."""
    url = "https://mast.stsci.edu/api/v0.1/Download/file"
    kepid = 123
    fmt_kepid = f"{kepid:09}"
    mast_url_pattern = re.compile(
        f"{url}\?uri=mast:Kepler/url/missions/kepler/lightcurves/{fmt_kepid[:4]}/{fmt_kepid}//kplr{fmt_kepid}-\d{{13}}_{cadence.value}\.fits"
    )

    for url in get_mast_urls(url, kepid, cadence):
        assert mast_url_pattern.match(url)
