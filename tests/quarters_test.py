import pytest

from gaia import quarters
from gaia.enums import Cadence
from gaia.quarters import get_quarter_prefixes


@pytest.fixture(
    params=[
        ((["1"]), ("1",)),
        ((["1", "2"]), ("1", "2")),
        ((["1"], ["2"]), ("1", "2")),
        ((["1"], ["2", "3"]), ("1", "2", "3")),
        ((["1", "2"], ["3", "4"]), ("1", "2", "3", "4")),
    ],
    ids=[
        "one_prefix",
        "two_prefixes",
        "two_lists_single_prefix",
        "two_lists_mixed",
        "two_lists_two_prefixes",
    ],
)
def prefixes(request):
    """Return quarter prefixes and expected prefixes."""
    return request.param


@pytest.mark.parametrize(
    "cadence,const_name",
    [
        (Cadence.LONG, "LONG_CADENCE_QUARTER_PREFIXES"),
        (Cadence.SHORT, "SHORT_CADENCE_QUARTER_PREFIXES"),
    ],
    ids=["cadence_long", "cadence_short"],
)
def test_get_quarter_prefixes(cadence, const_name, prefixes, monkeypatch):
    """Test check whether all quarter prefixes are returned."""
    prefs, expected = prefixes
    monkeypatch.setattr(quarters, const_name, prefs)
    result = get_quarter_prefixes(cadence)
    assert result == expected
