import itertools
from unittest import mock

import hypothesis.strategies as st
from hypothesis import given

from gaia.enums import Cadence
from gaia.quarters import get_quarter_prefixes


quarters_prefixes = st.lists(st.integers().map(str), min_size=1)


@given(
    st.tuples(quarters_prefixes) | st.tuples(quarters_prefixes, quarters_prefixes),
    st.sampled_from(Cadence),
)
def test_function__case(prefixes: tuple[list[str]], cadence: Cadence) -> None:
    with mock.patch("gaia.quarters.LONG_CADENCE_QUARTER_PREFIXES", prefixes), mock.patch(
        "gaia.quarters.SHORT_CADENCE_QUARTER_PREFIXES",
        prefixes,
    ):
        actual = get_quarter_prefixes(cadence)
    expected = tuple(itertools.chain.from_iterable(prefixes))
    assert actual == expected
