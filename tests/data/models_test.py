from dataclasses import asdict, dataclass

import pytest

from gaia.data.models import flatten_dict


@dataclass
class A:
    b: int


@dataclass
class C:
    d: A
    e: str


@pytest.mark.parametrize(
    "input,expected",
    [({"a": 1}, {"a": 1}), (asdict(C(A(1), e="test")), {"b": 1, "e": "test"})],
    ids=["dict", "dataclass"],
)
def test_flatten_dict__return_correct_data(input, expected):
    """Test that the correct, flat dictionary is returned with unnecessary keys omitted."""
    actual = flatten_dict(input)
    assert actual == expected
