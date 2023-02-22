from dataclasses import dataclass

import pytest

from gaia.data.models import FromDictMixin


@dataclass
class FromDictTestClass(FromDictMixin):
    a: int
    b: int
    c: int


def test_from_dict__class_is_not_dataclass():
    """Test that TypeError is raised when `from_dict()` is call on non dataclass."""

    class _FromDictTestClass(FromDictMixin):
        a: int

    with pytest.raises(TypeError):
        _FromDictTestClass.from_flat_dict({"a": 1, "b": 2})


@pytest.mark.parametrize(
    "data,mapping",
    [
        ({"a": 1, "b": 2, "d": 3}, None),
        ({"A": 1, "B": 2, "C": 3}, {"A": "a", "B": "b"}),
    ],
    ids=["no_mapping", "invalid_mapping"],
)
def test_from_dict__missing_data(data, mapping):
    """Test that KeyError is raised when some fields are missing in the dictionary data."""
    with pytest.raises(KeyError):
        FromDictTestClass.from_flat_dict(data, mapping=mapping)


@pytest.mark.parametrize(
    "data,expected",
    [
        ({"a": 1, "b": 2, "c": 3}, FromDictTestClass(a=1, b=2, c=3)),
        ({"a": 1, "b": 2, "c": 3, "d": 4}, FromDictTestClass(a=1, b=2, c=3)),
    ],
    ids=["standard", "additional_keys"],
)
def test_from_dict__return_correct_data_without_mappping(data, expected):
    """Test that properly mapped object is returned without dictionary keys mapping."""
    result = FromDictTestClass.from_flat_dict(data)
    assert result == expected


@pytest.mark.parametrize(
    "data,mapping,expected",
    [
        (
            {"A": 1, "B": 2, "C": 3},
            {"A": "a", "B": "b", "C": "c"},
            FromDictTestClass(a=1, b=2, c=3),
        ),
        (
            {"A": 1, "B": 2, "C": 3, "D": 4},
            {"A": "a", "B": "b", "C": "c", "D": "d"},
            FromDictTestClass(a=1, b=2, c=3),
        ),
        (
            {"A": 1, "B": 2, "C": 3},
            {"A": "a", "B": "b", "C": "c", "D": "d"},
            FromDictTestClass(a=1, b=2, c=3),
        ),
    ],
    ids=["standard", "additional_keys", "additional_mapping"],
)
def test_from_dict__return_correct_data_with_mappping(data, mapping, expected):
    """Test that properly mapped object is returned with dictionary keys mapping."""
    result = FromDictTestClass.from_flat_dict(data, mapping=mapping)
    assert result == expected
