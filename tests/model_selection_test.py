import hypothesis.strategies as st
from hypothesis import given

from gaia.model_selection import train_test_val_split


@given(st.lists(st.integers(), min_size=1))
def test_train_test_val_split__ids_only_in_single_set(ids):
    """Test that the same ids are placed only in one of the sets."""
    train, test, validation = train_test_val_split(ids)
    assert not train.intersection(test).intersection(validation)
