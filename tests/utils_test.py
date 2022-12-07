import pytest

from gaia.utils import check_kepid


@pytest.mark.parametrize("kepid", [-1, 0, 1_000_000_000])
def test_check_kepid__invalid_input(kepid):
    with pytest.raises(ValueError):
        check_kepid(kepid)


def test_check_kepid__valid_input():
    check_kepid(123)
