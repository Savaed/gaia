from unittest.mock import AsyncMock

import pytest

from gaia.utils import check_kepid, retry


@pytest.mark.parametrize("kepid", [-1, 0, 1_000_000_000])
def test_check_kepid__invalid_input(kepid):
    """Test that ValueError is raised when kepid is outside [1, 999 999 999]."""
    with pytest.raises(ValueError):
        check_kepid(kepid)


def test_check_kepid__valid_input():
    """Test that no error is raised when kepid is inside [1, 999 999 999]."""
    check_kepid(123)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "n,results",
    [(1, [Exception(), "ok"]), (2, [Exception(), Exception(), "ok"])],
    ids=[1, 2],
)
async def test_retry__retrying_specified_times(n, results, mocker):
    """Test check whether retries specified number of times."""
    mocker.patch("gaia.utils.asyncio.sleep")
    foo = AsyncMock(side_effect=results)
    deco = retry(n)(foo)
    await deco()
    assert foo.await_count == n + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("retries", [-1, 0])
async def test_retry__retries_number_less_than_1(retries):
    """Test that ValueError is raised when specified retries number is less than 1."""
    with pytest.raises(ValueError, match="'retries' must be at least 1"):

        @retry(retries)
        async def _():
            ...
