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
    "retries,results",
    [(1, [Exception(), "ok"]), (2, [Exception(), Exception(), "ok"])],
    ids=[1, 2],
)
async def test_retry__retrying_specified_times(retries, results, mocker):
    """Test that function is retrying the specified number of times."""
    mocker.patch("gaia.utils.asyncio.sleep")
    fn_mock = AsyncMock(side_effect=results)
    decorated_fn = retry(retries)(fn_mock)
    await decorated_fn()
    assert fn_mock.await_count == retries + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("retries", [-1, 0])
async def test_retry__retries_number_less_than_1(retries):
    """Test that ValueError is raised when retries number is less than 1."""
    with pytest.raises(ValueError, match="'retries' must be at least 1"):

        @retry(retries)
        async def _():  # pragma: no cover
            ...


@pytest.mark.asyncio
async def test_retry__raise_on_retries_limit(mocker):
    """Test that error is raised when the retry limit is reached."""
    mocker.patch("gaia.utils.asyncio.sleep")
    fn_mock = AsyncMock(side_effect=[KeyError("test error"), KeyError("test error")])
    decorated_fn = retry(1)(fn_mock)

    with pytest.raises(KeyError, match="test error"):
        await decorated_fn()  # Should raise due to 2 errors and limit set to 1
