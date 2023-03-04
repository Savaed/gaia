from unittest.mock import AsyncMock

import numpy as np
import pytest

from gaia.utils import check_kepid, json_numpy_decode, json_numpy_encode, retry


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
    foo = AsyncMock(side_effect=results)
    deco = retry(retries)(foo)
    await deco()
    assert foo.await_count == retries + 1


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
    foo = AsyncMock(side_effect=[KeyError("test error"), KeyError("test error")])
    deco = retry(1)(foo)

    with pytest.raises(KeyError, match="test error"):
        await deco()


@pytest.mark.parametrize(
    "obj",
    [
        dict(a=[1, 2.2], b="xyz", c=dict(d=1.2, e=None)),
        dict(a=[1, 2.2], b="xyz", c=np.array([1, 2])),
        dict(a=[1, 2.2], b="xyz", c=np.array([[1, 2], [3, 4]])),
    ],
    ids=["without_array", "object_with_1D_array", "object_with_2D_array"],
)
def test_numpy_decoder__encode_decode(obj):
    """Test that encoding/decoding works."""
    encoded = json_numpy_encode(obj)
    decoded = json_numpy_decode(encoded)
    assert all(
        (
            decoded["a"] == obj["a"],
            decoded["b"] == obj["b"],
            np.array_equal(decoded["c"], obj["c"]),
        ),
    )
