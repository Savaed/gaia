import functools
from operator import add
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from gaia.utils import check_kepid, compose, retry


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
async def test_retry__retrying_specified_times(retries, results):
    """Test that function is retrying the specified number of times."""
    with mock.patch("gaia.utils.asyncio.sleep"):
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
async def test_retry__raise_on_retries_limit():
    """Test that error is raised when the retry limit is reached."""
    with mock.patch("gaia.utils.asyncio.sleep"):
        fn_mock = AsyncMock(side_effect=[Exception, Exception])
        decorated_fn = retry(1)(fn_mock)

        with pytest.raises(Exception):
            await decorated_fn()  # Should raise due to 2 errors and limit set to 1


def test_compose__one_of_functions_raises_error():
    """Test that an expection raise in one of functions is not swollow by composed function."""

    def foo(_):
        raise ValueError

    compose_func = compose(lambda x: x, foo)  # type: ignore
    with pytest.raises(ValueError):
        compose_func(1)


def test_compose__return_correct_data_get_parameters():
    """Test that functions that get parameters work properly when composed."""
    add_func = functools.partial(add, 1)
    composed_func = compose(add_func, lambda x: x * x)  # type: ignore
    actual = composed_func(1)
    assert actual == 4


def test_compose__print_correct_text(capsys):
    """Test that a composed function handles the side effects of its functions correctly."""
    composed_func = compose(lambda _: print("hello"), lambda _: print("world"))  # type: ignore
    composed_func(None)
    actual = capsys.readouterr().out
    assert actual == "hello\nworld\n"
