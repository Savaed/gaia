import asyncio
import functools
import random
from collections.abc import Awaitable, Callable, Iterator
from typing import ParamSpec, TypeAlias, TypeVar

from gaia.log import logger


def check_kepid(kepid: int) -> None:
    """Check the validity of the passed `kepid`.

    Args:
        kepid (int): Target identifier (KOI/KIC)

    Raises:
        ValueError: `kepid` is outside [1, 999 999 999]
    """
    if not 0 < kepid < 1_000_000_000:
        raise ValueError(f"'kepid' must be in range 1 to 999 999 999 inclusive, but got {kepid=}")


Errors: TypeAlias = type[Exception] | tuple[type[Exception], ...]
P = ParamSpec("P")
R = TypeVar("R")


# TODO: Allow to use without parentheses
def retry(
    retries: int = 5,
    max_seconds: int = 64,
    on: Errors | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """A decorator to retry an asynchronous function with exponential backoff.

    When retry limit is reached this re-raise the original error.
    Backoff is based on https://cloud.google.com/iot/docs/how-tos/exponential-backoff

    Args:
        retries (int, optional): How many retries. Must be at least 1. Defaults to 5.
        max_seconds (int, optional): Upper limit for backoff. Must be positive. Defaults to 64.
        on (Errors | None, optional): Error(s) on which retry. If None, retry each `Exception`
        error. Defaults to None.

    Raises:
        ValueError: `retries` less than 1
    """
    if retries < 1:
        raise ValueError(f"Expected 'retries' to be at least 1, but got {retries=}")
    if max_seconds <= 0:
        raise ValueError(f"Expected 'max_seconds' to be > 0, but got {max_seconds=}")

    errors = on or Exception

    def wrapper(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            errors_counter = 0
            while True:
                try:
                    return await fn(*args, **kwargs)
                except errors:
                    errors_counter += 1
                    backoff = min((errors_counter**2) + random.random(), max_seconds)

                    if errors_counter > retries:
                        raise

                    logger.bind(backoff=backoff).warning(
                        f"Error encountered. Retring {errors_counter}/{retries}",
                    )

                    await asyncio.sleep(backoff)

        return wrapped

    return wrapper


TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


def compose(*composable_funcs: Callable[[TInput], TOutput]) -> Callable[[TInput], TOutput]:
    return functools.reduce(lambda f, g: lambda x: g(f(x)), composable_funcs)  # type: ignore


T = TypeVar("T")


# TODO: In python 3.12 change this function to itertools.batched()
def get_chunks(collection: list[T], size: int) -> Iterator[list[T]]:
    yield from [collection[n : n + size] for n in range(0, len(collection), size)]
