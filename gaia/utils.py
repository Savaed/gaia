import asyncio
import functools
import random
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeAlias, TypeVar

import structlog


log = structlog.stdlib.get_logger()


def check_kepid(kepid: int) -> None:
    """Check the validity of the passed `kepid`.

    Args:
        kepid (int): Target identifier (KOI/KIC)

    Raises:
        ValueError: `kepid` is outside [1, 999 999 999]
    """
    if not 0 < kepid < 1_000_000_000:
        raise ValueError(f"'kepid' must be in range 1 to 999 999 999 inclusive, but got {kepid=}")


# HACK: To properly mark an Exception as an argument type, `Type[Exception]` is necessary,
# but pyupgrade rewrites it as `type[Exception]`, which causes mypy error:
# 'Value of type 'Type [type]' is not indexable [index]'
Errors: TypeAlias = type[Exception] | tuple[type[Exception], ...]
P = ParamSpec("P")
R = TypeVar("R")


# TODO: Testing decorated functions is a bit tricky. Search for a better solution
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
        max_seconds (int, optional): Upper limit for backoff. Defaults to 64.
        on (Errors | None, optional): Error(s) on which retry. If None, retry each `Exception`
            error. Defaults to None.

    Raises:
        ValueError: `retries` less than 1
    """
    if retries < 1:
        raise ValueError(f"'retries' must be at least 1, but got {retries=}")

    errors = on or Exception

    def wrapper(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            n = 0
            while True:
                try:
                    return await fn(*args, **kwargs)
                except errors as ex:
                    n += 1
                    log.warning(ex)
                    backoff = min((n**2) + random.random(), max_seconds)

                    if n > retries:
                        log.error("Retries limit reached")
                        raise

                    log.info(f"Retrying {n}/{retries}", backoff_seconds=round(backoff, 2))
                    await asyncio.sleep(backoff)

        return wrapped

    return wrapper
