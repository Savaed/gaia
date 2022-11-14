"""Helpers to use with async functions."""

import asyncio
import functools
import signal
from collections.abc import Callable
from typing import Any

import structlog


log = structlog.stdlib.get_logger()


async def shutdown(
    loop: asyncio.AbstractEventLoop,
    exit_signal: signal.Signals | None = None,
) -> None:
    """
    Cancel all non-current tasks and shut down a script.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        Event loop
    signal : signal.Signals | None, optional
        Exit signal which caused shutdown procedure, by default None
    """
    cancell_msg = (
        f"Received {exit_signal=}" if exit_signal else "cancellation requested by the parent task"
    )
    tasks = (t for t in asyncio.all_tasks() if t is not asyncio.current_task())

    for task in tasks:
        task.cancel(cancell_msg)
    await asyncio.gather(*tasks, return_exceptions=True)

    if loop.is_running() and not loop.is_closed():
        loop.stop()


def handle_exception(
    loop: asyncio.AbstractEventLoop,
    context: dict[str, Any],
    callback: Callable[[], Any] | None = None,
) -> None:
    """
    Set global exception handler.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        Event loop
    context : dict[str, Any]
        Dict-like exception info
    callback : Callable[[], Any] | None, optional
        Callback function to execute when an exception is raised, by default None
    """
    # context["message"] will always be there, but context["exception"] may not
    # msg = context.get("exception", context["message"])
    final_callback = callback or functools.partial(shutdown, loop)
    asyncio.create_task(final_callback())


def prepare_loop(
    loop: asyncio.AbstractEventLoop,
    *args: Any,
    exit_signals: set[signal.Signals] | None = None,
    ex_handler: Any | None = None,
) -> None:
    """
    Set global exception handler and signal handlers for a specified event loop.

    When one of the specified exit signals is received, this function cancells
    all non-current tasks and shutdown a script.
    The default behaviour for handling exceptions is to log an exception message,
    cancel all non-current tasks and shutdown a script

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        Event loop
    exit_signals : Optional[set[signal.Signals]], optional
        Set of signals to be handle. If not specified, `SIGINT`, `SIGTERM`, `SIGHUP`
        will be handle, by default None
    ex_handler : Any | None, optional
        Handler function calling when an exception is raised, by default None

    Args
    ----
    Optional `ex_handler` positional arguments
    """
    exit_signals = exit_signals or {signal.SIGINT, signal.SIGHUP, signal.SIGTERM}

    for exit_signal in exit_signals:
        loop.add_signal_handler(
            exit_signal,
            lambda s=exit_signal: asyncio.create_task(shutdown(loop, s)),
        )

    final_ex_handler = functools.partial(ex_handler, *args) if ex_handler else handle_exception
    loop.set_exception_handler(final_ex_handler)  # type: ignore


async def cancel_tasks(tasks: list[asyncio.Future[Any] | asyncio.Task[Any]]) -> None:
    """Cancel outstanding tasks.

    Parameters
    ----------
    tasks : list[asyncio.Future[Any] | asyncio.Task[Any]]
        Tasks to cancel
    """
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
