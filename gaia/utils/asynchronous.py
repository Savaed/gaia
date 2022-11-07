"""Helpers to use with async functions."""

import asyncio
import functools
import signal
from collections.abc import Callable
from typing import Any

import structlog


log = structlog.stdlib.get_logger()


async def shutdown(
    loop: asyncio.AbstractEventLoop, exit_signal: signal.Signals | None = None
) -> None:
    """
    Cancel all non-current tasks and shut down a script.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        Event loop
    signal : Optional[signal.Signals], optional
        Exit signal which caused shutdown procedure, by default None
    """
    cancellation_message = "cancellation requested by the parent task"

    if exit_signal:
        message = f"Received exit signal {exit_signal.name}"
        log.info(message)
        cancellation_message = message.lower()

    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    log.info(f"Cancelling {len(tasks)} outstanding tasks")
    for task in tasks:
        task.cancel(cancellation_message)
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
    callback : Any, optional
        Callback function to execute when an exception is raised, by default None
    """
    # context["message"] will always be there, but context["exception"] may not
    msg = context.get("exception", context["message"])
    log.error(f"Caught exception on asyncio loop: {msg}", loop=loop)

    if callback:
        asyncio.create_task(callback())
    else:
        asyncio.create_task(shutdown(loop))


def prepare_loop(
    loop: asyncio.AbstractEventLoop,
    *args: Any,
    exit_signals: set[signal.Signals] | None = None,
    exception_handler: Any | None = None,
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
    exception_handler : Optional[Any], optional
        Handler function calling when an exception is raised, by default None

    Args
    ----
    Optional `exception_handler` positional arguments
    """
    if not exit_signals:
        exit_signals = {signal.SIGINT, signal.SIGHUP, signal.SIGTERM}

    for exit_signal in exit_signals:
        loop.add_signal_handler(
            exit_signal, lambda s=exit_signal: asyncio.create_task(shutdown(loop, s))
        )
    log.info("Asyncio signal handlers set", signals=exit_signals)

    if exception_handler:
        loop.set_exception_handler(functools.partial(exception_handler, *args))
    else:
        loop.set_exception_handler(handle_exception)
    log.info("Asyncio exception handlers set")


async def cancel_tasks(tasks: list[asyncio.Future[Any] | asyncio.Task[Any]]) -> None:
    """Cancel outstanding tasks.

    Parameters
    ----------
    tasks : list[Union[asyncio.Future, asyncio.Task]]
        Tasks to cancel
    """
    log.info("Cancelling outstanding tasks")
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    log.info(f"{len(tasks)} tasks cencelled")
