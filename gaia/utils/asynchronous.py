# type: ignore

"""Helpers to use with async functions."""

import asyncio
import functools
import signal
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import structlog


log = structlog.stdlib.get_logger()

_EventLoop: TypeAlias = asyncio.AbstractEventLoop
_Callback: TypeAlias = Callable[[], Any]


async def shutdown(loop: _EventLoop, exit_signal: signal.Signals | None = None) -> None:
    """Cancel all non-current tasks and shut down a script.

    Args:
        loop (_EventLoop): Event loop
        exit_signal (signal.Signals | None, optional): Exit signal which caused shutdown procedure.
            Defaults to None.
    """
    msg = f"Received {exit_signal=}" if exit_signal else "cancellation requested by parent task"
    log.debug(msg)
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    await cancel_tasks(tasks)
    log.debug(f"{len(tasks)} outstanding tasks cancelled")

    if loop.is_running() and not loop.is_closed():
        loop.stop()
        log.debug("Running event loop closed")


def handle_exception(
    loop: _EventLoop,
    context: dict[str, Any],
    callback: _Callback | None = None,
) -> None:
    """Set global exception handler for a specified event loop.

    Args:
        loop (_EventLoop): Event loop
        context (dict[str, Any]): Dict-like exception info
        callback (_Callback | None, optional): Callback function to execute when
            an exception is raised. Defaults to None.
    """
    # context["message"] will always be there, but context["exception"] may not
    log.error(f"Exception caught: {context.get('exception', context['message'])}")
    asyncio.create_task(callback() if callback else shutdown(loop))


def prepare_loop(
    loop: _EventLoop,
    *args: Any,
    exit_signals: set[signal.Signals] | None = None,
    ex_handler: _Callback | None = None,
) -> None:
    """Set global exception handler and signal handlers for a specified event loop.

    When one of the specified exit signals is received, this function cancells
    all non-current tasks and shutdown a script.
    The default behaviour for handling exceptions is to log an exception message,
    cancel all non-current tasks and shutdown a script

    Args:
        loop (_EventLoop): Event loop
        exit_signals (set[signal.Signals] | None, optional): Set of signals to be handle.
            If not specified, `SIGINT`, `SIGTERM`, `SIGHUP` will be handle. Defaults to None.
        ex_handler (_Callback | None, optional): Handler function calling when an
            exception is raised. Defaults to None.
        *args (Any): Optional `ex_handler` positional arguments
    """
    exit_signals = exit_signals or {signal.SIGINT, signal.SIGHUP, signal.SIGTERM}

    for exit_signal in exit_signals:
        loop.add_signal_handler(
            exit_signal,
            lambda s=exit_signal: asyncio.create_task(shutdown(loop, s)),
        )

    final_ex_handler = functools.partial(ex_handler, *args) if ex_handler else handle_exception
    loop.set_exception_handler(final_ex_handler)


async def cancel_tasks(tasks: Sequence[asyncio.Future[Any] | asyncio.Task[Any]]) -> None:
    """Cancel outstanding tasks

    Args:
        tasks (list[_Task]): Tasks to cancel
    """
    log.debug(f"Cancelling {len(tasks)} outstanding tasks")
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    log.info("Outstanding tasks cancelled")
