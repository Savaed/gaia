"""Helpers to use with async functions."""

import asyncio
import signal
from collections.abc import Callable, Coroutine
from typing import Any, TypeAlias

from gaia.log import logger


EventLoop: TypeAlias = asyncio.AbstractEventLoop
_Handler: TypeAlias = Callable[[signal.Signals | None], Coroutine[Any, Any, None]]
_CancellableList: TypeAlias = list[asyncio.Task[Any]] | list[asyncio.Future[Any]]


async def shutdown(exit_signal: signal.Signals | None = None) -> None:
    """Cancel all outstanding and non-current tasks and shut down a script.

    Args:
        exit_signal (signal.Signals | None, optional): Exit signal which caused shutdown procedure.
        Defaults to None.
    """
    if exit_signal:
        logger.info(f"Received {exit_signal.name} exit signal")

    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    await cancel_tasks(tasks)
    logger.info("Shutdown complete")


def set_signals_handler(
    loop: EventLoop,
    handler: _Handler,
    exit_signals: set[signal.Signals] | None = None,
) -> None:
    """Set signals handler for a specified event loop.

    Args:
        loop (_EventLoop): Event loop
        handler (_Handler): Handler to be called when the signal received
        exit_signals (set[signal.Signals] | None, optional): Signals to be handled.
        If not specified, `SIGINT`, `SIGTERM`, and `SIGHUP` will be handled. Defaults to None.
    """
    exit_signals = exit_signals or {signal.SIGINT, signal.SIGHUP, signal.SIGTERM}
    for exit_signal in exit_signals:
        loop.add_signal_handler(
            exit_signal,
            lambda s=exit_signal: asyncio.create_task(handler(s)),
        )


def log_exception(_: EventLoop, context: dict[str, Any]) -> None:
    """Log any unhandled exceptions from tasks as `logging.ERROR`.

    See `asyncio.loop.call_exception_handler()` docs for `context` details.

    Args:
        _ (_EventLoop): Event loop, unused
        context (dict[str, Any]): Exception info with at least 'message' key
    """
    # In case of context['excpetion'] is a bare Exception evaluated to an empty string
    exception = context.get("exception", None)
    logger.error(f"Exception caught: {str(exception) or context['message'] }")


async def cancel_tasks(tasks: _CancellableList) -> None:
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Tasks cancelled")


def prepare_loop(loop: EventLoop) -> None:
    """Set exception and signal default handlers.

    Args:
        loop (EventLoop): `asyncio` event loop
    """
    loop.set_exception_handler(log_exception)
    set_signals_handler(loop, shutdown)
