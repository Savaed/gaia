from typing import Any

from rich.console import Console


GAIA_HEADER = """
 ██████╗  █████╗ ██╗ █████╗
██╔════╝ ██╔══██╗██║██╔══██╗
██║  ███╗███████║██║███████║
██║   ██║██╔══██║██║██╔══██║
╚██████╔╝██║  ██║██║██║  ██║
 ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
"""


def print_header(description: str | None = None, **kwargs: Any) -> None:
    with Console(width=120) as console:
        console.print(f"[green]{GAIA_HEADER}")

        if description:
            console.print(f"[bold]{description}\n")

        if kwargs:
            max_key_len = max(map(len, kwargs))

            for key, value in kwargs.items():
                console.print(f"  [dim]{key:.<{max_key_len+3}} {value}")

        console.rule()
