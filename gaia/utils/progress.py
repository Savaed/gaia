"""Auto-updating progress allows adding new tasks and displays a progress bar(s) based on rich."""

from typing import Any, Optional

from rich import progress
from rich.table import Column
from rich.text import Text


class JobsCompleteColumn(progress.MofNCompleteColumn):
    """Similiar to MofNCompleteColumn, but automatically select the best-matched unit scale."""

    unit_scales: dict[str, float] = {
        "K": 1_000,
        "M": 1_000_000,
        "G": 10e9,
        "T": 10e12,
        "P": 10e15,
        "E": 10e18,
        "Z": 10e21,
        "Y": 10e24,
    }

    def __init__(
        self,
        separator: str = "/",
        table_column: Optional[Column] = None,
    ) -> None:
        self.separator = separator
        super().__init__(table_column=table_column)

    def render(self, task: progress.Task) -> Text:
        """Show completed/total."""
        completed = task.completed
        total = task.total or 0

        completed_scales = list(
            filter(lambda el: completed / el[1] > 1, self.unit_scales.items())
        )
        total_scales = list(
            filter(lambda el: total / el[1] > 1, self.unit_scales.items())
        )

        completed_unit, completed_scale = (
            completed_scales[-1] if completed_scales else (None, 1)
        )
        total_unit, total_scale = total_scales[-1] if total_scales else (None, 1)

        scaled_completed = completed / completed_scale
        scaled_total = total / total_scale

        completed_txt = (
            f"{scaled_completed:.1f}" if completed_unit else f"{scaled_completed:.0f}"
        )
        completed_units_txt = completed_unit or ""
        scaled_total_txt = (
            f"{scaled_total:.1f}" if total_unit else f"{scaled_total:.0f}"
        )
        total_txt = f"{self.separator}{scaled_total_txt}{total_unit or ''}"
        return Text(
            f"{completed_txt}{completed_units_txt}{total_txt}",
            style="progress.download",
        )


class ProgressBar(progress.Progress):
    def __init__(self, file_transfer: bool = False, **kwargs: Any) -> None:
        """Custom version of `rich.Progress` to track tasks and display progress bar(s).

        Parameters
        ----------
        file_transfer : bool, optional
            Whether to use a file download progress column or a normal completed tasks column,
            by default False
        """
        super().__init__(
            progress.SpinnerColumn(),
            *progress.Progress.get_default_columns(),
            "•",
            progress.DownloadColumn() if file_transfer else JobsCompleteColumn(),
            "•",
            "elapsed",
            progress.TimeElapsedColumn(),
            **kwargs,
        )
