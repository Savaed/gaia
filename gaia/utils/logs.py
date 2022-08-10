"""Logging utilities which extends a standard loggin capabilities."""

import logging
import uuid


class LogGuidFilter(logging.Filter):
    """Add GUID to each log."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.id = str(uuid.uuid4())
        return True
