"""Logging utilities which extends a standard loggin capabilities."""


import logging.config
from pathlib import Path

import structlog


def configure_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(exist_ok=True)

    iso_timestamper = structlog.processors.TimeStamper(fmt="iso", utc=False)
    pre_chain = [
        # Add the log level and a timestamp to the event_dict if the log entry
        # is not from structlog.
        structlog.stdlib.add_log_level,
        # Add extra attributes of LogRecord objects to the event dictionary
        # so that values passed in the extra parameter of log methods pass
        # through to log output.
        structlog.stdlib.ExtraAdder(),
        iso_timestamper,
    ]

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "colored": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
                        structlog.dev.ConsoleRenderer(colors=True, pad_event=50),
                    ],
                },
                "flat": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        iso_timestamper,
                        structlog.processors.CallsiteParameterAdder(
                            [
                                structlog.processors.CallsiteParameter.PATHNAME,
                                structlog.processors.CallsiteParameter.FUNC_NAME,
                                structlog.processors.CallsiteParameter.THREAD_NAME,
                                structlog.processors.CallsiteParameter.LINENO,
                            ],
                        ),
                        structlog.processors.KeyValueRenderer(
                            sort_keys=True,
                            key_order=["timestamp", "level", "logger", "event"],
                        ),
                    ],
                    "foreign_pre_chain": pre_chain,
                },
                "json": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        iso_timestamper,
                        structlog.processors.CallsiteParameterAdder(
                            [
                                structlog.processors.CallsiteParameter.PATHNAME,
                                structlog.processors.CallsiteParameter.FUNC_NAME,
                                structlog.processors.CallsiteParameter.THREAD_NAME,
                                structlog.processors.CallsiteParameter.LINENO,
                            ],
                        ),
                        structlog.processors.JSONRenderer(sort_keys=True),
                    ],
                    "foreign_pre_chain": pre_chain,
                },
            },
            "handlers": {
                "console": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",
                    "formatter": "colored",
                },
                "json_file": {
                    "level": "DEBUG",
                    "class": "logging.handlers.RotatingFileHandler",
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                    "filename": f"{log_dir}/json.log",
                    "formatter": "json",
                },
                "flat_file": {
                    "level": "DEBUG",
                    "class": "logging.handlers.RotatingFileHandler",
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                    "filename": f"{log_dir}/flat.log",
                    "formatter": "flat",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["console", "json_file", "flat_file"],
                    "level": "DEBUG",
                    "propagate": True,
                },
            },
        },
    )

    structlog.configure_once(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
