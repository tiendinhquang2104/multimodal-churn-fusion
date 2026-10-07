"""Minimal logging setup for experiment scripts."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a concise timestamped format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
