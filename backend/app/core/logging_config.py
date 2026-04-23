"""Optional rotating file logging for the API process."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_file_logging(log_directory: str | Path | None) -> None:
    """Attach a rotating file handler to the root logger when ``log_directory`` is set."""
    if not log_directory:
        return
    path = Path(log_directory)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / "app.log"
    handler = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    if not any(getattr(h, "baseFilename", None) == str(log_file.resolve()) for h in root.handlers):
        root.addHandler(handler)
