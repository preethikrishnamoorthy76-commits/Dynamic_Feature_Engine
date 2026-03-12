"""Section 7: Structured logging helpers for per-feature lifecycle events.

Usage example:
    import time
    from backend.runtime_engine.structured_logging import configure_logger, log_feature_event

    logger = configure_logger(verbose=True)
    start = time.perf_counter()
    log_feature_event(logger, start, 0, "START", "F_RAW_TEXT")
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional


def configure_logger(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("runtime_feature_engine")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def log_feature_event(
    logger: logging.Logger,
    start_time: float,
    wave_index: int,
    status: str,
    fid: str,
    duration_ms: Optional[float] = None,
    detail: Optional[str] = None,
) -> None:
    elapsed = time.perf_counter() - start_time
    line = f"[{elapsed:0.3f}s] [WAVE {wave_index}] {status:<6} {fid}"
    if duration_ms is not None:
        line += f" ({duration_ms:.0f}ms)"
    if detail:
        line += f" ({detail})"
    logger.info(line)
