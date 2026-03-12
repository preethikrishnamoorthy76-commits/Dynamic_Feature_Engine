"""Execution status sentinels used in cache/results.

Usage example:
    from backend.runtime_engine.status import FAILED, SKIPPED

    assert FAILED != SKIPPED
"""

FAILED = "__FAILED__"
SKIPPED = "__SKIPPED__"
DONE = "DONE"
CACHED = "CACHED"
START = "START"
FAIL = "FAIL"
SKIP = "SKIP"
