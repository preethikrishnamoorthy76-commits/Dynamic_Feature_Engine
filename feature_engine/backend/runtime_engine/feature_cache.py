"""Section 2: Thread-safe shared feature cache.

Usage example:
    from backend.runtime_engine.feature_cache import FeatureCache

    cache = FeatureCache()
    cache.set("F_RAW_TEXT", "hello")
    assert cache.has("F_RAW_TEXT")
    assert cache.get("F_RAW_TEXT") == "hello"
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict


class FeatureCache:
    """Thread-safe cache shared across the entire engine run."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._lock = Lock()

    def has(self, fid: str) -> bool:
        with self._lock:
            return fid in self._data

    def get(self, fid: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(fid, default)

    def set(self, fid: str, value: Any) -> None:
        with self._lock:
            self._data[fid] = value

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)
