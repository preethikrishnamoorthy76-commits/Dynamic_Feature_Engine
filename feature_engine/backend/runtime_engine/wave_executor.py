"""Sections 3 + 5: Parallel wave execution with cache reuse and partial-failure handling.

Usage example:
    import time
    from backend.runtime_engine.config import FEATURES
    from backend.runtime_engine.feature_cache import FeatureCache
    from backend.runtime_engine.structured_logging import configure_logger
    from backend.runtime_engine.wave_executor import execute_wave

    cache = FeatureCache()
    logger = configure_logger(verbose=True)
    start = time.perf_counter()

    def executor_fn(fid: str):
        time.sleep(FEATURES[fid]["cost"] / 1000.0)
        return f"{fid}_VALUE"

    values, events = execute_wave(
        wave=["F_RAW_TEXT", "F_USER_ID"],
        features=FEATURES,
        cache=cache,
        executor_fn=executor_fn,
        wave_index=0,
        run_start_time=start,
        logger=logger,
        per_feature_timing={},
        counters={"executed": 0, "cache_hits": 0, "skipped": 0},
    )
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, List, Tuple

from backend.runtime_engine.feature_cache import FeatureCache
from backend.runtime_engine.status import CACHED, DONE, FAIL, FAILED, SKIP, SKIPPED, START
from backend.runtime_engine.structured_logging import log_feature_event


def execute_wave(
    wave: List[str],
    features: Dict[str, Dict[str, object]],
    cache: FeatureCache,
    executor_fn: Callable[[str, Dict[str, Any]], Any],
    wave_index: int,
    run_start_time: float,
    logger,
    per_feature_timing: Dict[str, float],
    counters: Dict[str, int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Execute one wave in parallel.

    Returns:
        - dict of {fid: value_or_sentinel} for the wave
        - list of event records for CLI/reporting
    """
    events: List[Dict[str, Any]] = []
    counters_lock = Lock()
    timing_lock = Lock()

    def _run_feature(fid: str) -> Dict[str, Any]:
        if cache.has(fid):
            with counters_lock:
                counters["cache_hits"] += 1
            log_feature_event(logger, run_start_time, wave_index, CACHED, fid, detail="cache hit")
            return {
                "wave": wave_index,
                "feature": fid,
                "status": CACHED,
                "time_ms": 0.0,
                "cache_hit": True,
            }

        deps = features[fid].get("deps", [])
        upstream_failed = next((dep for dep in deps if cache.get(dep) in (FAILED, SKIPPED)), None)
        if upstream_failed:
            with counters_lock:
                counters["skipped"] += 1
            cache.set(fid, SKIPPED)
            log_feature_event(
                logger,
                run_start_time,
                wave_index,
                SKIP,
                fid,
                detail=f"upstream {upstream_failed} FAILED",
            )
            return {
                "wave": wave_index,
                "feature": fid,
                "status": SKIP,
                "time_ms": 0.0,
                "cache_hit": False,
            }

        start = time.perf_counter()
        cache_snapshot = cache.as_dict()
        with counters_lock:
            counters["cache_misses"] += 1
        log_feature_event(logger, run_start_time, wave_index, START, fid)
        try:
            value = executor_fn(fid, cache_snapshot)
            duration_ms = (time.perf_counter() - start) * 1000.0
            cache.set(fid, value)
            with timing_lock:
                per_feature_timing[fid] = duration_ms
            with counters_lock:
                counters["executed"] += 1
            log_feature_event(logger, run_start_time, wave_index, DONE, fid, duration_ms=duration_ms)
            return {
                "wave": wave_index,
                "feature": fid,
                "status": DONE,
                "time_ms": duration_ms,
                "cache_hit": False,
            }
        except Exception as exc:  # pylint: disable=broad-except
            duration_ms = (time.perf_counter() - start) * 1000.0
            cache.set(fid, FAILED)
            with timing_lock:
                per_feature_timing[fid] = duration_ms
            with counters_lock:
                counters["failed"] += 1
            log_feature_event(logger, run_start_time, wave_index, FAIL, fid, duration_ms=duration_ms, detail=str(exc))
            return {
                "wave": wave_index,
                "feature": fid,
                "status": FAIL,
                "time_ms": duration_ms,
                "cache_hit": False,
                "error": str(exc),
            }

    if not wave:
        return {}, events

    max_workers = max(1, len(wave))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_run_feature, fid): fid for fid in sorted(wave)}
        for fut in as_completed(future_map):
            events.append(fut.result())

    events.sort(key=lambda item: item["feature"])
    return {fid: cache.get(fid) for fid in sorted(wave)}, events
