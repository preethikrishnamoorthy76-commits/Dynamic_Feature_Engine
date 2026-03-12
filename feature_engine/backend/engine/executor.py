from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, List, Tuple

from backend.features.registry import FeatureRegistry

logger = logging.getLogger(__name__)


class FeatureExecutionError(RuntimeError):
    pass


class WaveExecutor:
    def __init__(
        self,
        registry: FeatureRegistry,
        compute_functions: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Any]],
    ) -> None:
        self.registry = registry
        self.compute_functions = compute_functions

    def execute(
        self,
        waves: List[List[str]],
        input_data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
        cache: Dict[str, Any] = {}
        cache_lock = Lock()
        compute_counts: Dict[str, int] = {}
        failures: Dict[str, str] = {}
        feature_timings: Dict[str, float] = {}
        wave_timings: List[Dict[str, Any]] = []
        reused_count = 0
        cache_hits = 0
        cache_misses = 0

        total_start = time.perf_counter()
        for wave_index, wave in enumerate(waves):
            logger.info("Starting wave %s with features=%s", wave_index, wave)
            wave_start = time.perf_counter()

            def run_feature(feature_name: str) -> None:
                nonlocal reused_count, cache_hits, cache_misses
                with cache_lock:
                    if feature_name in cache:
                        reused_count += 1
                        cache_hits += 1
                        logger.info("Feature '%s' reused from cache", feature_name)
                        return
                    cache_misses += 1
                    current_cache_snapshot = dict(cache)

                definition = self.registry.get(feature_name)
                compute_fn = self.compute_functions[definition.compute_fn_key]
                feature_start = time.perf_counter()
                try:
                    result = compute_fn(input_data, current_cache_snapshot)
                except Exception as exc:
                    with cache_lock:
                        failures[feature_name] = str(exc)
                        feature_timings[feature_name] = (time.perf_counter() - feature_start) * 1000
                    logger.exception("Failed computing feature '%s'", feature_name)
                    return

                with cache_lock:
                    if feature_name in cache:
                        reused_count += 1
                        cache_hits += 1
                        return
                    cache[feature_name] = result
                    compute_counts[feature_name] = compute_counts.get(feature_name, 0) + 1
                    feature_timings[feature_name] = (time.perf_counter() - feature_start) * 1000

            with ThreadPoolExecutor(max_workers=max(1, min(len(wave), 8))) as pool:
                futures = [pool.submit(run_feature, feature_name) for feature_name in wave]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        logger.exception("Unexpected worker failure in wave %s: %s", wave_index, exc)

            wave_duration_ms = (time.perf_counter() - wave_start) * 1000
            wave_timings.append(
                {
                    "wave": wave_index,
                    "duration_ms": wave_duration_ms,
                    "features": list(wave),
                }
            )
            logger.info(
                "Finished wave %s in %.2f ms",
                wave_index,
                wave_duration_ms,
            )

        total_time_ms = (time.perf_counter() - total_start) * 1000
        fastest_feature = None
        slowest_feature = None
        if feature_timings:
            fastest_feature = min(feature_timings, key=feature_timings.get)
            slowest_feature = max(feature_timings, key=feature_timings.get)

        return cache, {
            "computed_count": len(compute_counts),
            "reused_count": reused_count,
            "feature_compute_counts": compute_counts,
            "engine_time_ms": total_time_ms,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "feature_timings": {k: round(v, 3) for k, v in feature_timings.items()},
            "wave_timings": [
                {
                    "wave": wt["wave"],
                    "duration_ms": round(float(wt["duration_ms"]), 3),
                    "features": wt["features"],
                }
                for wt in wave_timings
            ],
            "fastest_feature": fastest_feature,
            "slowest_feature": slowest_feature,
        }, failures
