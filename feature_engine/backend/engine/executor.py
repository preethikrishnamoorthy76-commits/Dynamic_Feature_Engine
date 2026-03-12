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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cache: Dict[str, Any] = {}
        cache_lock = Lock()
        compute_counts: Dict[str, int] = {}
        reused_count = 0

        total_start = time.perf_counter()
        for wave_index, wave in enumerate(waves):
            logger.info("Starting wave %s with features=%s", wave_index, wave)
            wave_start = time.perf_counter()

            def run_feature(feature_name: str) -> None:
                nonlocal reused_count
                with cache_lock:
                    if feature_name in cache:
                        reused_count += 1
                        logger.info("Feature '%s' reused from cache", feature_name)
                        return
                    current_cache_snapshot = dict(cache)

                definition = self.registry.get(feature_name)
                compute_fn = self.compute_functions[definition.compute_fn_key]
                try:
                    result = compute_fn(input_data, current_cache_snapshot)
                except Exception as exc:
                    raise FeatureExecutionError(
                        f"Failed computing feature '{feature_name}': {exc}"
                    ) from exc

                with cache_lock:
                    if feature_name in cache:
                        reused_count += 1
                        return
                    cache[feature_name] = result
                    compute_counts[feature_name] = compute_counts.get(feature_name, 0) + 1

            with ThreadPoolExecutor(max_workers=max(1, min(len(wave), 8))) as pool:
                futures = [pool.submit(run_feature, feature_name) for feature_name in wave]
                for future in as_completed(futures):
                    future.result()

            logger.info(
                "Finished wave %s in %.2f ms",
                wave_index,
                (time.perf_counter() - wave_start) * 1000,
            )

        total_time_ms = (time.perf_counter() - total_start) * 1000
        return cache, {
            "computed_count": len(compute_counts),
            "reused_count": reused_count,
            "feature_compute_counts": compute_counts,
            "engine_time_ms": total_time_ms,
        }
