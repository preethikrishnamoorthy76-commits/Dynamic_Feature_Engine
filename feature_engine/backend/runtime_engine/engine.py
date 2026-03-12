"""Section 4: FeatureExecutionEngine orchestrator.

Usage example:
    from backend.runtime_engine.config import FEATURES, MODELS
    from backend.runtime_engine.engine import FeatureExecutionEngine

    engine = FeatureExecutionEngine(features=FEATURES, models=MODELS)
    output = engine.run(["M1", "M3"])
    print(output["waves"])
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Set

from backend.runtime_engine.feature_cache import FeatureCache
from backend.runtime_engine.metrics import build_metrics
from backend.runtime_engine.status import FAILED
from backend.runtime_engine.structured_logging import configure_logger
from backend.runtime_engine.wave_executor import execute_wave
from backend.runtime_engine.wave_planner import build_waves, collect_transitive_features


class FeatureExecutionEngine:
    """Runtime graph orchestrator with wave-parallel execution."""

    def __init__(
        self,
        features: Dict[str, Dict[str, object]],
        models: Dict[str, Dict[str, List[str]]],
        executor_fn: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        compute_functions: Optional[Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Any]]] = None,
        verbose: bool = False,
    ) -> None:
        self.features = features
        self.models = models
        self.logger = configure_logger(verbose=verbose)
        self.executor_fn = executor_fn or self._default_executor_fn
        self.compute_functions = compute_functions

    def _default_executor_fn(self, fid: str, _cache_snapshot: Dict[str, Any]) -> Any:
        # Deterministic synthetic execution based on configured feature cost.
        cost_ms = int(self.features[fid].get("cost", 0))
        if cost_ms > 0:
            time.sleep(cost_ms / 1000.0)
        return f"{fid}_VALUE"

    def _required_features_for_models(self, model_ids: List[str]) -> Set[str]:
        unknown = [model_id for model_id in model_ids if model_id not in self.models]
        if unknown:
            raise KeyError(f"Unknown model ids: {unknown}")

        required: Set[str] = set()
        for model_id in sorted(model_ids):
            required.update(self.models[model_id].get("features", []))

        return collect_transitive_features(required, self.features)

    def _naive_per_model_compute_count(self, model_ids: List[str]) -> int:
        naive_total = 0
        for model_id in sorted(model_ids):
            features = set(self.models[model_id].get("features", []))
            closure = collect_transitive_features(features, self.features)
            naive_total += len(closure)
        return naive_total

    def run(
        self,
        model_ids: List[str],
        input_data: Optional[Dict[str, Any]] = None,
        fail_features: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Execute required features for requested models with wave parallelism."""
        fail_features = fail_features or set()
        input_data = input_data or {}

        unified_feature_set = self._required_features_for_models(model_ids)
        waves = build_waves(unified_feature_set, self.features)

        cache = FeatureCache()
        counters = {"executed": 0, "cache_hits": 0, "cache_misses": 0, "skipped": 0, "failed": 0}
        per_feature_timing: Dict[str, float] = {}
        all_events: List[Dict[str, Any]] = []
        wave_timings: List[Dict[str, Any]] = []

        def guarded_executor(fid: str, cache_snapshot: Dict[str, Any]) -> Any:
            if fid in fail_features:
                raise RuntimeError("simulated failure")
            if self.compute_functions is not None:
                compute_fn_key = str(self.features[fid].get("compute_fn_key", fid))
                if compute_fn_key not in self.compute_functions:
                    raise KeyError(f"Missing compute function for feature: {fid}")
                return self.compute_functions[compute_fn_key](input_data, cache_snapshot)
            return self.executor_fn(fid, cache_snapshot)

        wall_start = time.perf_counter()
        for wave_index, wave in enumerate(waves):
            wave_start = time.perf_counter()
            _, events = execute_wave(
                wave=wave,
                features=self.features,
                cache=cache,
                executor_fn=guarded_executor,
                wave_index=wave_index,
                run_start_time=wall_start,
                logger=self.logger,
                per_feature_timing=per_feature_timing,
                counters=counters,
            )
            wave_timings.append(
                {
                    "wave": wave_index,
                    "duration_ms": round((time.perf_counter() - wave_start) * 1000.0, 3),
                    "features": list(wave),
                }
            )
            all_events.extend(events)

        total_wall_time_ms = (time.perf_counter() - wall_start) * 1000.0
        naive_total = self._naive_per_model_compute_count(model_ids)
        metrics = build_metrics(
            executed_count=counters["executed"],
            cache_hits=counters["cache_hits"],
            cache_misses=counters["cache_misses"],
            skipped_count=counters["skipped"],
            waves_count=len(waves),
            naive_total=naive_total,
            per_feature_timing=per_feature_timing,
            total_wall_time_ms=total_wall_time_ms,
        )

        failure_details = {
            event["feature"]: str(event.get("error", "unknown error"))
            for event in all_events
            if event.get("status") == "FAIL"
        }
        failures = sorted([fid for fid, value in cache.as_dict().items() if value == FAILED])
        return {
            "results": {k: v for k, v in sorted(cache.as_dict().items())},
            "waves": [list(wave) for wave in waves],
            "metrics": metrics,
            "events": all_events,
            "failures": failures,
            "failure_details": failure_details,
            "wave_timings": wave_timings,
        }
