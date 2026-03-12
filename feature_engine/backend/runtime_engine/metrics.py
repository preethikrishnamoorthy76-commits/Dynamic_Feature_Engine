"""Section 6: Metrics collector for engine runs.

Usage example:
    from backend.runtime_engine.metrics import build_metrics

    metrics = build_metrics(
        executed_count=4,
        cache_hits=2,
        skipped_count=1,
        waves_count=3,
        naive_total=10,
        per_feature_timing={"F_RAW_TEXT": 22.3},
        total_wall_time_ms=140.7,
    )
"""

from __future__ import annotations

from typing import Dict


def build_metrics(
    executed_count: int,
    cache_hits: int,
    cache_misses: int,
    skipped_count: int,
    waves_count: int,
    naive_total: int,
    per_feature_timing: Dict[str, float],
    total_wall_time_ms: float,
) -> Dict[str, object]:
    if naive_total <= 0:
        compute_saved_pct = 0.0
    else:
        compute_saved_pct = max(0.0, ((naive_total - executed_count) / naive_total) * 100.0)

    fastest_feature = None
    slowest_feature = None
    if per_feature_timing:
        fastest_feature = min(per_feature_timing, key=per_feature_timing.get)
        slowest_feature = max(per_feature_timing, key=per_feature_timing.get)

    return {
        "total_features_executed": int(executed_count),
        "cache_hits": int(cache_hits),
        "cache_misses": int(cache_misses),
        "skipped_features": int(skipped_count),
        "waves_count": int(waves_count),
        "compute_saved_pct": round(compute_saved_pct, 2),
        "fastest_feature": fastest_feature,
        "slowest_feature": slowest_feature,
        "per_feature_timing": {k: round(float(v), 3) for k, v in sorted(per_feature_timing.items())},
        "total_wall_time_ms": round(float(total_wall_time_ms), 3),
    }
