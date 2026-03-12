"""Dynamic runtime feature execution modules.

Usage example:
    from backend.runtime_engine import FEATURES, MODELS, FeatureExecutionEngine

    engine = FeatureExecutionEngine(FEATURES, MODELS)
    out = engine.run(["M1", "M2"])
    print(out["metrics"])
"""

from backend.runtime_engine.config import FEATURES, MODELS
from backend.runtime_engine.engine import FeatureExecutionEngine
from backend.runtime_engine.feature_cache import FeatureCache
from backend.runtime_engine.project_adapter import (
    PROJECT_COMPUTE_FUNCTIONS,
    PROJECT_FEATURES,
    PROJECT_MODELS,
)
from backend.runtime_engine.wave_planner import build_waves, collect_transitive_features

__all__ = [
    "FEATURES",
    "MODELS",
    "PROJECT_COMPUTE_FUNCTIONS",
    "PROJECT_FEATURES",
    "PROJECT_MODELS",
    "FeatureCache",
    "FeatureExecutionEngine",
    "build_waves",
    "collect_transitive_features",
]
