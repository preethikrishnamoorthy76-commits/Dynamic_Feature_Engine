"""Adapter to connect runtime engine modules with the real backend project graph.

Usage example:
    from backend.runtime_engine.project_adapter import (
        PROJECT_FEATURES,
        PROJECT_MODELS,
        PROJECT_COMPUTE_FUNCTIONS,
    )
    from backend.runtime_engine.engine import FeatureExecutionEngine

    engine = FeatureExecutionEngine(
        features=PROJECT_FEATURES,
        models=PROJECT_MODELS,
        compute_functions=PROJECT_COMPUTE_FUNCTIONS,
    )
"""

from __future__ import annotations

from typing import Any, Dict

from backend.features.compute_functions import COMPUTE_FUNCTIONS
from backend.features.registry import build_feature_registry
from backend.models import MODEL_FEATURE_REQUIREMENTS


def _build_project_features(default_cost: int = 1) -> Dict[str, Dict[str, Any]]:
    registry = build_feature_registry()
    features: Dict[str, Dict[str, Any]] = {}
    for item in registry.list_all():
        features[item.name] = {
            "deps": list(item.depends_on),
            "cost": default_cost,
            "compute_fn_key": item.compute_fn_key,
        }
    return features


def _build_project_models() -> Dict[str, Dict[str, Any]]:
    return {
        model_id: {"features": list(required_features)}
        for model_id, required_features in MODEL_FEATURE_REQUIREMENTS.items()
    }


PROJECT_FEATURES = _build_project_features(default_cost=1)
PROJECT_MODELS = _build_project_models()
PROJECT_COMPUTE_FUNCTIONS = COMPUTE_FUNCTIONS
