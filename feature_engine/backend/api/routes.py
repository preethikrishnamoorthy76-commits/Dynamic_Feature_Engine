from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.engine.dependency_resolver import DependencyResolver
from backend.engine.executor import FeatureExecutionError, WaveExecutor
from backend.features.compute_functions import COMPUTE_FUNCTIONS
from backend.features.registry import FeatureDefinition, build_feature_registry
from backend.models import MODEL_FEATURE_REQUIREMENTS, load_models, model_paths_exist
from backend.training.train_all_models import train_and_save_all_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["engine"])


class ExecuteInputData(BaseModel):
    # ── core fields (required) ──────────────────────────────────────────────────
    user_age: int = Field(ge=0, le=120)
    product_price: float = Field(gt=0)
    transaction_history: List[float] = Field(min_length=1)
    device_fingerprint: str = Field(min_length=3)

    # ── fraud model extra fields ────────────────────────────────────────────────
    distance_from_home: float = Field(default=10.0, ge=0)
    previous_fraud_attempts: int = Field(default=0, ge=0)
    is_night_transaction: int = Field(default=0, ge=0, le=1)
    card_present: int = Field(default=1, ge=0, le=1)
    international_transaction: int = Field(default=0, ge=0, le=1)

    # ── churn model extra fields ────────────────────────────────────────────────
    tenure_months: int = Field(default=12, ge=0)
    last_purchase_days: int = Field(default=30, ge=0)
    support_tickets: int = Field(default=0, ge=0)
    complaints: int = Field(default=0, ge=0)
    discount_used: int = Field(default=0, ge=0, le=1)
    email_open_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    app_visits_per_week: int = Field(default=5, ge=0)
    payment_delays: int = Field(default=0, ge=0)

    # ── pricing model extra fields ──────────────────────────────────────────────
    competitor_price: float | None = Field(default=None, gt=0)
    inventory_level: int = Field(default=100, ge=0)
    customer_rating: float = Field(default=4.0, ge=1.0, le=5.0)
    seasonal_factor: float = Field(default=1.0, gt=0)

    # ── recommendation model extra fields ──────────────────────────────────────
    avg_rating_given: float = Field(default=3.5, ge=1.0, le=5.0)
    browsing_time_min: int = Field(default=15, ge=0)
    items_in_cart: int = Field(default=2, ge=0)
    wishlist_items: int = Field(default=5, ge=0)
    clicked_ads: int = Field(default=3, ge=0)


class ExecuteRequest(BaseModel):
    models: List[Literal["fraud", "pricing", "churn", "recommendation"]] = Field(min_length=1)
    input_data: ExecuteInputData


class FeatureInfo(BaseModel):
    name: str
    depends_on: List[str]
    compute_fn_key: str


class ModelInfo(BaseModel):
    model_key: str
    required_features: List[str]


class GraphNode(BaseModel):
    id: str
    kind: Literal["raw_input", "derived_feature"]


class GraphEdge(BaseModel):
    source: str
    target: str


class DependencyGraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    mermaid: str


class ExecutionPlan(BaseModel):
    waves: List[List[str]]
    total_features_computed: int
    features_reused: int


class Metrics(BaseModel):
    engine_time_ms: float
    sequential_time_ms: float
    speedup_factor: float
    features_saved_from_recompute: int


class ExecuteResponse(BaseModel):
    results: Dict[str, Dict[str, Any]]
    execution_plan: ExecutionPlan
    metrics: Metrics


class ComparisonResponse(BaseModel):
    engine_results: Dict[str, Dict[str, Any]]
    sequential_results: Dict[str, Dict[str, Any]]
    engine_time_ms: float
    sequential_time_ms: float
    speedup_factor: float


_registry = build_feature_registry()
_resolver = DependencyResolver(_registry)
_executor = WaveExecutor(_registry, COMPUTE_FUNCTIONS)
_models: Dict[str, Any] = {}


def _ensure_models_loaded() -> Dict[str, Any]:
    global _models
    if _models:
        return _models

    if not model_paths_exist():
        logger.warning("Trained model artifacts not found. Running training pipeline now.")
        train_and_save_all_models(n_rows=1000)

    _models = load_models()
    return _models


def _compute_required_union(selected_models: List[str]) -> List[str]:
    required = set()
    for model_key in selected_models:
        required.update(MODEL_FEATURE_REQUIREMENTS[model_key])
    return sorted(required)


def _predict_with_models(selected_models: List[str], cache: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    models = _ensure_models_loaded()
    results: Dict[str, Dict[str, Any]] = {}

    for model_key in selected_models:
        required_features = MODEL_FEATURE_REQUIREMENTS[model_key]
        model_input = {name: cache[name] for name in required_features}
        results[model_key] = models[model_key].predict(model_input)

    return results


def _run_sequential_baseline(selected_models: List[str], input_data: Dict[str, Any]) -> tuple[Dict[str, Dict[str, Any]], float, int]:
    models = _ensure_models_loaded()
    start = time.perf_counter()
    results: Dict[str, Dict[str, Any]] = {}
    computed_count = 0

    for model_key in selected_models:
        required_features = MODEL_FEATURE_REQUIREMENTS[model_key]
        waves = _resolver.resolve_waves(required_features)
        model_cache: Dict[str, Any] = {}
        for wave in waves:
            for feature_name in wave:
                definition = _registry.get(feature_name)
                model_cache[feature_name] = COMPUTE_FUNCTIONS[definition.compute_fn_key](input_data, model_cache)
        computed_count += len(model_cache)
        model_input = {name: model_cache[name] for name in required_features}
        results[model_key] = models[model_key].predict(model_input)

    elapsed_ms = (time.perf_counter() - start) * 1000
    return results, elapsed_ms, computed_count


@router.get("/features", response_model=List[FeatureInfo])
def list_features() -> List[FeatureInfo]:
    all_features: List[FeatureDefinition] = _registry.list_all()
    return [
        FeatureInfo(name=item.name, depends_on=item.depends_on, compute_fn_key=item.compute_fn_key)
        for item in all_features
    ]


@router.get("/models", response_model=List[ModelInfo])
def list_models() -> List[ModelInfo]:
    return [
        ModelInfo(model_key=model_key, required_features=required)
        for model_key, required in MODEL_FEATURE_REQUIREMENTS.items()
    ]


@router.get("/graph", response_model=DependencyGraphResponse)
def dependency_graph() -> DependencyGraphResponse:
    raw_inputs = {
        "transaction_history",
        "device_fingerprint",
        "product_price",
        "user_age",
    }

    nodes: Dict[str, GraphNode] = {
        key: GraphNode(id=key, kind="raw_input") for key in sorted(raw_inputs)
    }
    edges: List[GraphEdge] = []

    # Explicit raw input provenance for features that are computed directly from request payload.
    raw_feature_dependencies: Dict[str, List[str]] = {
        "transaction_velocity": ["transaction_history"],
        "device_risk_score": ["device_fingerprint"],
        "user_age": ["user_age"],
        "product_price": ["product_price"],
    }

    for definition in _registry.list_all():
        if definition.name not in nodes:
            nodes[definition.name] = GraphNode(id=definition.name, kind="derived_feature")

        if definition.depends_on:
            for parent in definition.depends_on:
                edges.append(GraphEdge(source=parent, target=definition.name))
        elif definition.name in raw_feature_dependencies:
            for parent in raw_feature_dependencies[definition.name]:
                edges.append(GraphEdge(source=parent, target=definition.name))

    mermaid_lines = [
        "graph LR",
        "classDef raw fill:#fcefd7,stroke:#e5b56b,stroke-width:1px,color:#2a2a2a;",
        "classDef feat fill:#dff3ff,stroke:#5ba8d6,stroke-width:1px,color:#1f2b34;",
    ]

    for node in sorted(nodes.values(), key=lambda n: n.id):
        label = node.id.replace("_", " ")
        mermaid_lines.append(f'{node.id}["{label}"]')
        css_class = "raw" if node.kind == "raw_input" else "feat"
        mermaid_lines.append(f"class {node.id} {css_class};")

    for edge in edges:
        mermaid_lines.append(f"{edge.source} --> {edge.target}")

    return DependencyGraphResponse(nodes=list(nodes.values()), edges=edges, mermaid="\n".join(mermaid_lines))


@router.post("/execute", response_model=ExecuteResponse)
def execute_models(payload: ExecuteRequest) -> ExecuteResponse:
    selected_models = payload.models
    if not selected_models:
        raise HTTPException(status_code=400, detail="At least one model must be selected.")

    required_union = _compute_required_union(selected_models)
    waves = _resolver.resolve_waves(required_union)
    requested_total = sum(len(MODEL_FEATURE_REQUIREMENTS[model_key]) for model_key in selected_models)

    try:
        cache, engine_stats = _executor.execute(waves=waves, input_data=payload.input_data.model_dump())
    except FeatureExecutionError as exc:
        logger.exception("Feature execution failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected execution failure")
        raise HTTPException(status_code=500, detail=f"Unexpected execution error: {exc}") from exc

    results = _predict_with_models(selected_models, cache)
    _, sequential_time_ms, sequential_computed_count = _run_sequential_baseline(
        selected_models=selected_models,
        input_data=payload.input_data.model_dump(),
    )

    engine_time_ms = float(engine_stats["engine_time_ms"])
    features_reused = max(0, requested_total - len(required_union))
    speedup = float(sequential_time_ms / engine_time_ms) if engine_time_ms > 0 else 0.0

    return ExecuteResponse(
        results=results,
        execution_plan=ExecutionPlan(
            waves=waves,
            total_features_computed=int(engine_stats["computed_count"]),
            features_reused=features_reused,
        ),
        metrics=Metrics(
            engine_time_ms=round(engine_time_ms, 3),
            sequential_time_ms=round(sequential_time_ms, 3),
            speedup_factor=round(speedup, 3),
            features_saved_from_recompute=max(0, sequential_computed_count - int(engine_stats["computed_count"])),
        ),
    )


@router.post("/comparison", response_model=ComparisonResponse)
def compare_execution(payload: ExecuteRequest) -> ComparisonResponse:
    selected_models = payload.models
    required_union = _compute_required_union(selected_models)
    waves = _resolver.resolve_waves(required_union)

    engine_start = time.perf_counter()
    cache, _ = _executor.execute(waves=waves, input_data=payload.input_data.model_dump())
    engine_results = _predict_with_models(selected_models, cache)
    engine_time_ms = (time.perf_counter() - engine_start) * 1000

    sequential_results, sequential_time_ms, _ = _run_sequential_baseline(
        selected_models=selected_models,
        input_data=payload.input_data.model_dump(),
    )

    speedup = float(sequential_time_ms / engine_time_ms) if engine_time_ms > 0 else 0.0
    return ComparisonResponse(
        engine_results=engine_results,
        sequential_results=sequential_results,
        engine_time_ms=round(engine_time_ms, 3),
        sequential_time_ms=round(sequential_time_ms, 3),
        speedup_factor=round(speedup, 3),
    )
