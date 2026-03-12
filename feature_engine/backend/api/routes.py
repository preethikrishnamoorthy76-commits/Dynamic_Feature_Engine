from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.engine.dependency_resolver import DependencyResolver
from backend.engine.executor import WaveExecutor
from backend.features.compute_functions import COMPUTE_FUNCTIONS
from backend.features.registry import FeatureDefinition, build_feature_registry
from backend.models import MODEL_FEATURE_REQUIREMENTS, load_models, model_paths_exist
from backend.training.train_all_models import train_and_save_all_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["engine"])


class ExecuteInputData(BaseModel):
    # Core fields
    user_age: int = Field(ge=0, le=120)
    product_price: float = Field(gt=0)
    transaction_history: List[float] = Field(min_length=1)
    device_fingerprint: str = Field(min_length=3)

    # Fraud
    distance_from_home: float = Field(default=10.0, ge=0)
    previous_fraud_attempts: int = Field(default=0, ge=0)
    is_night_transaction: int = Field(default=0, ge=0, le=1)
    card_present: int = Field(default=1, ge=0, le=1)
    international_transaction: int = Field(default=0, ge=0, le=1)

    # Churn
    tenure_months: int = Field(default=12, ge=0)
    last_purchase_days: int = Field(default=30, ge=0)
    support_tickets: int = Field(default=0, ge=0)
    complaints: int = Field(default=0, ge=0)
    discount_used: int = Field(default=0, ge=0, le=1)
    email_open_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    app_visits_per_week: int = Field(default=5, ge=0)
    payment_delays: int = Field(default=0, ge=0)

    # Pricing
    competitor_price: float | None = Field(default=None, gt=0)
    inventory_level: int = Field(default=100, ge=0)
    customer_rating: float = Field(default=4.0, ge=1.0, le=5.0)
    seasonal_factor: float = Field(default=1.0, gt=0)

    # Recommendation
    avg_rating_given: float = Field(default=3.5, ge=1.0, le=5.0)
    browsing_time_min: int = Field(default=15, ge=0)
    items_in_cart: int = Field(default=2, ge=0)
    wishlist_items: int = Field(default=5, ge=0)
    clicked_ads: int = Field(default=3, ge=0)

    # Categorical source fields for one-hot features
    product_category: Literal["Clothing", "Electronics", "Home", "Sports"] = "Electronics"
    user_loyalty_tier: Literal["Silver", "Gold", "Platinum"] = "Silver"
    user_gender: Literal["M", "F", "Other"] = "M"
    season: Literal["Spring", "Summer", "Autumn", "Winter"] = "Summer"
    purchase_history_category: Literal[
        "Clothing",
        "Electronics",
        "Home & Kitchen",
        "Sports",
        "Toys",
    ] = "Electronics"
    previous_category: Literal[
        "Clothing",
        "Electronics",
        "Home & Kitchen",
        "Sports",
        "Toys",
    ] = "Electronics"


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


class WaveTiming(BaseModel):
    wave: int
    duration_ms: float
    features: List[str]


class ObservabilityMetrics(BaseModel):
    cache_hits: int
    cache_misses: int
    fastest_feature: str | None
    slowest_feature: str | None
    feature_timings: Dict[str, float]
    wave_timings: List[WaveTiming]


class ExecuteResponse(BaseModel):
    results: Dict[str, Dict[str, Any]]
    execution_plan: ExecutionPlan
    metrics: Metrics
    partial_failures: Dict[str, str]
    observability: ObservabilityMetrics


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
_last_execution_metrics: ObservabilityMetrics | None = None


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
        missing = [name for name in required_features if name not in cache]
        if missing:
            results[model_key] = {"error": f"missing features: {missing}"}
            continue

        model_input = {name: cache[name] for name in required_features}
        try:
            results[model_key] = models[model_key].predict(model_input)
        except Exception as exc:
            logger.exception("Prediction failed for model '%s'", model_key)
            results[model_key] = {"error": f"prediction failed: {exc}"}

    return results


def _run_sequential_baseline(
    selected_models: List[str], input_data: Dict[str, Any]
) -> tuple[Dict[str, Dict[str, Any]], float, int]:
    models = _ensure_models_loaded()
    start = time.perf_counter()
    results: Dict[str, Dict[str, Any]] = {}
    computed_count = 0

    for model_key in selected_models:
        required_features = MODEL_FEATURE_REQUIREMENTS[model_key]
        waves = _resolver.resolve_waves(required_features)
        model_cache: Dict[str, Any] = {}
        model_failures: Dict[str, str] = {}

        for wave in waves:
            for feature_name in wave:
                if feature_name in model_cache:
                    continue
                definition = _registry.get(feature_name)
                try:
                    model_cache[feature_name] = COMPUTE_FUNCTIONS[definition.compute_fn_key](input_data, model_cache)
                except Exception as exc:
                    model_failures[feature_name] = str(exc)

        computed_count += len(model_cache)
        missing = [name for name in required_features if name not in model_cache]
        if missing:
            results[model_key] = {"error": f"missing features: {missing}"}
            continue

        model_input = {name: model_cache[name] for name in required_features}
        try:
            results[model_key] = models[model_key].predict(model_input)
        except Exception as exc:
            results[model_key] = {"error": f"prediction failed: {exc}"}

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
        "product_category",
        "user_loyalty_tier",
        "user_gender",
        "season",
        "purchase_history_category",
        "previous_category",
    }

    nodes: Dict[str, GraphNode] = {
        key: GraphNode(id=key, kind="raw_input") for key in sorted(raw_inputs)
    }
    edges: List[GraphEdge] = []

    raw_feature_dependencies: Dict[str, List[str]] = {
        "transaction_velocity": ["transaction_history"],
        "transaction_amount": ["transaction_history"],
        "total_purchases": ["transaction_history"],
        "avg_purchase_value": ["transaction_history"],
        "total_spent": ["transaction_history"],
        "device_risk_score": ["device_fingerprint"],
        "user_age": ["user_age"],
        "product_price": ["product_price"],
        "base_price": ["product_price"],
        "product_category_Clothing": ["product_category"],
        "product_category_Electronics": ["product_category"],
        "product_category_Home": ["product_category"],
        "product_category_Sports": ["product_category"],
        "user_loyalty_tier_Gold": ["user_loyalty_tier"],
        "user_loyalty_tier_Platinum": ["user_loyalty_tier"],
        "user_loyalty_tier_Silver": ["user_loyalty_tier"],
        "user_gender_M": ["user_gender"],
        "user_gender_Other": ["user_gender"],
        "season_Spring": ["season"],
        "season_Summer": ["season"],
        "season_Winter": ["season"],
        "purchase_history_category_Clothing": ["purchase_history_category"],
        "purchase_history_category_Electronics": ["purchase_history_category"],
        "purchase_history_category_Home & Kitchen": ["purchase_history_category"],
        "purchase_history_category_Sports": ["purchase_history_category"],
        "purchase_history_category_Toys": ["purchase_history_category"],
        "previous_category_Clothing": ["previous_category"],
        "previous_category_Electronics": ["previous_category"],
        "previous_category_Home & Kitchen": ["previous_category"],
        "previous_category_Sports": ["previous_category"],
        "previous_category_Toys": ["previous_category"],
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
        safe_id = node.id.replace(" ", "__")
        label = node.id.replace("_", " ")
        mermaid_lines.append(f'{safe_id}["{label}"]')
        css_class = "raw" if node.kind == "raw_input" else "feat"
        mermaid_lines.append(f"class {safe_id} {css_class};")

    for edge in edges:
        source = edge.source.replace(" ", "__")
        target = edge.target.replace(" ", "__")
        mermaid_lines.append(f"{source} --> {target}")

    return DependencyGraphResponse(nodes=list(nodes.values()), edges=edges, mermaid="\n".join(mermaid_lines))


@router.get("/metrics/last", response_model=ObservabilityMetrics)
def last_metrics() -> ObservabilityMetrics:
    if _last_execution_metrics is None:
        raise HTTPException(status_code=404, detail="No execution metrics available yet.")
    return _last_execution_metrics


@router.post("/execute", response_model=ExecuteResponse)
def execute_models(payload: ExecuteRequest) -> ExecuteResponse:
    global _last_execution_metrics

    selected_models = payload.models
    if not selected_models:
        raise HTTPException(status_code=400, detail="At least one model must be selected.")

    required_union = _compute_required_union(selected_models)
    waves = _resolver.resolve_waves(required_union)
    requested_total = sum(len(MODEL_FEATURE_REQUIREMENTS[model_key]) for model_key in selected_models)

    cache, engine_stats, failures = _executor.execute(waves=waves, input_data=payload.input_data.model_dump())

    results = _predict_with_models(selected_models, cache)
    _, sequential_time_ms, sequential_computed_count = _run_sequential_baseline(
        selected_models=selected_models,
        input_data=payload.input_data.model_dump(),
    )

    engine_time_ms = float(engine_stats["engine_time_ms"])
    features_reused = max(0, requested_total - len(required_union))
    speedup = float(sequential_time_ms / engine_time_ms) if engine_time_ms > 0 else 0.0

    observability = ObservabilityMetrics(
        cache_hits=int(engine_stats.get("cache_hits", 0)),
        cache_misses=int(engine_stats.get("cache_misses", 0)),
        fastest_feature=engine_stats.get("fastest_feature"),
        slowest_feature=engine_stats.get("slowest_feature"),
        feature_timings={k: float(v) for k, v in engine_stats.get("feature_timings", {}).items()},
        wave_timings=[WaveTiming(**item) for item in engine_stats.get("wave_timings", [])],
    )
    _last_execution_metrics = observability

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
        partial_failures=failures,
        observability=observability,
    )


@router.post("/comparison", response_model=ComparisonResponse)
def compare_execution(payload: ExecuteRequest) -> ComparisonResponse:
    selected_models = payload.models
    required_union = _compute_required_union(selected_models)
    waves = _resolver.resolve_waves(required_union)

    engine_start = time.perf_counter()
    cache, _, _ = _executor.execute(waves=waves, input_data=payload.input_data.model_dump())
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
