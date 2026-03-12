from backend.engine.dependency_resolver import DependencyResolver
from backend.engine.executor import WaveExecutor
from backend.features.compute_functions import COMPUTE_FUNCTIONS
from backend.features.registry import build_feature_registry


def test_executor_computes_features_once() -> None:
    registry = build_feature_registry()
    resolver = DependencyResolver(registry)
    executor = WaveExecutor(registry=registry, compute_functions=COMPUTE_FUNCTIONS)

    required = ["risk_composite_score", "churn_risk_score", "final_recommended_price"]
    waves = resolver.resolve_waves(required)
    input_data = {
        "user_age": 34,
        "product_price": 149.99,
        "transaction_history": [23.5, 67.0, 145.2, 89.0],
        "device_fingerprint": "abc123xyz",
    }

    cache, stats, failures = executor.execute(waves=waves, input_data=input_data)

    assert "risk_composite_score" in cache
    assert "churn_risk_score" in cache
    assert "final_recommended_price" in cache
    assert failures == {}
    assert all(count == 1 for count in stats["feature_compute_counts"].values())
