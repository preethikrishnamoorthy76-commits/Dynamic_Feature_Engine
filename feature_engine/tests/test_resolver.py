from backend.engine.dependency_resolver import DependencyResolver
from backend.features.registry import build_feature_registry


def test_dependency_resolver_waves_for_pricing_and_fraud() -> None:
    registry = build_feature_registry()
    resolver = DependencyResolver(registry)
    required = ["risk_composite_score", "final_recommended_price"]

    waves = resolver.resolve_waves(required)

    flattened = [item for wave in waves for item in wave]
    assert "transaction_velocity" in flattened
    assert "device_risk_score" in flattened
    assert "discount_eligible" in flattened
    assert "demand_score" in flattened
    assert "risk_composite_score" in flattened
    assert "final_recommended_price" in flattened

    position = {name: index for index, wave in enumerate(waves) for name in wave}
    assert position["transaction_velocity"] < position["risk_composite_score"]
    assert position["device_risk_score"] < position["risk_composite_score"]
    assert position["product_price"] < position["discount_eligible"]
    assert position["discount_eligible"] < position["final_recommended_price"]
