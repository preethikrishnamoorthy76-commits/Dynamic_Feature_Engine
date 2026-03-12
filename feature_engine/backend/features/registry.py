from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    depends_on: List[str] = field(default_factory=list)
    compute_fn_key: str = ""


class FeatureRegistry:
    def __init__(self) -> None:
        self._definitions: Dict[str, FeatureDefinition] = {}

    def register(self, definition: FeatureDefinition) -> None:
        self._definitions[definition.name] = definition

    def get(self, feature_name: str) -> FeatureDefinition:
        if feature_name not in self._definitions:
            raise KeyError(f"Feature '{feature_name}' is not registered.")
        return self._definitions[feature_name]

    def list_all(self) -> List[FeatureDefinition]:
        return list(self._definitions.values())


def build_feature_registry() -> FeatureRegistry:
    registry = FeatureRegistry()

    def reg(name: str, depends_on: List[str] | None = None) -> None:
        registry.register(FeatureDefinition(name=name, depends_on=depends_on or [], compute_fn_key=name))

    # ── raw inputs (no dependencies) ──────────────────────────────────────────
    for raw in [
        "user_age", "product_price",
        "tenure_months", "last_purchase_days",
        "support_tickets", "complaints", "discount_used",
        "email_open_rate", "app_visits_per_week", "payment_delays",
        "distance_from_home", "previous_fraud_attempts",
        "card_present", "international_transaction",
        "inventory_level", "customer_rating", "seasonal_factor",
        "avg_rating_given", "browsing_time_min",
        "items_in_cart", "wishlist_items", "clicked_ads",
    ]:
        reg(raw)

    # ── derived from raw inputs (no registry deps, use input_data internally) ─
    reg("transaction_velocity")          # from transaction_history
    reg("transaction_amount")            # max(transaction_history)
    reg("total_purchases")               # len(transaction_history)
    reg("avg_purchase_value")            # mean(transaction_history)
    reg("total_spent")                   # sum(transaction_history)
    reg("device_risk_score")             # hash(device_fingerprint)
    reg("is_night_transaction")          # input field or system time
    reg("days_since_last_purchase")      # alias for last_purchase_days
    reg("base_price")                    # alias for product_price
    reg("competitor_price")              # input field or product_price*0.95

    # ── level-1: depends on level-0 ───────────────────────────────────────────
    reg("demand_score", ["transaction_velocity"])

    # ── legacy features kept for backward compatibility ────────────────────────
    reg("discount_eligible", ["user_age", "product_price"])
    reg("risk_composite_score", ["transaction_velocity", "device_risk_score"])
    reg("churn_risk_score", ["user_age", "transaction_velocity", "discount_eligible"])
    reg("final_recommended_price", ["product_price", "discount_eligible", "demand_score"])

    return registry
