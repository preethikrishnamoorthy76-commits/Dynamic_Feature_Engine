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

    # ── one-hot categorical features (no dependencies) ───────────────────────
    for one_hot in [
        "product_category_Clothing",
        "product_category_Electronics",
        "product_category_Home",
        "product_category_Sports",
        "user_loyalty_tier_Gold",
        "user_loyalty_tier_Platinum",
        "user_loyalty_tier_Silver",
        "user_gender_M",
        "user_gender_Other",
        "season_Spring",
        "season_Summer",
        "season_Winter",
        "purchase_history_category_Clothing",
        "purchase_history_category_Electronics",
        "purchase_history_category_Home & Kitchen",
        "purchase_history_category_Sports",
        "purchase_history_category_Toys",
        "previous_category_Clothing",
        "previous_category_Electronics",
        "previous_category_Home & Kitchen",
        "previous_category_Sports",
        "previous_category_Toys",
    ]:
        reg(one_hot)

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

    # ── level-2: depends on level-1/level-0 ───────────────────────────────────
    reg("purchase_frequency_score", ["transaction_velocity", "total_purchases"])
    reg("spending_power_index", ["avg_purchase_value", "total_spent"])
    reg("device_behavior_score", ["device_risk_score", "is_night_transaction"])
    reg("price_sensitivity_score", ["base_price", "competitor_price"])
    reg("engagement_score", ["email_open_rate", "app_visits_per_week"])
    reg("support_burden_score", ["support_tickets", "complaints"])
    reg("recency_score", ["days_since_last_purchase", "last_purchase_days"])
    reg("loyalty_index", ["tenure_months", "discount_used", "payment_delays"])

    # ── level-3 ────────────────────────────────────────────────────────────────
    reg("customer_lifetime_value_score", ["spending_power_index", "purchase_frequency_score", "loyalty_index"])
    reg("churn_signal_strength", ["engagement_score", "recency_score", "support_burden_score"])
    reg("fraud_signal_composite", ["device_behavior_score", "purchase_frequency_score", "price_sensitivity_score"])
    reg("market_position_score", ["price_sensitivity_score", "spending_power_index"])
    reg("behavioral_risk_index", ["device_behavior_score", "support_burden_score", "recency_score"])

    # ── level-4 ────────────────────────────────────────────────────────────────
    reg("customer_segment_score", ["customer_lifetime_value_score", "churn_signal_strength", "behavioral_risk_index"])
    reg("pricing_elasticity_score", ["market_position_score", "customer_lifetime_value_score"])
    reg("fraud_composite_final", ["fraud_signal_composite", "behavioral_risk_index"])

    # ── level-5 ────────────────────────────────────────────────────────────────
    reg("dynamic_price_adjustment", ["pricing_elasticity_score", "customer_segment_score"])
    reg("unified_risk_score", ["fraud_composite_final", "customer_segment_score"])

    # ── level-6 ────────────────────────────────────────────────────────────────
    reg("final_customer_score", ["unified_risk_score", "dynamic_price_adjustment", "customer_segment_score"])

    # ── legacy features kept for backward compatibility ────────────────────────
    reg("discount_eligible", ["user_age", "product_price"])
    reg("risk_composite_score", ["transaction_velocity", "device_risk_score"])
    reg("churn_risk_score", ["user_age", "transaction_velocity", "discount_eligible"])
    reg("final_recommended_price", ["product_price", "discount_eligible", "demand_score"])

    return registry
