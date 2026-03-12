from __future__ import annotations

import datetime
import hashlib
import math
from typing import Any, Callable, Dict

# ── helpers ────────────────────────────────────────────────────────────────────

def _raw(key: str, default: Any = 0) -> Callable[[Dict[str, Any], Dict[str, Any]], Any]:
    """Return a compute function that reads a value directly from input_data."""
    def _fn(input_data: Dict[str, Any], _cache: Dict[str, Any]) -> Any:
        return input_data.get(key, default)
    _fn.__name__ = f"compute_{key}"
    return _fn


# ── raw-input pass-throughs ────────────────────────────────────────────────────

def compute_user_age(input_data: Dict[str, Any], _: Dict[str, Any]) -> int:
    return int(input_data["user_age"])


def compute_product_price(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return float(input_data["product_price"])


# Pricing alias for product_price
def compute_base_price(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return float(input_data["product_price"])


def compute_competitor_price(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return float(input_data.get("competitor_price") or input_data["product_price"] * 0.95)


# ── features derived from transaction_history ──────────────────────────────────

def compute_transaction_velocity(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    history = input_data["transaction_history"]
    if not history:
        return 0.0
    return float(sum(float(v) for v in history) / (len(history) ** 2))


def compute_transaction_amount(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    history = input_data["transaction_history"]
    return float(max(float(v) for v in history)) if history else 0.0


def compute_total_purchases(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return float(len(input_data["transaction_history"]))


def compute_avg_purchase_value(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    history = input_data["transaction_history"]
    return float(sum(float(v) for v in history) / len(history)) if history else 0.0


def compute_total_spent(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return float(sum(float(v) for v in input_data["transaction_history"]))


# ── feature derived from device_fingerprint ────────────────────────────────────

def compute_device_risk_score(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    fingerprint = str(input_data["device_fingerprint"])
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    return float(round(int(digest[:8], 16) / 0xFFFFFFFF, 6))


# ── contextual auto-computed features ──────────────────────────────────────────

def compute_is_night_transaction(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    if "is_night_transaction" in input_data:
        return float(input_data["is_night_transaction"])
    hour = datetime.datetime.now().hour
    return float(1 if (hour < 6 or hour >= 22) else 0)


def compute_days_since_last_purchase(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return float(input_data.get("last_purchase_days", input_data.get("days_since_last_purchase", 30)))


# ── one-hot encoded feature helpers ───────────────────────────────────────────

def compute_product_category_Clothing(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("product_category") == "Clothing" else 0.0


def compute_product_category_Electronics(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("product_category") == "Electronics" else 0.0


def compute_product_category_Home(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("product_category") == "Home" else 0.0


def compute_product_category_Sports(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("product_category") == "Sports" else 0.0


def compute_user_loyalty_tier_Gold(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("user_loyalty_tier") == "Gold" else 0.0


def compute_user_loyalty_tier_Platinum(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("user_loyalty_tier") == "Platinum" else 0.0


def compute_user_loyalty_tier_Silver(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("user_loyalty_tier") == "Silver" else 0.0


def compute_user_gender_M(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("user_gender") == "M" else 0.0


def compute_user_gender_Other(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("user_gender") == "Other" else 0.0


def compute_season_Spring(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("season") == "Spring" else 0.0


def compute_season_Summer(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("season") == "Summer" else 0.0


def compute_season_Winter(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("season") == "Winter" else 0.0


def compute_purchase_history_category_Clothing(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("purchase_history_category") == "Clothing" else 0.0


def compute_purchase_history_category_Electronics(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("purchase_history_category") == "Electronics" else 0.0


def compute_purchase_history_category_Home_and_Kitchen(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("purchase_history_category") == "Home & Kitchen" else 0.0


def compute_purchase_history_category_Sports(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("purchase_history_category") == "Sports" else 0.0


def compute_purchase_history_category_Toys(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("purchase_history_category") == "Toys" else 0.0


def compute_previous_category_Clothing(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("previous_category") == "Clothing" else 0.0


def compute_previous_category_Electronics(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("previous_category") == "Electronics" else 0.0


def compute_previous_category_Home_and_Kitchen(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("previous_category") == "Home & Kitchen" else 0.0


def compute_previous_category_Sports(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("previous_category") == "Sports" else 0.0


def compute_previous_category_Toys(input_data: Dict[str, Any], _: Dict[str, Any]) -> float:
    return 1.0 if input_data.get("previous_category") == "Toys" else 0.0


# ── level-1 computed features (depend on level-0) ─────────────────────────────

def compute_demand_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    return float(max(0.6, min(2.0, 1.0 + float(cache["transaction_velocity"]) * 0.7)))


def compute_purchase_frequency_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    velocity = max(0.0, float(cache["transaction_velocity"]))
    purchases = max(0.0, float(cache["total_purchases"]))
    return float(min(5.0, velocity * math.log1p(purchases)))


def compute_spending_power_index(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    avg_purchase_value = max(0.0, float(cache["avg_purchase_value"]))
    total_spent = max(0.0, float(cache["total_spent"]))
    return float((avg_purchase_value * math.log1p(total_spent)) / 1000.0)


def compute_device_behavior_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    risk = max(0.0, float(cache["device_risk_score"]))
    is_night = bool(float(cache["is_night_transaction"]) >= 0.5)
    return float(risk * (1.3 if is_night else 1.0))


def compute_price_sensitivity_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    base = float(cache["base_price"])
    competitor = float(cache["competitor_price"])
    return float(abs(base - competitor) / max(base, 1.0))


def compute_engagement_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    email_open_rate = min(max(float(cache["email_open_rate"]), 0.0), 1.0)
    visits = min(max(float(cache["app_visits_per_week"]), 0.0), 20.0)
    return float((email_open_rate * 0.4) + ((visits / 20.0) * 0.6))


def compute_support_burden_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    tickets = max(0.0, float(cache["support_tickets"]))
    complaints = max(0.0, float(cache["complaints"]))
    return float((tickets * 0.6) + (complaints * 0.4))


def compute_recency_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    days = max(0.0, float(cache["days_since_last_purchase"]))
    return float(1.0 / (1.0 + math.log1p(days)))


def compute_loyalty_index(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    tenure = max(0.0, float(cache["tenure_months"]))
    discount_used = max(0.0, float(cache["discount_used"]))
    payment_delays = max(0.0, float(cache["payment_delays"]))
    return float((tenure / 120.0) - (payment_delays * 0.05) + (discount_used * 0.1))


def compute_customer_lifetime_value_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    spending_power = max(0.0, float(cache["spending_power_index"]))
    purchase_frequency = max(0.0, float(cache["purchase_frequency_score"]))
    loyalty = max(float(cache["loyalty_index"]), 0.1)
    return float(spending_power * purchase_frequency * loyalty)


def compute_churn_signal_strength(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    engagement = min(max(float(cache["engagement_score"]), 0.0), 1.0)
    recency = min(max(float(cache["recency_score"]), 0.0), 1.0)
    support = max(0.0, float(cache["support_burden_score"]))
    return float(((1.0 - engagement) * 0.5) + ((1.0 - recency) * 0.3) + ((support / 10.0) * 0.2))


def compute_fraud_signal_composite(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    device_behavior = max(0.0, float(cache["device_behavior_score"]))
    purchase_frequency = max(0.0, float(cache["purchase_frequency_score"]))
    price_sensitivity = max(0.0, float(cache["price_sensitivity_score"]))
    return float((device_behavior * 0.5) + (purchase_frequency * 0.3) + (price_sensitivity * 0.2))


def compute_market_position_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    price_sensitivity = min(max(float(cache["price_sensitivity_score"]), 0.0), 2.0)
    spending_power = max(0.0, float(cache["spending_power_index"]))
    return float((1.0 - price_sensitivity) * spending_power)


def compute_behavioral_risk_index(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    device_behavior = max(0.0, float(cache["device_behavior_score"]))
    support = max(0.0, float(cache["support_burden_score"]))
    recency = min(max(float(cache["recency_score"]), 0.0), 1.0)
    return float((device_behavior * 0.4) + (support * 0.4) + ((1.0 - recency) * 0.2))


def compute_customer_segment_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    clv = max(0.0, float(cache["customer_lifetime_value_score"]))
    churn_strength = max(0.0, float(cache["churn_signal_strength"]))
    risk = max(0.0, float(cache["behavioral_risk_index"]))
    return float(clv * (1.0 - churn_strength) * (1.0 - (risk * 0.1)))


def compute_pricing_elasticity_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    market_position = float(cache["market_position_score"])
    clv = max(float(cache["customer_lifetime_value_score"]), 0.01)
    return float(market_position / clv)


def compute_fraud_composite_final(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    fraud_signal = max(0.0, float(cache["fraud_signal_composite"]))
    behavior_risk = max(0.0, float(cache["behavioral_risk_index"]))
    return float((fraud_signal * 0.6) + (behavior_risk * 0.4))


def compute_dynamic_price_adjustment(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    elasticity = float(cache["pricing_elasticity_score"])
    _segment = float(cache["customer_segment_score"])
    return float(1.0 + ((elasticity - 0.5) * 0.2))


def compute_unified_risk_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    fraud_final = max(0.0, float(cache["fraud_composite_final"]))
    segment = float(cache["customer_segment_score"])
    return float((fraud_final * 0.7) + ((1.0 - segment) * 0.3))


def compute_final_customer_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    unified_risk = max(0.0, float(cache["unified_risk_score"]))
    dynamic_adjust = float(cache["dynamic_price_adjustment"])
    segment = float(cache["customer_segment_score"])
    return float(segment * dynamic_adjust * (1.0 - (unified_risk * 0.5)))


# ── legacy derived features kept for backward compat ──────────────────────────

def compute_discount_eligible(_: Dict[str, Any], cache: Dict[str, Any]) -> bool:
    return bool(cache["user_age"] > 25 and cache["product_price"] > 100)


def compute_risk_composite_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    return float(round(0.65 * float(cache["transaction_velocity"]) + 0.35 * float(cache["device_risk_score"]), 6))


def compute_churn_risk_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    age_factor = 1.0 - min(max(float(cache["user_age"]) / 100.0, 0.0), 1.0)
    velocity_factor = min(max(float(cache["transaction_velocity"]), 0.0), 1.5)
    discount_factor = 0.0 if bool(cache["discount_eligible"]) else 1.0
    return float(round(0.3 * age_factor + 0.5 * velocity_factor + 0.2 * discount_factor, 6))


def compute_final_recommended_price(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    base = float(cache["product_price"])
    multiplier = 0.85 if bool(cache["discount_eligible"]) else 1.0
    return float(round(base * multiplier * float(cache["demand_score"]), 6))


# ── master dispatch map ────────────────────────────────────────────────────────

COMPUTE_FUNCTIONS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Any]] = {
    # Raw pass-throughs
    "user_age": compute_user_age,
    "product_price": compute_product_price,
    "base_price": compute_base_price,
    "competitor_price": compute_competitor_price,
    "tenure_months": _raw("tenure_months", 12),
    "last_purchase_days": _raw("last_purchase_days", 30),
    "support_tickets": _raw("support_tickets", 0),
    "complaints": _raw("complaints", 0),
    "discount_used": _raw("discount_used", 0),
    "email_open_rate": _raw("email_open_rate", 0.3),
    "app_visits_per_week": _raw("app_visits_per_week", 5),
    "payment_delays": _raw("payment_delays", 0),
    "distance_from_home": _raw("distance_from_home", 10.0),
    "previous_fraud_attempts": _raw("previous_fraud_attempts", 0),
    "card_present": _raw("card_present", 1),
    "international_transaction": _raw("international_transaction", 0),
    "inventory_level": _raw("inventory_level", 100),
    "customer_rating": _raw("customer_rating", 4.0),
    "seasonal_factor": _raw("seasonal_factor", 1.0),
    "avg_rating_given": _raw("avg_rating_given", 3.5),
    "browsing_time_min": _raw("browsing_time_min", 15),
    "items_in_cart": _raw("items_in_cart", 2),
    "wishlist_items": _raw("wishlist_items", 5),
    "clicked_ads": _raw("clicked_ads", 3),
    # One-hot categorical features
    "product_category_Clothing": compute_product_category_Clothing,
    "product_category_Electronics": compute_product_category_Electronics,
    "product_category_Home": compute_product_category_Home,
    "product_category_Sports": compute_product_category_Sports,
    "user_loyalty_tier_Gold": compute_user_loyalty_tier_Gold,
    "user_loyalty_tier_Platinum": compute_user_loyalty_tier_Platinum,
    "user_loyalty_tier_Silver": compute_user_loyalty_tier_Silver,
    "user_gender_M": compute_user_gender_M,
    "user_gender_Other": compute_user_gender_Other,
    "season_Spring": compute_season_Spring,
    "season_Summer": compute_season_Summer,
    "season_Winter": compute_season_Winter,
    "purchase_history_category_Clothing": compute_purchase_history_category_Clothing,
    "purchase_history_category_Electronics": compute_purchase_history_category_Electronics,
    "purchase_history_category_Home & Kitchen": compute_purchase_history_category_Home_and_Kitchen,
    "purchase_history_category_Sports": compute_purchase_history_category_Sports,
    "purchase_history_category_Toys": compute_purchase_history_category_Toys,
    "previous_category_Clothing": compute_previous_category_Clothing,
    "previous_category_Electronics": compute_previous_category_Electronics,
    "previous_category_Home & Kitchen": compute_previous_category_Home_and_Kitchen,
    "previous_category_Sports": compute_previous_category_Sports,
    "previous_category_Toys": compute_previous_category_Toys,
    # Derived from transaction_history
    "transaction_velocity": compute_transaction_velocity,
    "transaction_amount": compute_transaction_amount,
    "total_purchases": compute_total_purchases,
    "avg_purchase_value": compute_avg_purchase_value,
    "total_spent": compute_total_spent,
    # Derived from device_fingerprint
    "device_risk_score": compute_device_risk_score,
    # Contextual
    "is_night_transaction": compute_is_night_transaction,
    "days_since_last_purchase": compute_days_since_last_purchase,
    # Level-1 derived
    "demand_score": compute_demand_score,
    # Level-2
    "purchase_frequency_score": compute_purchase_frequency_score,
    "spending_power_index": compute_spending_power_index,
    "device_behavior_score": compute_device_behavior_score,
    "price_sensitivity_score": compute_price_sensitivity_score,
    "engagement_score": compute_engagement_score,
    "support_burden_score": compute_support_burden_score,
    "recency_score": compute_recency_score,
    "loyalty_index": compute_loyalty_index,
    # Level-3
    "customer_lifetime_value_score": compute_customer_lifetime_value_score,
    "churn_signal_strength": compute_churn_signal_strength,
    "fraud_signal_composite": compute_fraud_signal_composite,
    "market_position_score": compute_market_position_score,
    "behavioral_risk_index": compute_behavioral_risk_index,
    # Level-4
    "customer_segment_score": compute_customer_segment_score,
    "pricing_elasticity_score": compute_pricing_elasticity_score,
    "fraud_composite_final": compute_fraud_composite_final,
    # Level-5
    "dynamic_price_adjustment": compute_dynamic_price_adjustment,
    "unified_risk_score": compute_unified_risk_score,
    # Level-6
    "final_customer_score": compute_final_customer_score,
    # Legacy
    "discount_eligible": compute_discount_eligible,
    "risk_composite_score": compute_risk_composite_score,
    "churn_risk_score": compute_churn_risk_score,
    "final_recommended_price": compute_final_recommended_price,
}
