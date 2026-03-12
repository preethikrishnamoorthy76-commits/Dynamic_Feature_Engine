from __future__ import annotations

import datetime
import hashlib
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


# ── level-1 computed features (depend on level-0) ─────────────────────────────

def compute_demand_score(_: Dict[str, Any], cache: Dict[str, Any]) -> float:
    return float(max(0.6, min(2.0, 1.0 + float(cache["transaction_velocity"]) * 0.7)))


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
    # Legacy
    "discount_eligible": compute_discount_eligible,
    "risk_composite_score": compute_risk_composite_score,
    "churn_risk_score": compute_churn_risk_score,
    "final_recommended_price": compute_final_recommended_price,
}
