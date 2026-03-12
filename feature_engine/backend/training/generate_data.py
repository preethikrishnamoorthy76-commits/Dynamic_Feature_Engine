from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series


def generate_fraud_data(n_rows: int = 1000, seed: int = 42) -> DatasetBundle:
    """Distributions match FRAUD_DETECTION notebook exactly."""
    rng = np.random.default_rng(seed)

    user_age = rng.integers(18, 70, size=n_rows)
    transaction_amount = rng.uniform(10, 5000, size=n_rows).round(2)
    transaction_velocity = rng.integers(1, 50, size=n_rows).astype(float)
    device_risk_score = rng.uniform(0, 1, size=n_rows).round(3)
    distance_from_home = rng.uniform(0, 1000, size=n_rows).round(1)
    previous_fraud_attempts = rng.integers(0, 5, size=n_rows).astype(float)
    is_night_transaction = rng.choice([0, 1], size=n_rows, p=[0.7, 0.3]).astype(float)
    card_present = rng.choice([0, 1], size=n_rows, p=[0.2, 0.8]).astype(float)
    international_transaction = rng.choice([0, 1], size=n_rows, p=[0.9, 0.1]).astype(float)

    fraud_prob = (
        0.3 * (transaction_amount > 3000).astype(float)
        + 0.2 * (device_risk_score > 0.8).astype(float)
        + 0.25 * (distance_from_home > 500).astype(float)
        + 0.15 * (previous_fraud_attempts > 2).astype(float)
        + 0.1 * is_night_transaction
    ).clip(0, 0.9)

    y = pd.Series((rng.random(n_rows) < fraud_prob).astype(int), name="is_fraud")
    X = pd.DataFrame(
        {
            "user_age": user_age,
            "transaction_amount": transaction_amount,
            "transaction_velocity": transaction_velocity,
            "device_risk_score": device_risk_score,
            "distance_from_home": distance_from_home,
            "previous_fraud_attempts": previous_fraud_attempts,
            "is_night_transaction": is_night_transaction,
            "card_present": card_present,
            "international_transaction": international_transaction,
        }
    )
    return DatasetBundle(X=X, y=y)


def generate_churn_data(n_rows: int = 1000, seed: int = 62) -> DatasetBundle:
    """Distributions match CHURN_PREDICTION notebook exactly."""
    rng = np.random.default_rng(seed)

    user_age = rng.integers(18, 80, size=n_rows)
    tenure_months = rng.integers(1, 73, size=n_rows).astype(float)
    total_purchases = rng.integers(1, 200, size=n_rows).astype(float)
    avg_purchase_value = rng.uniform(10, 500, size=n_rows)
    last_purchase_days = rng.integers(1, 365, size=n_rows).astype(float)
    support_tickets = rng.integers(0, 10, size=n_rows).astype(float)
    complaints = rng.integers(0, 6, size=n_rows).astype(float)
    discount_used = rng.choice([0, 1], size=n_rows).astype(float)
    email_open_rate = rng.uniform(0, 1, size=n_rows)
    app_visits_per_week = rng.integers(0, 30, size=n_rows).astype(float)
    payment_delays = rng.integers(0, 5, size=n_rows).astype(float)

    churn = (
        (last_purchase_days > 120)
        | (complaints > 3)
        | (payment_delays > 2)
        | ((tenure_months < 6) & (support_tickets > 4))
    ).astype(int)

    y = pd.Series(churn, name="churn")
    X = pd.DataFrame(
        {
            "user_age": user_age,
            "tenure_months": tenure_months,
            "total_purchases": total_purchases,
            "avg_purchase_value": avg_purchase_value,
            "last_purchase_days": last_purchase_days,
            "support_tickets": support_tickets,
            "complaints": complaints,
            "discount_used": discount_used,
            "email_open_rate": email_open_rate,
            "app_visits_per_week": app_visits_per_week,
            "payment_delays": payment_delays,
        }
    )
    return DatasetBundle(X=X, y=y)


def generate_pricing_data(n_rows: int = 1000, seed: int = 52) -> DatasetBundle:
    """Distributions match PRICING_RECOMMENDATION notebook (numeric features only)."""
    rng = np.random.default_rng(seed)

    user_age = rng.integers(18, 70, size=n_rows)
    base_price = rng.uniform(10, 1000, size=n_rows).round(2)
    competitor_price = rng.uniform(8, 1100, size=n_rows).round(2)
    demand_score = np.clip(1.0 + rng.gamma(shape=2.0, scale=0.15, size=n_rows) * 0.7, 0.6, 2.0)
    inventory_level = rng.integers(0, 500, size=n_rows).astype(float)
    customer_rating = rng.uniform(1, 5, size=n_rows).round(1)
    seasonal_factor = rng.uniform(0.7, 1.5, size=n_rows).round(2)
    days_since_last_purchase = rng.integers(1, 180, size=n_rows).astype(float)

    optimal_price = (
        competitor_price * 0.95
        * (1.0 + (demand_score - 1.0) / 2.0)
        * seasonal_factor
    ).round(2)
    optimal_price = np.clip(optimal_price, base_price * 0.7, base_price * 1.3)

    y = pd.Series(optimal_price, name="optimal_price")
    X = pd.DataFrame(
        {
            "base_price": base_price,
            "competitor_price": competitor_price,
            "demand_score": demand_score,
            "inventory_level": inventory_level,
            "customer_rating": customer_rating,
            "seasonal_factor": seasonal_factor,
            "days_since_last_purchase": days_since_last_purchase,
            "user_age": user_age,
        }
    )
    return DatasetBundle(X=X, y=y)


def generate_recommendation_data(n_rows: int = 1000, seed: int = 72) -> Tuple[pd.DataFrame, pd.Series]:
    """Distributions match PRODUCT_RECOMMENDATION notebook (numeric features only)."""
    rng = np.random.default_rng(seed)

    user_age = rng.integers(18, 70, size=n_rows)
    total_spent = rng.uniform(0, 5000, size=n_rows).round(2)
    avg_rating_given = rng.uniform(1, 5, size=n_rows).round(1)
    browsing_time_min = rng.integers(0, 120, size=n_rows).astype(float)
    items_in_cart = rng.integers(0, 10, size=n_rows).astype(float)
    wishlist_items = rng.integers(0, 30, size=n_rows).astype(float)
    clicked_ads = rng.integers(0, 20, size=n_rows).astype(float)

    categories = np.where(
        user_age < 25,
        np.where(total_spent > 2000, "electronics", "sports"),
        np.where(
            user_age < 40,
            np.where(avg_rating_given > 3.5, "fashion", "home"),
            np.where(browsing_time_min > 60, "home", "books"),
        ),
    )

    X = pd.DataFrame(
        {
            "user_age": user_age,
            "total_spent": total_spent,
            "avg_rating_given": avg_rating_given,
            "browsing_time_min": browsing_time_min,
            "items_in_cart": items_in_cart,
            "wishlist_items": wishlist_items,
            "clicked_ads": clicked_ads,
        }
    )
    y = pd.Series(categories, name="recommended_category")
    return X, y
