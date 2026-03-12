from __future__ import annotations

from pathlib import Path
from typing import Dict

from backend.models.churn_model import ChurnPredictionModel
from backend.models.fraud_model import FraudDetectionModel
from backend.models.pricing_model import PricingRecommendationModel
from backend.models.recommendation_model import ProductRecommendationModel

MODEL_FILE_NAMES = {
    "fraud": "fraud_model.pkl",
    "pricing": "pricing_model.pkl",
    "churn": "churn_model.pkl",
    "recommendation": "recommendation_model.pkl",
}

MODEL_FEATURE_REQUIREMENTS = {
    "fraud": [
        "user_age",
        "transaction_amount",
        "transaction_velocity",
        "device_risk_score",
        "distance_from_home",
        "previous_fraud_attempts",
        "is_night_transaction",
        "card_present",
        "international_transaction",
    ],
    "pricing": [
        "base_price",
        "competitor_price",
        "demand_score",
        "inventory_level",
        "customer_rating",
        "seasonal_factor",
        "days_since_last_purchase",
        "user_age",
    ],
    "churn": [
        "user_age",
        "tenure_months",
        "total_purchases",
        "avg_purchase_value",
        "last_purchase_days",
        "support_tickets",
        "complaints",
        "discount_used",
        "email_open_rate",
        "app_visits_per_week",
        "payment_delays",
    ],
    "recommendation": [
        "user_age",
        "total_spent",
        "avg_rating_given",
        "browsing_time_min",
        "items_in_cart",
        "wishlist_items",
        "clicked_ads",
    ],
}


def model_save_dir() -> Path:
    return Path(__file__).resolve().parent / "saved"


def load_models() -> Dict[str, object]:
    save_dir = model_save_dir()
    return {
        "fraud": FraudDetectionModel.load(save_dir / MODEL_FILE_NAMES["fraud"]),
        "pricing": PricingRecommendationModel.load(save_dir / MODEL_FILE_NAMES["pricing"]),
        "churn": ChurnPredictionModel.load(save_dir / MODEL_FILE_NAMES["churn"]),
        "recommendation": ProductRecommendationModel.load(save_dir / MODEL_FILE_NAMES["recommendation"]),
    }


def model_paths_exist() -> bool:
    save_dir = model_save_dir()
    return all((save_dir / file_name).exists() for file_name in MODEL_FILE_NAMES.values())
