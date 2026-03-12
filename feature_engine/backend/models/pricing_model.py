from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class PricingRecommendationModel:
    """GradientBoosting price regressor with optional scaler and one-hot feature support."""

    feature_names: List[str] = field(
        default_factory=lambda: [
            "base_price",
            "competitor_price",
            "demand_score",
            "inventory_level",
            "customer_rating",
            "seasonal_factor",
            "days_since_last_purchase",
            "user_age",
            "product_category_Clothing",
            "product_category_Electronics",
            "product_category_Home",
            "product_category_Sports",
            "user_loyalty_tier_Gold",
            "user_loyalty_tier_Platinum",
            "user_loyalty_tier_Silver",
        ]
    )
    estimator: Any | None = None
    scaler: Any | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        self.estimator = GradientBoostingRegressor(random_state=42, n_estimators=200, max_depth=4)
        self.estimator.fit(X, y)

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        if self.estimator is None:
            raise RuntimeError("PricingRecommendationModel is not trained or loaded.")

        values = np.array(
            [[float(feature_dict.get(name, 0.0)) for name in self.feature_names]],
            dtype=float,
        )
        values_infer = self.scaler.transform(values) if self.scaler is not None else values
        prediction = float(self.estimator.predict(values_infer)[0])
        base = float(feature_dict["base_price"])
        competitor = float(feature_dict["competitor_price"])
        discount_applied = prediction < base * 0.99
        return {
            "recommended_price": round(max(1.0, prediction), 2),
            "discount_applied": discount_applied,
        }

    def save(self, path: Path) -> None:
        if self.estimator is None:
            raise RuntimeError("Cannot save an untrained PricingRecommendationModel.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"estimator": self.estimator, "scaler": self.scaler, "feature_names": self.feature_names},
            path,
        )

    @classmethod
    def load(cls, path: Path, scaler_path: Path | None = None) -> "PricingRecommendationModel":
        payload = joblib.load(path)

        if scaler_path is not None:
            # Supports Colab exports where model and scaler are saved separately.
            return cls(estimator=payload, scaler=joblib.load(scaler_path))

        if isinstance(payload, dict):
            estimator = payload.get("estimator")
            if estimator is None:
                raise RuntimeError("Pricing model payload missing 'estimator'.")
            feature_names = payload.get("feature_names")
            scaler = payload.get("scaler")
            if feature_names is None:
                feature_names = cls().feature_names
            return cls(feature_names=feature_names, estimator=estimator, scaler=scaler)

        # Supports legacy direct estimator joblib dumps.
        return cls(estimator=payload)
