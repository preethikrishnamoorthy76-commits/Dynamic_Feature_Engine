from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class ProductRecommendationModel:
    """RandomForest product recommender with optional scaler and one-hot feature support."""

    feature_names: List[str] = field(
        default_factory=lambda: [
            "user_age",
            "total_spent",
            "avg_rating_given",
            "browsing_time_min",
            "items_in_cart",
            "wishlist_items",
            "clicked_ads",
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
        ]
    )
    estimator: Any | None = None
    scaler: Any | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        self.estimator = RandomForestClassifier(n_estimators=220, random_state=42)
        self.estimator.fit(X, y)

    def predict(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.estimator is None:
            raise RuntimeError("ProductRecommendationModel is not trained or loaded.")

        values = np.array(
            [[float(feature_dict.get(name, 0.0)) for name in self.feature_names]],
            dtype=float,
        )
        values_infer = self.scaler.transform(values) if self.scaler is not None else values
        category = str(self.estimator.predict(values_infer)[0])
        confidence = float(self.estimator.predict_proba(values_infer)[0].max())
        return {
            "recommended_category": category,
            "confidence": round(confidence, 4),
        }

    def save(self, path: Path) -> None:
        if self.estimator is None:
            raise RuntimeError("Cannot save an untrained ProductRecommendationModel.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"estimator": self.estimator, "scaler": self.scaler, "feature_names": self.feature_names},
            path,
        )

    @classmethod
    def load(cls, path: Path, scaler_path: Path | None = None) -> "ProductRecommendationModel":
        payload = joblib.load(path)

        if scaler_path is not None:
            # Supports Colab exports where model and scaler are saved separately.
            return cls(estimator=payload, scaler=joblib.load(scaler_path))

        if isinstance(payload, dict):
            estimator = payload.get("estimator")
            if estimator is None:
                raise RuntimeError("Recommendation model payload missing 'estimator'.")
            feature_names = payload.get("feature_names")
            scaler = payload.get("scaler")
            if feature_names is None:
                feature_names = cls().feature_names
            return cls(feature_names=feature_names, estimator=estimator, scaler=scaler)

        # Supports legacy direct estimator joblib dumps.
        return cls(estimator=payload)
