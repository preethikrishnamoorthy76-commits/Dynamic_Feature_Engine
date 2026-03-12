from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]

from sklearn.ensemble import RandomForestClassifier


@dataclass
class ChurnPredictionModel:
    """XGBoost churn classifier — features match CHURN_PREDICTION notebook."""

    feature_names: List[str] = field(
        default_factory=lambda: [
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
        ]
    )
    estimator: Any | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if XGBClassifier is not None:
            self.estimator = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )
        else:
            self.estimator = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)

        self.estimator.fit(X, y)

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        if self.estimator is None:
            raise RuntimeError("ChurnPredictionModel is not trained or loaded.")

        values = np.array([[feature_dict[name] for name in self.feature_names]], dtype=float)
        probability = float(self.estimator.predict_proba(values)[0][1])
        if probability < 0.33:
            risk = "low"
        elif probability < 0.66:
            risk = "medium"
        else:
            risk = "high"

        return {
            "churn_probability": round(probability, 4),
            "churn_risk": risk,
        }

    def save(self, path: Path) -> None:
        if self.estimator is None:
            raise RuntimeError("Cannot save an untrained ChurnPredictionModel.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "feature_names": self.feature_names}, path)

    @classmethod
    def load(cls, path: Path) -> "ChurnPredictionModel":
        payload = joblib.load(path)
        return cls(feature_names=payload["feature_names"], estimator=payload["estimator"])
