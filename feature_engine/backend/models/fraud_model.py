from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - graceful fallback when xgboost is unavailable
    XGBClassifier = None  # type: ignore[assignment]


@dataclass
class FraudDetectionModel:
    """XGBoost fraud classifier — features match FRAUD_DETECTION notebook."""

    feature_names: List[str] = field(
        default_factory=lambda: [
            "user_age",
            "transaction_amount",
            "transaction_velocity",
            "device_risk_score",
            "distance_from_home",
            "previous_fraud_attempts",
            "is_night_transaction",
            "card_present",
            "international_transaction",
        ]
    )
    estimator: Any | None = None
    scaler: Any | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if XGBClassifier is not None:
            self.estimator = XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            )
        else:
            self.estimator = RandomForestClassifier(n_estimators=300, random_state=42)

        self.estimator.fit(X_scaled, y)

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        if self.estimator is None or self.scaler is None:
            raise RuntimeError("FraudDetectionModel is not trained or loaded.")

        values = np.array([[feature_dict[name] for name in self.feature_names]], dtype=float)
        values_scaled = self.scaler.transform(values)
        probability = float(self.estimator.predict_proba(values_scaled)[0][1])
        return {
            "fraud_probability": round(probability, 4),
            "is_fraud": probability >= 0.619,
        }

    def save(self, path: Path) -> None:
        if self.estimator is None:
            raise RuntimeError("Cannot save an untrained FraudDetectionModel.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"estimator": self.estimator, "scaler": self.scaler, "feature_names": self.feature_names},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "FraudDetectionModel":
        payload = joblib.load(path)
        return cls(
            feature_names=payload["feature_names"],
            estimator=payload["estimator"],
            scaler=payload.get("scaler"),
        )
