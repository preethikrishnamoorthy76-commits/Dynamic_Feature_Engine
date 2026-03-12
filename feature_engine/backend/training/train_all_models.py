from __future__ import annotations

import logging
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from backend.models.churn_model import ChurnPredictionModel
from backend.models.fraud_model import FraudDetectionModel
from backend.models.pricing_model import PricingRecommendationModel
from backend.models.recommendation_model import ProductRecommendationModel
from backend.training.generate_data import (
    generate_churn_data,
    generate_fraud_data,
    generate_pricing_data,
    generate_recommendation_data,
)

logger = logging.getLogger(__name__)


def train_and_save_all_models(n_rows: int = 1000) -> None:
    save_dir = Path(__file__).resolve().parents[1] / "models" / "saved"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Fraud ──────────────────────────────────────────────────────────────────
    fraud_bundle = generate_fraud_data(n_rows=n_rows)
    X_train, X_test, y_train, y_test = train_test_split(
        fraud_bundle.X.values, fraud_bundle.y.values, test_size=0.2, random_state=42, stratify=fraud_bundle.y.values
    )
    fraud_model = FraudDetectionModel()
    fraud_model.train(X_train, y_train)
    fraud_pred = fraud_model.estimator.predict(fraud_model.scaler.transform(X_test))
    logger.info(
        "Fraud model  | features=%s | accuracy=%.4f f1=%.4f",
        fraud_model.feature_names,
        accuracy_score(y_test, fraud_pred),
        f1_score(y_test, fraud_pred),
    )
    fraud_model.save(save_dir / "fraud_model.pkl")

    # ── Pricing ────────────────────────────────────────────────────────────────
    pricing_bundle = generate_pricing_data(n_rows=n_rows)
    X_train, X_test, y_train, y_test = train_test_split(
        pricing_bundle.X.values, pricing_bundle.y.values, test_size=0.2, random_state=42
    )
    pricing_model = PricingRecommendationModel()
    pricing_model.train(X_train, y_train)
    pricing_pred = pricing_model.estimator.predict(X_test)
    logger.info(
        "Pricing model | features=%s | r2=%.4f mae=%.4f",
        pricing_model.feature_names,
        r2_score(y_test, pricing_pred),
        mean_absolute_error(y_test, pricing_pred),
    )
    pricing_model.save(save_dir / "pricing_model.pkl")

    # ── Churn ──────────────────────────────────────────────────────────────────
    churn_bundle = generate_churn_data(n_rows=n_rows)
    X_train, X_test, y_train, y_test = train_test_split(
        churn_bundle.X.values, churn_bundle.y.values, test_size=0.2, random_state=42, stratify=churn_bundle.y.values
    )
    churn_model = ChurnPredictionModel()
    churn_model.train(X_train, y_train)
    churn_pred = churn_model.estimator.predict(X_test)
    logger.info(
        "Churn model  | features=%s | accuracy=%.4f f1=%.4f",
        churn_model.feature_names,
        accuracy_score(y_test, churn_pred),
        f1_score(y_test, churn_pred),
    )
    churn_model.save(save_dir / "churn_model.pkl")

    # ── Recommendation ─────────────────────────────────────────────────────────
    reco_X, reco_y = generate_recommendation_data(n_rows=n_rows)
    X_train, X_test, y_train, y_test = train_test_split(
        reco_X.values, reco_y.values, test_size=0.2, random_state=42, stratify=reco_y.values
    )
    recommendation_model = ProductRecommendationModel()
    recommendation_model.train(X_train, y_train)
    recommendation_pred = recommendation_model.estimator.predict(X_test)
    logger.info(
        "Reco model   | features=%s | accuracy=%.4f f1_macro=%.4f",
        recommendation_model.feature_names,
        accuracy_score(y_test, recommendation_pred),
        f1_score(y_test, recommendation_pred, average="macro"),
    )
    recommendation_model.save(save_dir / "recommendation_model.pkl")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    train_and_save_all_models(n_rows=1000)


if __name__ == "__main__":
    main()
