# Dynamic Feature Execution Engine (HackHub 2026 - Problem Statement 4)

Production-grade FastAPI backend that executes feature DAGs in parallel waves and serves multiple real ML model predictions in one response.

## Architecture

- `backend/features/registry.py`
  - Defines all feature metadata and dependencies.
- `backend/engine/dependency_resolver.py`
  - Builds required DAG subset and resolves wave order using Kahn's topological sort.
- `backend/engine/executor.py`
  - Executes each wave with `ThreadPoolExecutor`, thread-safe cache, and wave logging.
- `backend/models/*.py`
  - Real trainable ML model wrappers with `train`, `predict`, `save`, and `load`.
- `backend/training/generate_data.py`
  - Synthetic data generators (1000 rows per model by default).
- `backend/training/train_all_models.py`
  - Trains and saves all four model artifacts to `backend/models/saved/`.
- `backend/api/routes.py`
  - `/api/execute`, `/api/features`, `/api/models`, `/api/comparison`.
- `frontend/index.html`
  - Lightweight UI to test the engine.

## Feature Dependency Graph

Raw input fields:
- `transaction_history`
- `device_fingerprint`
- `product_price`
- `user_age`

Registered computed features:
- `transaction_velocity` <- raw transaction history
- `device_risk_score` <- raw fingerprint hash
- `user_age`
- `product_price`
- `discount_eligible` <- `user_age`, `product_price`
- `demand_score` <- `transaction_velocity`
- `risk_composite_score` <- `transaction_velocity`, `device_risk_score`
- `churn_risk_score` <- `user_age`, `transaction_velocity`, `discount_eligible`
- `final_recommended_price` <- `product_price`, `discount_eligible`, `demand_score`

## Models

- `fraud` -> `FraudDetectionModel` (XGBoost classifier with RandomForest fallback)
- `pricing` -> `PricingRecommendationModel` (GradientBoosting regressor)
- `churn` -> `ChurnPredictionModel` (LogisticRegression classifier)
- `recommendation` -> `ProductRecommendationModel` (RandomForest classifier)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train Models (run once before server)

```bash
python -m backend.training.train_all_models
```

This writes:
- `backend/models/saved/fraud_model.pkl`
- `backend/models/saved/pricing_model.pkl`
- `backend/models/saved/churn_model.pkl`
- `backend/models/saved/recommendation_model.pkl`

## Run API Server

```bash
uvicorn backend.main:app --reload
```

- API docs: `http://127.0.0.1:8000/docs`
- Frontend: `http://127.0.0.1:8000/`

## Endpoints

### `POST /api/execute`
Runs selected models through the shared feature engine and returns unified predictions + wave plan + speed metrics.

### `GET /api/features`
Lists all registered features with dependencies.

### `GET /api/models`
Lists available model keys and required features.

### `POST /api/comparison`
Runs engine and sequential baseline for the same payload and returns timing comparison.

## Test

```bash
python -m pytest -q
```

## Notes

- Feature compute functions are pure and deterministic.
- Cache is thread-safe and feature results are reused across all requested models.
- The dependency resolver computes only the needed subgraph for the chosen model subset.
