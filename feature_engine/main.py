"""Section 8: CLI entrypoint for the runtime feature engine.

Usage examples:
    python main.py --models M1 M3
    python main.py --models M1 M2 --fail F_TOKENS
    python main.py --models M1 M2 M3 --verbose
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from backend.runtime_engine import (
    PROJECT_COMPUTE_FUNCTIONS,
    PROJECT_FEATURES,
    PROJECT_MODELS,
    FeatureExecutionEngine,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic Feature Execution Engine CLI")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model IDs (fraud pricing churn recommendation) or 'ALL'",
    )
    parser.add_argument("--fail", nargs="*", default=[], help="Optional feature IDs to force-fail")
    parser.add_argument("--verbose", action="store_true", help="Enable structured execution logs")
    parser.add_argument("--plan-only", action="store_true", help="Show plan without executing")
    return parser.parse_args()


def _resolve_models(model_args: List[str]) -> List[str]:
    """Convert 'ALL' to list of all models, or return the provided list."""
    if len(model_args) == 1 and model_args[0].upper() == "ALL":
        return list(PROJECT_MODELS.keys())
    return model_args


def _default_input_data() -> Dict[str, Any]:
    return {
        "user_age": 29,
        "product_price": 1200.0,
        "transaction_history": [150, 220, 180, 210, 160, 300],
        "device_fingerprint": "abc123xyz",
        "distance_from_home": 12.5,
        "previous_fraud_attempts": 0,
        "is_night_transaction": 0,
        "card_present": 1,
        "international_transaction": 0,
        "tenure_months": 18,
        "last_purchase_days": 21,
        "support_tickets": 1,
        "complaints": 0,
        "discount_used": 1,
        "email_open_rate": 0.42,
        "app_visits_per_week": 6,
        "payment_delays": 0,
        "competitor_price": 1150.0,
        "inventory_level": 140,
        "customer_rating": 4.3,
        "seasonal_factor": 1.05,
        "avg_rating_given": 4.0,
        "browsing_time_min": 19,
        "items_in_cart": 3,
        "wishlist_items": 7,
        "clicked_ads": 2,
        "product_category": "Electronics",
        "user_loyalty_tier": "Silver",
        "user_gender": "M",
        "season": "Summer",
        "purchase_history_category": "Electronics",
        "previous_category": "Electronics",
    }


def _print_summary(events: List[Dict[str, Any]]) -> None:
    headers = ("Wave", "Feature", "Status", "Time(ms)", "CacheHit")
    rows = [
        (
            str(ev.get("wave", "-")),
            str(ev.get("feature", "-")),
            str(ev.get("status", "-")),
            f"{float(ev.get('time_ms', 0.0)):.2f}",
            "Y" if bool(ev.get("cache_hit", False)) else "N",
        )
        for ev in events
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, col in enumerate(row):
            widths[idx] = max(widths[idx], len(col))

    def fmt(row: tuple[str, str, str, str, str]) -> str:
        return " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in sorted(rows, key=lambda r: (int(r[0]) if r[0].isdigit() else 9999, r[1])):
        print(fmt(row))


def main() -> None:
    args = _parse_args()
    
    # Resolve model list
    model_ids = _resolve_models(args.models)

    engine = FeatureExecutionEngine(
        features=PROJECT_FEATURES,
        models=PROJECT_MODELS,
        compute_functions=PROJECT_COMPUTE_FUNCTIONS,
        verbose=args.verbose,
    )
    
    # Show plan
    print(f"\n{'='*70}")
    print("EXECUTION PLAN")
    print(f"{'='*70}")
    print(f"Models:        {', '.join(model_ids)}")
    print(f"Failures:      {args.fail if args.fail else 'None'}")
    print(f"Verbose:       {args.verbose}")
    print(f"{'='*70}\n")
    
    if args.plan_only:
        print("[--plan-only flag set] Stopping after plan display.")
        return
    
    output = engine.run(
        model_ids=model_ids,
        input_data=_default_input_data(),
        fail_features=set(args.fail),
    )

    print("\nExecution Summary")
    _print_summary(output["events"])

    print("\nWaves:")
    for idx, wave in enumerate(output["waves"]):
        print(f"  Wave {idx}: {wave}")

    print("\nMetrics:")
    for key, value in output["metrics"].items():
        print(f"  {key}: {value}")

    print("\nFailures:")
    print(f"  {output['failures'] if output['failures'] else 'None'}")


if __name__ == "__main__":
    main()
