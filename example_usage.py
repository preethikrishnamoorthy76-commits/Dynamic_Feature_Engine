import time
import random
from dynamic_feature_engine import DynamicFeatureEngine

def sleep_simulate(name: str, duration: float, required=None):
    """Simulates network/db I/O delay."""
    time.sleep(duration)
    # Return a dummy value based on what it is
    return f"{name}_val"

# --- Feature Compute Functions with magical context ---
def compute_total_transactions():
    return sleep_simulate("total_tx", 0.5)

def compute_avg_transaction(total_transactions):
    # Depending on total_transactions
    return sleep_simulate("avg_tx", 0.3)

def compute_tenure():
    return sleep_simulate("tenure", 0.2)

def compute_engagement():
    return sleep_simulate("engagement", 0.6)

def compute_customer_score(avg_transaction, tenure, engagement):
    # Deep dependency (Level 3)
    return sleep_simulate("score", 0.4)

def compute_contract_type(context):
    # Can access the runtime request context magically
    cus_id = context.get("customer_id", "unknown")
    return sleep_simulate(f"contract_type_{cus_id}", 0.1)

def compute_risk_factors(avg_transaction, context):
    return sleep_simulate("risk", 0.7)

def failing_feature(customer_score):
    time.sleep(0.2)
    raise ValueError("Network Timeout contacting external API")

def fallback_feature(context):
    print("      [Fallback Invoked] Using default caching table.")
    time.sleep(0.1)
    return "fallback_safe_val"

# --- 1. Instantiate the Engine ---
# Limiting workers to 4 to see real thread pooling in action
engine = DynamicFeatureEngine(max_workers=4)

# --- 2. Register Features (The Anti-Gravity Registry) ---
engine.register_feature("total_transactions", compute_total_transactions, deps=[])
engine.register_feature("tenure", compute_tenure, deps=[])
engine.register_feature("engagement", compute_engagement, deps=[])

# Level 2 Features
engine.register_feature("avg_transaction", compute_avg_transaction, deps=["total_transactions"])
engine.register_feature("contract_type", compute_contract_type, deps=[])

# Level 3 Features
engine.register_feature("customer_score", compute_customer_score, deps=["avg_transaction", "tenure", "engagement"])
engine.register_feature("risk_factors", compute_risk_factors, deps=["avg_transaction"])

# A feature that fails but has a fallback mechanism
engine.register_feature("external_credit_check", failing_feature, deps=["customer_score"], fallback=fallback_feature)


# --- 3. Register Models (Floating Adapters) ---
engine.register_model("churn", needs=["customer_score", "contract_type"])
engine.register_model("fraud", needs=["customer_score", "risk_factors", "external_credit_check"])


if __name__ == "__main__":
    print("\n" + "*"*70)
    print("🪄  DYNAMIC FEATURE EXECUTION ENGINE LAUNCH SEQUENCE  🪄")
    print("*"*70)
    
    # --- 4. Formulate the floating request ---
    request = {
        "models": ["churn", "fraud"], 
        "customer_id": "CUST-99231"
    }
    
    # 5. EXECUTE THE MAGIC
    print("\nSending Payload:")
    print(request)
    results = engine.execute(request)
    
    # 6. View the shared results
    print("\n" + "="*60)
    print("✨ DECOUPLED MODEL RESULTS GATHERED ✨")
    print("="*60)
    for model_name, specific_features in results.items():
        print(f"📦 Model: {model_name.upper()}")
        for feat, val in specific_features.items():
            print(f"   ├─ {feat}: {val}")
    print("="*60 + "\n")
