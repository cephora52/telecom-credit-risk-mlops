from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "telecom_credit_scoring_data.csv"
MODEL_PATH = ROOT / "models" / "default_risk_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

TARGET = "default_risk"

FEATURES = [
    "operator",
    "credit_score",
    "credit_rating",
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
    "payment_consistency_score",
    "late_payment_count",
    "account_type",
]
