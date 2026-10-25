from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"

# Tạo thư mục nếu chưa có
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SNAPSHOT_DATE = "2026-06-30"  # Ngày snapshot để tính tenure
TARGET_COL = "churn_3m"
ID_COL = "customer_id"

# Danh sách cột cần xử lý
CATEGORICAL_COLS = ["gender", "province", "occupation", "card_type", 
                    "salary_via_bank", "last_contact_channel", "currency"]
NUMERIC_COLS = ["age", "income_month", "credit_limit", "utilization_6m",
                "txn_count_3m", "txn_amount_3m", "spend_change_pct",
                "app_logins_30d", "complaints_12m", "late_payments_12m",
                "other_bank_cards", "promo_ended_60d"]