import pandas as pd
import numpy as np
from src.config import SNAPSHOT_DATE

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo features bổ sung từ dữ liệu gốc"""
    df = df.copy()
    
    # 1. Tính tenure (số ngày kể từ khi mở thẻ)
    if "open_date" in df.columns:
        df["open_date"] = pd.to_datetime(df["open_date"], errors="coerce", dayfirst=True)
        snapshot = pd.to_datetime(SNAPSHOT_DATE)
        df["tenure_days"] = (snapshot - df["open_date"]).dt.days
        df["tenure_months"] = df["tenure_days"] / 30.44
        df["tenure_years"] = df["tenure_days"] / 365.25
    
    # 2. Tỷ lệ sử dụng hạn mức (utilization ratio)
    if "credit_limit" in df.columns and "txn_amount_3m" in df.columns:
        df["utilization_ratio"] = df["txn_amount_3m"] / df["credit_limit"].replace(0, np.nan)
    
    # 3. Số giao dịch trung bình mỗi tháng
    if "txn_count_3m" in df.columns:
        df["avg_txn_per_month"] = df["txn_count_3m"] / 3
    
    # 4. Số tiền giao dịch trung bình
    if "txn_amount_3m" in df.columns and "txn_count_3m" in df.columns:
        df["avg_txn_amount"] = df["txn_amount_3m"] / df["txn_count_3m"].replace(0, np.nan)
    
    # 5. Tần suất đăng nhập app
    if "app_logins_30d" in df.columns:
        df["login_frequency"] = df["app_logins_30d"] / 30
    
    # 6. Tỷ lệ thanh toán muộn
    if "late_payments_12m" in df.columns and "txn_count_3m" in df.columns:
        df["late_payment_ratio"] = df["late_payments_12m"] / df["txn_count_3m"].replace(0, np.nan)
    
    # 7. Số khiếu nại trên mỗi giao dịch
    if "complaints_12m" in df.columns and "txn_count_3m" in df.columns:
        df["complaint_ratio"] = df["complaints_12m"] / df["txn_count_3m"].replace(0, np.nan)
    
    # 8. Biến động chi tiêu (absolute value)
    if "spend_change_pct" in df.columns:
        df["spend_volatility"] = df["spend_change_pct"].abs()
    
    return df