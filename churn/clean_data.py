import pandas as pd
import numpy as np


def clean_data(df):
    """Làm sạch dữ liệu churn, giữ lại cờ báo ô thiếu và giá trị bất thường."""
    df = df.copy()

    # 1. Chuẩn hóa chuỗi
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()

    # 2. Chuẩn hóa gender
    if "gender" in df.columns:
        gender_map = {
            "F": "F", "FEMALE": "F", "NỮ": "F", "NU": "F",
            "M": "M", "MALE": "M", "NAM": "M"
        }
        df["gender"] = (
            df["gender"].str.upper().replace(gender_map)
        )

    # 3. Chuẩn hóa province
    if "province" in df.columns:
        province_map = {
            "TP.HCM": "HCM",
            "TP HCM": "HCM",
            "TP. HCM": "HCM",
            "THÀNH PHỐ HỒ CHÍ MINH": "HCM",
            "HỒ CHÍ MINH": "HCM",
            "HO CHI MINH": "HCM",
            "BÀ RỊA VŨNG TÀU": "Bà Rịa - Vũng Tàu",
            "BÀ RỊA-VŨNG TÀU": "Bà Rịa - Vũng Tàu",
            "BA RIA VUNG TAU": "Bà Rịa - Vũng Tàu"
        }
        df["province"] = df["province"].replace(province_map)

    # 4. Chuyển cột số về kiểu numeric nếu cần
    numeric_cols = [
        "age", "income_month", "other_bank_cards",
        "credit_limit", "utilization_6m", "txn_count_3m",
        "txn_amount_3m", "spend_change_pct", "app_logins_30d",
        "complaints_12m", "late_payments_12m"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Lưu dấu ô thiếu ban đầu
    for col in ["age", "income_month", "other_bank_cards",
                "app_logins_30d", "occupation", "salary_via_bank",
                "last_contact_channel"]:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype("int8")

    # 6. Đánh dấu và xử lý giá trị bất thường
    if "age" in df.columns:
        df["age_outlier_flag"] = (
            (df["age"] < 18) | (df["age"] > 100)
        ).astype("int8")
        df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = np.nan

    for col in ["income_month", "other_bank_cards"]:
        if col in df.columns:
            df[f"{col}_invalid_flag"] = (df[col] < 0).astype("int8")
            df.loc[df[col] < 0, col] = np.nan

    # Những biến đếm không thể âm
    for col in ["txn_count_3m", "app_logins_30d",
                "complaints_12m", "late_payments_12m"]:
        if col in df.columns:
            df[f"{col}_invalid_flag"] = (df[col] < 0).astype("int8")
            df.loc[df[col] < 0, col] = np.nan

    # Các giá trị tỷ lệ sử dụng thẻ phải nằm trong [0, 1]
    if "utilization_6m" in df.columns:
        invalid = ~df["utilization_6m"].between(0, 1)
        df["utilization_6m_invalid_flag"] = invalid.astype("int8")
        df.loc[invalid, "utilization_6m"] = np.nan

    # 7. Chuẩn hóa ngày; ngày không đọc được trở thành NaT
    if "open_date" in df.columns:
        df["open_date"] = pd.to_datetime(
            df["open_date"],
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )

    # 8. Điền thiếu cho biến số bằng trung vị
    # Không điền biến mục tiêu churn_3m hoặc các cột ID
    exclude = {"churn_3m", "customer_id"}
    for col in df.select_dtypes(include="number").columns:
        if col not in exclude and df[col].isna().any():
            median_value = df[col].median()
            if pd.notna(median_value):
                df[col] = df[col].fillna(median_value)

    # 9. Điền thiếu cho biến phân loại
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col not in exclude:
            df[col] = df[col].fillna("Unknown")

    return df


if __name__ == "__main__":
    input_path = "/content/churn/train.csv"
    output_path = "/content/cleaned_churn.csv"

    raw = pd.read_csv(input_path)
    cleaned = clean_data(raw)
    cleaned.to_csv(output_path, index=False)

    print("Kích thước ban đầu:", raw.shape)
    print("Kích thước sau làm sạch:", cleaned.shape)
    print("Số ô thiếu còn lại:")
    print(cleaned.isna().sum()[cleaned.isna().sum() > 0])
    print("Đã lưu:", output_path)
