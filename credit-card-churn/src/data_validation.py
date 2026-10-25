import pandas as pd
import numpy as np
from typing import Dict, Any

def validate_train(df: pd.DataFrame) -> Dict[str, Any]:
    """Kiểm tra chất lượng dữ liệu train.csv"""
    issues = {}
    
    # 1. Check duplicates hoàn toàn
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues["full_duplicates"] = f"{dup_count} dòng trùng lặp hoàn toàn"
        print(f"⚠️  Found {dup_count} full duplicate rows")
    
    # 2. Check missing values
    missing = df.isnull().sum()
    if missing.any():
        missing_pct = (missing / len(df) * 100).round(2)
        issues["missing_values"] = missing_pct[missing_pct > 0].to_dict()
        print(f"️  Missing values: {issues['missing_values']}")
    
    # 3. Check age outliers (age > 100 hoặc age < 18)
    if "age" in df.columns:
        invalid_age = ((df["age"] < 18) | (df["age"] > 100)).sum()
        if invalid_age > 0:
            issues["invalid_age"] = f"{invalid_age} khách hàng có tuổi không hợp lệ"
            print(f"Found {invalid_age} invalid ages")
    
    # 4. Check negative income/credit_limit
    if "income_month" in df.columns:
        neg_income = (df["income_month"] < 0).sum()
        if neg_income > 0:
            issues["negative_income"] = f"{neg_income} khách hàng có thu nhập âm"
    
    if "credit_limit" in df.columns:
        neg_credit = (df["credit_limit"] < 0).sum()
        if neg_credit > 0:
            issues["negative_credit"] = f"{neg_credit} khách hàng có hạn mức âm"
    
    # 5. Check churn rate
    if "churn_3m" in df.columns:
        churn_rate = df["churn_3m"].mean()
        issues["churn_rate"] = f"{churn_rate:.3f} ({churn_rate*100:.1f}%)"
        print(f"  Churn rate: {issues['churn_rate']}")
    
    return issues

def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa các cột văn bản"""
    df = df.copy()
    
    # Chuẩn hóa gender
    gender_map = {
        "f": "F", "female": "F", "nữ": "F", "nu": "F",
        "m": "M", "male": "M", "nam": "M"
    }
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.lower().map(gender_map).fillna(df["gender"])
    
    # Chuẩn hóa province
    province_map = {
        "tp.hcm": "HCM", "tp hcm": "HCM", "hcm": "HCM", "ho chi minh": "HCM",
        "ha noi": "HN", "hanoi": "HN", "hà nội": "HN",
        "da nang": "DN", "danang": "DN", "đà nẵng": "DN"
    }
    if "province" in df.columns:
        df["province"] = df["province"].astype(str).str.strip().str.lower().map(province_map).fillna(df["province"])
    
    # Chuẩn hóa occupation
    occ_map = {
        "salaried": "salaried", "salaried ": "salaried", "SALARIED": "salaried",
        "self_employed": "self_employed", "self-employed": "self_employed",
        "self employed": "self_employed", "Self-employed": "self_employed",
        "student": "student", "Student": "student",
        "retired": "retired", "Retired": "retired",
        "civil_servant": "civil_servant", "Civil servant": "civil_servant",
        "other": "other", "Other": "other"
    }
    if "occupation" in df.columns:
        df["occupation"] = df["occupation"].astype(str).str.strip().map(occ_map).fillna(df["occupation"])
    
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý outliers nghiêm trọng"""
    df = df.copy()
    
    # Loại bỏ age > 100 hoặc < 18
    if "age" in df.columns:
        df = df[(df["age"] >= 18) & (df["age"] <= 100)]
    
    # Loại bỏ income_month âm
    if "income_month" in df.columns:
        df = df[df["income_month"] >= 0]
    
    # Loại bỏ credit_limit âm
    if "credit_limit" in df.columns:
        df = df[df["credit_limit"] >= 0]
    
    return df