import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import CATEGORICAL_COLS, NUMERIC_COLS

def create_preprocessing_pipeline() -> ColumnTransformer:
    """Tạo pipeline xử lý dữ liệu"""
    
    # Numeric: impute median + scale
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical: impute mode + one-hot
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS)
        ]
    )
    
    return preprocessor

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame = None):
    """
    Xử lý dữ liệu train và test
    
    Returns:
        X_train, y_train, X_test (nếu có), preprocessor
    """
    # Tách features và target
    X_train = train_df.drop(columns=["churn_3m", "customer_id"])
    y_train = train_df["churn_3m"]
    
    # Tạo và fit preprocessor
    preprocessor = create_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    
    X_test_processed = None
    if test_df is not None:
        X_test = test_df.drop(columns=["customer_id"])
        X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, y_train, X_test_processed, preprocessor