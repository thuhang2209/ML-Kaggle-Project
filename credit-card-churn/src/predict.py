import pandas as pd
import numpy as np
from src.config import SUBMISSIONS_DIR

def generate_submission(model, X_test, test_df, output_name="submission.csv"):
    """Tạo file submission Kaggle"""
    # Predict probability
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Clip về [0, 1]
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    # Tạo submission
    sub = pd.DataFrame({
        "customer_id": test_df["customer_id"],
        "churn_3m": y_pred_proba
    })
    
    # Lưu file
    save_path = SUBMISSIONS_DIR / output_name
    sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path} ({len(sub)} rows)")
    
    return sub