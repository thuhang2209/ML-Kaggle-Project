import numpy as np
from sklearn.metrics import mean_squared_error


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính toán RMSLE chuẩn mực theo yêu cầu cuộc thi."""
    y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred_clipped)
    return float(np.sqrt(mean_squared_error(log_true, log_pred)))