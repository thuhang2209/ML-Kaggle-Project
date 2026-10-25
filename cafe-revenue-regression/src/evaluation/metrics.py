import numpy as np

def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Tính Root Mean Squared Logarithmic Error.
    Clip prediction về >= 0 trước khi tính log để tránh lỗi math domain.
    """
    y_pred_clipped = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred_clipped)) ** 2))