import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, brier_score_loss

def calculate_metrics(y_true, y_pred_proba):
    """Tính các metric đánh giá"""
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = np.trapz(precision, recall)
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier
    }

def lift_at_top_k(y_true, y_pred_proba, k=0.1):
    """Tính lift at top K%"""
    n = len(y_true)
    top_k_idx = np.argsort(y_pred_proba)[::-1][:int(n * k)]
    top_k_true = y_true[top_k_idx]
    
    lift = top_k_true.mean() / y_true.mean()
    return lift