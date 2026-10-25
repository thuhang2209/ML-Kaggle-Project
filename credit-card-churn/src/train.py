import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from src.config import RANDOM_SEED

def train_baseline_logistic(X_train, y_train):
    """Huấn luyện Logistic Regression baseline"""
    model = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    
    print(f"5-fold AUC: {auc_scores.round(3)} mean {auc_scores.mean().round(3)}")
    
    # Train trên toàn bộ data
    model.fit(X_train, y_train)
    return model, auc_scores.mean()

def train_ml_models(X_train, y_train):
    """Huấn luyện các mô hình ML truyền thống"""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "NaiveBayes": GaussianNB()
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results = {}
    
    for name, model in models.items():
        auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        results[name] = {
            "mean_auc": auc_scores.mean(),
            "std_auc": auc_scores.std(),
            "model": model
        }
        print(f"{name}: {auc_scores.mean():.3f} (+/- {auc_scores.std():.3f})")
    
    return results