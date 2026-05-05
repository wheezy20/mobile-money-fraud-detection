"""
Section D Phase 2: Train three baseline models.

Trains Logistic Regression, Random Forest, and XGBoost on the prepared
training set and evaluates each on the validation set.

Uses class_weight='balanced' (or scale_pos_weight for XGBoost) to handle
the 0.30% fraud rate without resampling. No hyperparameter tuning yet —
that comes in Phase 4 once we identify the strongest baseline.

Outputs:
    - models/baseline_logreg.pkl
    - models/baseline_rf.pkl
    - models/baseline_xgb.pkl
    - reports/figures/fig_04_baseline_comparison.png
    - Console summary of all three models' validation metrics
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mobile_money_fraud"
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

sns.set_style("whitegrid")

# %%
# Load the splits
splits_dir = PROJECT_ROOT / "data" / "processed" / "splits"

X_train = pd.read_csv(splits_dir / "X_train.csv")
X_val = pd.read_csv(splits_dir / "X_val.csv")
y_train = pd.read_csv(splits_dir / "y_train.csv").iloc[:, 0]
y_val = pd.read_csv(splits_dir / "y_val.csv").iloc[:, 0]

print(f"Train: {X_train.shape}, fraud rate: {y_train.mean() * 100:.4f}%")
print(f"Val:   {X_val.shape}, fraud rate: {y_val.mean() * 100:.4f}%")

# %%
# Define a helper that trains, evaluates, and saves one model.
def train_and_evaluate(name, model, X_tr, y_tr, X_va, y_va, save_dir):
    """Train one model, evaluate on validation, return a results dict."""
    print(f"\n{'=' * 60}")
    print(f"Training {name}...")
    print(f"{'=' * 60}")
    
    start = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start
    
    # Predictions and probabilities
    y_pred = model.predict(X_va)
    y_proba = model.predict_proba(X_va)[:, 1]  # probability of class 1 (fraud)
    
    # Metrics
    pr_auc = average_precision_score(y_va, y_proba)
    roc_auc = roc_auc_score(y_va, y_proba)
    f1 = f1_score(y_va, y_pred)
    precision = precision_score(y_va, y_pred, zero_division=0)
    recall = recall_score(y_va, y_pred)
    cm = confusion_matrix(y_va, y_pred)
    
    print(f"\nTraining time: {train_time:.1f}s")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"\nConfusion matrix (val set):")
    print(f"  TN={cm[0,0]:>7,}  FP={cm[0,1]:>5,}")
    print(f"  FN={cm[1,0]:>7,}  TP={cm[1,1]:>5,}")
    
    # Save the trained model
    save_path = save_dir / f"baseline_{name}.pkl"
    joblib.dump(model, save_path)
    print(f"Saved: {save_path}")
    
    return {
        "name": name,
        "train_time_s": train_time,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": cm[1, 1],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tn": cm[0, 0],
    }

# %%
# Set up models with class-imbalance handling
models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(exist_ok=True)

# Compute scale_pos_weight for XGBoost = (negatives / positives)
# This is XGBoost's equivalent of class_weight='balanced'
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight for XGBoost: {scale_pos_weight:.1f}")

models = {
    "logreg": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    ),
    "rf": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,        # cap to avoid overfitting on baseline
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "xgb": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    ),
}

# %%
# Train all three and collect results
results = []
for name, model in models.items():
    result = train_and_evaluate(
        name, model, X_train, y_train, X_val, y_val, models_dir
    )
    results.append(result)

# %%
# Summarise comparison
results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)
print(results_df[["name", "pr_auc", "roc_auc", "f1", "precision", "recall", "train_time_s"]].to_string(index=False))

# Save comparison table for the report
results_df.to_csv(PROJECT_ROOT / "reports" / "baseline_results.csv", index=False)
print(f"\nSaved comparison to: reports/baseline_results.csv")
# %%
