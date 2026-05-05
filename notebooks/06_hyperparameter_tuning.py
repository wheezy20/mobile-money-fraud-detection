"""
Section D Phase 4: Hyperparameter tuning of Random Forest.

Strategy:
1. Use RandomizedSearchCV with 30 parameter combinations
2. Tune on a stratified 25% subsample of training data (for speed)
3. Use 3-fold cross-validation
4. Score by average_precision (PR-AUC) — the primary metric per Section A
5. Refit best parameters on the full training set
6. Evaluate the tuned model on the validation set
7. Compare against the baseline RF

Expected runtime: ~30-45 minutes total.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



# %%
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mobile_money_fraud"
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# %%
# Load training data and split out a 25% stratified tuning subsample
splits_dir = PROJECT_ROOT / "data" / "processed" / "splits"

X_train_full = pd.read_csv(splits_dir / "X_train.csv")
y_train_full = pd.read_csv(splits_dir / "y_train.csv").iloc[:, 0]
X_val = pd.read_csv(splits_dir / "X_val.csv")
y_val = pd.read_csv(splits_dir / "y_val.csv").iloc[:, 0]

print(f"Full training set: {X_train_full.shape}, fraud rate: {y_train_full.mean()*100:.4f}%")

# Take a stratified 25% subsample for tuning
X_tune, _, y_tune, _ = train_test_split(
    X_train_full,
    y_train_full,
    train_size=0.25,
    stratify=y_train_full,
    random_state=42,
)

print(f"Tuning subsample:  {X_tune.shape}, fraud rate: {y_tune.mean()*100:.4f}%")
print(f"Tuning fraud cases: {y_tune.sum():,}")

# %%
# Define the parameter search space
param_distributions = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [8, 12, 16, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2", 0.5],
    "class_weight":      ["balanced", "balanced_subsample"],
}

# Compute total search space size for context
total_combinations = 1
for v in param_distributions.values():
    total_combinations *= len(v)
print(f"Full parameter grid: {total_combinations} combinations")
print(f"Random search will sample 30 of these")

# %%
# Run RandomizedSearchCV
print("\nStarting hyperparameter search...")
print("This will take ~30-45 minutes.")
print("=" * 60)

start = time.time()

base_estimator = RandomForestClassifier(random_state=42, n_jobs=-1)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_estimator,
    param_distributions=param_distributions,
    n_iter=30,
    cv=cv,
    scoring="average_precision",  # PR-AUC, our primary metric
    n_jobs=-1,
    random_state=42,
    verbose=2,
    return_train_score=True,
)

search.fit(X_tune, y_tune)

elapsed = time.time() - start
print(f"\n{'=' * 60}")
print(f"Search complete in {elapsed/60:.1f} minutes")
print(f"Best PR-AUC (CV mean): {search.best_score_:.4f}")
print(f"Best parameters:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

# %%
# Save full CV results for inspection
cv_results = pd.DataFrame(search.cv_results_)
cv_results = cv_results[[
    "param_n_estimators", "param_max_depth", "param_min_samples_split",
    "param_min_samples_leaf", "param_max_features", "param_class_weight",
    "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score",
]].sort_values("rank_test_score")

print("\nTop 10 parameter combinations by CV PR-AUC:")
print(cv_results.head(10).to_string(index=False))

cv_results.to_csv(PROJECT_ROOT / "reports" / "tuning_cv_results.csv", index=False)
print(f"\nFull CV results saved to: reports/tuning_cv_results.csv")

# %%
# Refit the best model on the FULL training set
print("\nRefitting best model on full training set...")
best_params = search.best_params_

start = time.time()
tuned_rf = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1,
)
tuned_rf.fit(X_train_full, y_train_full)
elapsed = time.time() - start

print(f"Refit complete in {elapsed:.1f} seconds")

# Save the tuned model
tuned_path = PROJECT_ROOT / "models" / "tuned_rf.pkl"
joblib.dump(tuned_rf, tuned_path)
print(f"Saved: {tuned_path}")

# %%
# Evaluate the tuned model against the baseline RF
print("\nEvaluating tuned model on validation set...")

# Tuned predictions
y_proba_tuned = tuned_rf.predict_proba(X_val)[:, 1]
y_pred_tuned = (y_proba_tuned >= 0.5).astype(int)

# Load baseline RF for comparison
baseline_rf = joblib.load(PROJECT_ROOT / "models" / "baseline_rf.pkl")
y_proba_baseline = baseline_rf.predict_proba(X_val)[:, 1]
y_pred_baseline = (y_proba_baseline >= 0.5).astype(int)


def compute_metrics(y_true, y_pred, y_proba, label):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "model": label,
        "pr_auc":     average_precision_score(y_true, y_proba),
        "roc_auc":    roc_auc_score(y_true, y_proba),
        "f1":         f1_score(y_true, y_pred),
        "precision":  precision_score(y_true, y_pred, zero_division=0),
        "recall":     recall_score(y_true, y_pred),
        "tp": cm[1, 1], "fp": cm[0, 1], "fn": cm[1, 0], "tn": cm[0, 0],
    }


comparison = pd.DataFrame([
    compute_metrics(y_val, y_pred_baseline, y_proba_baseline, "RF Baseline"),
    compute_metrics(y_val, y_pred_tuned, y_proba_tuned, "RF Tuned"),
])

print("\n" + "=" * 70)
print("BASELINE vs TUNED — VALIDATION SET")
print("=" * 70)
pd.set_option("display.float_format", "{:.4f}".format)
print(comparison.to_string(index=False))

# Lift analysis
baseline_pr_auc = comparison.iloc[0]["pr_auc"]
tuned_pr_auc = comparison.iloc[1]["pr_auc"]
lift = tuned_pr_auc - baseline_pr_auc
relative_lift = (lift / baseline_pr_auc) * 100

print(f"\nPR-AUC improvement: {lift:+.4f} ({relative_lift:+.2f}% relative)")

comparison.to_csv(PROJECT_ROOT / "reports" / "tuning_comparison.csv", index=False)
print("Saved: reports/tuning_comparison.csv")
# %%
