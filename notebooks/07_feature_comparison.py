"""
Section D Phase 5: Recipient frequency feature comparison study (corrected).

Trains the tuned Random Forest configuration twice on UNSCALED features:
1. Without recipient_count_24h (the 9-feature baseline)
2. With recipient_count_24h    (the 10-feature variant)

NOTE: An earlier version of this script applied StandardScaler before RF
training. Tree-based models don't require scaling (splits are threshold-
based, not distance-based) and the scaling interacted with class_weight=
'balanced_subsample' to produce subtly different tree structures. This
corrected version uses unscaled features, consistent with Phase 4.

Outputs:
    - models/tuned_rf_with_recipient.pkl
    - reports/feature_comparison.csv
    - reports/figures/fig_09_feature_importance.png
"""



#%%

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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.preprocessing import (
    engineer_features,
    add_recipient_frequency,
)

sns.set_style("whitegrid")

# %%
# Tuned hyperparameters from Phase 4
TUNED_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "log2",
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}

# %%
# Load filtered dataset and engineer features WITH recipient_count_24h
print("Loading filtered dataset...")
df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "paysim_filtered.csv")
print(f"Loaded: {df.shape}")

print("\nEngineering features (with rolling-count step)...")
start = time.time()
df_with_recip = engineer_features(df)
df_with_recip = add_recipient_frequency(df_with_recip, window_hours=24)
print(f"Done in {time.time() - start:.1f}s")

# %%
# Build feature matrix WITH the extra feature
FEATURES_WITH_RECIP = [
    "amount_log", "drainage_ratio", "error_orig", "error_dest",
    "hour_of_day", "oldbalanceOrg", "oldbalanceDest",
    "type_encoded", "is_unusual_hour",
    "recipient_count_24h",
]

X_with = df_with_recip[FEATURES_WITH_RECIP].copy()
y_with = df_with_recip["isFraud"].copy()

print(f"\nFeature matrix WITH recipient_count_24h: {X_with.shape}")

# %%
# Same stratified split as Phase 1 (random_state=42)
print("\nSplitting...")
X_temp, X_test_with, y_temp, y_test_with = train_test_split(
    X_with, y_with, test_size=0.15, stratify=y_with, random_state=42,
)
X_train_with, X_val_with, y_train_with, y_val_with = train_test_split(
    X_temp, y_temp, test_size=15/85, stratify=y_temp, random_state=42,
)

# Drop the new column to create the without-recipient version
X_train_without = X_train_with.drop(columns=["recipient_count_24h"])
X_val_without = X_val_with.drop(columns=["recipient_count_24h"])

print(f"Without recipient: {X_train_without.shape}")
print(f"With recipient:    {X_train_with.shape}")

# %%
# Use UNSCALED features — RF doesn't require scaling
print("\nUsing unscaled features (consistent with Phase 4 RF training).")

# %%
# Train both models on UNSCALED features
print("\nTraining tuned RF WITHOUT recipient_count_24h...")
start = time.time()
model_without = RandomForestClassifier(**TUNED_PARAMS)
model_without.fit(X_train_without, y_train_with)
elapsed_without = time.time() - start
print(f"Done in {elapsed_without:.1f}s")

print("\nTraining tuned RF WITH recipient_count_24h...")
start = time.time()
model_with = RandomForestClassifier(**TUNED_PARAMS)
model_with.fit(X_train_with, y_train_with)
elapsed_with = time.time() - start
print(f"Done in {elapsed_with:.1f}s")

# Save the with-recipient model
joblib.dump(model_with, PROJECT_ROOT / "models" / "tuned_rf_with_recipient.pkl")
print(f"\nSaved: models/tuned_rf_with_recipient.pkl")

# %%
# Evaluate both on the validation set
def evaluate(model, X_val, y_val, label):
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    
    return {
        "model": label,
        "pr_auc":     average_precision_score(y_val, y_proba),
        "roc_auc":    roc_auc_score(y_val, y_proba),
        "f1":         f1_score(y_val, y_pred),
        "precision":  precision_score(y_val, y_pred, zero_division=0),
        "recall":     recall_score(y_val, y_pred),
        "tp": int(cm[1, 1]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tn": int(cm[0, 0]),
    }, y_proba


results_without, _ = evaluate(
    model_without, X_val_without, y_val_with, "Tuned RF (9 features)"
)
results_with, _ = evaluate(
    model_with, X_val_with, y_val_with, "Tuned RF (10 features)"
)

comparison = pd.DataFrame([results_without, results_with])
print("\n" + "=" * 70)
print("FEATURE COMPARISON STUDY — VALIDATION SET")
print("=" * 70)
pd.set_option("display.float_format", "{:.4f}".format)
print(comparison.to_string(index=False))

# Lift
pr_lift = results_with["pr_auc"] - results_without["pr_auc"]
f1_lift = results_with["f1"] - results_without["f1"]
fp_change = results_with["fp"] - results_without["fp"]

print(f"\nPR-AUC change: {pr_lift:+.5f}")
print(f"F1 change:     {f1_lift:+.5f}")
print(f"FP change:     {fp_change:+d}  ({results_without['fp']} -> {results_with['fp']})")

comparison.to_csv(PROJECT_ROOT / "reports" / "feature_comparison.csv", index=False)
print(f"\nSaved: reports/feature_comparison.csv")

# %%
# Feature importance — does recipient_count_24h actually rank in the model?
print("\nFeature importance ranking (with recipient_count_24h):")
importances = pd.DataFrame({
    "feature": FEATURES_WITH_RECIP,
    "importance": model_with.feature_importances_,
}).sort_values("importance", ascending=False)
print(importances.to_string(index=False))

# %%
# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))

colors = ["crimson" if f == "recipient_count_24h" else "steelblue" 
          for f in importances["feature"]]

ax.barh(importances["feature"], importances["importance"], color=colors)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Random Forest Feature Importance\nrecipient_count_24h shown in red")

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "fig_09_feature_importance.png",
            dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSaved: fig_09_feature_importance.png")
# %%
