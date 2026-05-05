"""
Section D Phase 5b: Final test-set evaluation.

The test set (15% of filtered data, 415,562 rows, 1,232 fraud cases) has
been held out since Phase 1. This script touches it ONCE to produce the
final reportable performance numbers.

The selected model is the tuned Random Forest with 9 features (no
recipient_count_24h, per the Phase 5 finding that this feature provided
zero lift).

Outputs:
    - reports/final_test_results.csv
    - reports/figures/fig_10_final_pr_curve.png
    - reports/figures/fig_11_final_confusion_matrix.png
"""


#%%

import warnings
warnings.filterwarnings("ignore", category=UserWarning)




# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mobile_money_fraud"
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
)

sns.set_style("whitegrid")

# %%
# Load the tuned model and the test set
print("Loading tuned Random Forest model...")
model = joblib.load(PROJECT_ROOT / "models" / "tuned_rf.pkl")
print(f"Model: {type(model).__name__}")
print(f"Hyperparameters: {model.get_params()}")

# %%
# Load test data
splits_dir = PROJECT_ROOT / "data" / "processed" / "splits"
X_test = pd.read_csv(splits_dir / "X_test.csv")
y_test = pd.read_csv(splits_dir / "y_test.csv").iloc[:, 0]

print(f"\nTest set: {X_test.shape}")
print(f"Test fraud cases: {y_test.sum():,} ({y_test.mean()*100:.4f}%)")

# %%
# Generate predictions and probabilities
print("\nGenerating predictions on test set...")
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

print("Predictions complete.")

# %%
# Compute the full metrics suite
print("\n" + "=" * 70)
print("FINAL TEST-SET PERFORMANCE")
print("=" * 70)

cm = confusion_matrix(y_test, y_pred)
tp, fp = cm[1, 1], cm[0, 1]
fn, tn = cm[1, 0], cm[0, 0]

# Standard metrics
pr_auc    = average_precision_score(y_test, y_proba)
roc_auc   = roc_auc_score(y_test, y_proba)
f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred)

# Recall at 90% precision (per Section A success metric)
prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_test, y_proba)
valid = prec_curve[:-1] >= 0.90
if valid.any():
    valid_recalls = rec_curve[:-1][valid]
    valid_thresholds = thresh_curve[valid]
    best_idx = np.argmax(valid_recalls)
    recall_at_90 = valid_recalls[best_idx]
    threshold_at_90 = valid_thresholds[best_idx]
else:
    recall_at_90 = 0.0
    threshold_at_90 = None

# Cost-weighted (FN=5x, FP=1x as defined in Phase 3)
COST_FN, COST_FP = 5.0, 1.0
total_cost = fn * COST_FN + fp * COST_FP

# Print results
print(f"PR-AUC:                  {pr_auc:.4f}")
print(f"ROC-AUC:                 {roc_auc:.4f}")
print(f"F1 (default threshold):  {f1:.4f}")
print(f"Precision:               {precision:.4f}")
print(f"Recall:                  {recall:.4f}")
print(f"Recall @ 90% precision:  {recall_at_90:.4f}")
print(f"Threshold @ 90% prec:    {threshold_at_90:.4f}" if threshold_at_90 is not None else "Threshold @ 90% prec: N/A")
print(f"Cost-weighted (5:1):     {total_cost:,.0f}")
print()
print("Confusion matrix:")
print(f"  TN={tn:>7,}  FP={fp:>5,}")
print(f"  FN={fn:>7,}  TP={tp:>5,}")

# %%
# Compare to validation set numbers (sanity check for overfitting)
print("\n" + "=" * 70)
print("VALIDATION vs TEST")
print("=" * 70)

# Load validation predictions to compute the same metrics
X_val = pd.read_csv(splits_dir / "X_val.csv")
y_val = pd.read_csv(splits_dir / "y_val.csv").iloc[:, 0]

y_proba_val = model.predict_proba(X_val)[:, 1]
y_pred_val = (y_proba_val >= 0.5).astype(int)

val_pr_auc = average_precision_score(y_val, y_proba_val)
val_f1 = f1_score(y_val, y_pred_val)
val_precision = precision_score(y_val, y_pred_val, zero_division=0)
val_recall = recall_score(y_val, y_pred_val)

comparison = pd.DataFrame({
    "Metric": ["PR-AUC", "F1", "Precision", "Recall"],
    "Validation": [val_pr_auc, val_f1, val_precision, val_recall],
    "Test":       [pr_auc, f1, precision, recall],
})
comparison["Δ (Test − Val)"] = comparison["Test"] - comparison["Validation"]

pd.set_option("display.float_format", "{:.4f}".format)
print(comparison.to_string(index=False))

# %%
# Save the final results table
results_dict = {
    "model": "Tuned Random Forest (9 features)",
    "n_test_samples": len(y_test),
    "n_test_fraud": int(y_test.sum()),
    "pr_auc": pr_auc,
    "roc_auc": roc_auc,
    "f1_default_threshold": f1,
    "precision_default": precision,
    "recall_default": recall,
    "recall_at_90_precision": recall_at_90,
    "threshold_at_90_precision": threshold_at_90 if threshold_at_90 is not None else np.nan,
    "tp": int(tp),
    "fp": int(fp),
    "fn": int(fn),
    "tn": int(tn),
    "cost_weighted_5_to_1": total_cost,
}

results_df = pd.DataFrame([results_dict])
results_df.to_csv(PROJECT_ROOT / "reports" / "final_test_results.csv", index=False)
print(f"\nSaved: reports/final_test_results.csv")

# %%
# Figure 10: Final PR curve on the test set
print("\nPlotting final PR curve...")
fig, ax = plt.subplots(figsize=(9, 6))

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
ax.plot(recall_curve, precision_curve, color="forestgreen", linewidth=2.5,
        label=f"Tuned RF (PR-AUC = {pr_auc:.4f})")

# Random baseline
ax.axhline(y=y_test.mean(), linestyle="--", color="gray", alpha=0.5,
           label=f"Random baseline ({y_test.mean()*100:.2f}%)")

# Mark the operational point: recall at 90% precision
ax.scatter([recall_at_90], [0.90], color="crimson", s=120, zorder=5,
           label=f"Operational point: {recall_at_90:.3f} recall @ 90% precision")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Final Test-Set Precision-Recall Curve\nTuned Random Forest")
ax.legend(loc="lower left")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "fig_10_final_pr_curve.png",
            dpi=150, bbox_inches="tight")
plt.show()

# %%
# Figure 11: Final confusion matrix
print("Plotting final confusion matrix...")
fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=ax,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            cbar=False, annot_kws={"size": 14})
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title(f"Final Test-Set Confusion Matrix (threshold = 0.5)\n"
             f"PR-AUC = {pr_auc:.4f}, F1 = {f1:.4f}", fontsize=12)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "figures" / "fig_11_final_confusion_matrix.png",
            dpi=150, bbox_inches="tight")
plt.show()

print(f"\nSaved: fig_10_final_pr_curve.png, fig_11_final_confusion_matrix.png")
print("\nPhase 5b complete. Test set has been touched once.")
# %%
