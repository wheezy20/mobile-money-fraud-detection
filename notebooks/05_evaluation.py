"""
Section D Phase 3: Detailed evaluation of baseline models.

Loads the three trained baseline models from Phase 2 and produces:
- Precision-Recall curves (primary metric per Section A)
- ROC curves (secondary, for completeness)
- Confusion matrices at default threshold
- Threshold-tuning analysis
- Cost-weighted evaluation
- Recall at fixed precision (90%) per Section A's success metrics

Outputs:
    - reports/figures/fig_05_pr_curves.png
    - reports/figures/fig_06_roc_curves.png
    - reports/figures/fig_07_confusion_matrices.png
    - reports/figures/fig_08_threshold_analysis.png
    - reports/evaluation_results.csv
"""

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
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sns.set_style("whitegrid")

# %%
# Load models and validation data
print("Loading models and validation data...")

models_dir = PROJECT_ROOT / "models"
splits_dir = PROJECT_ROOT / "data" / "processed" / "splits"
figures_dir = PROJECT_ROOT / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

models = {
    "Logistic Regression": joblib.load(models_dir / "baseline_logreg.pkl"),
    "Random Forest":       joblib.load(models_dir / "baseline_rf.pkl"),
    "XGBoost":             joblib.load(models_dir / "baseline_xgb.pkl"),
}

X_val = pd.read_csv(splits_dir / "X_val.csv")
y_val = pd.read_csv(splits_dir / "y_val.csv").iloc[:, 0]

print(f"Validation set: {X_val.shape}, fraud rate: {y_val.mean() * 100:.4f}%")

# %%
# Compute predicted probabilities for all three models, once
print("Computing predicted probabilities...")
probas = {}
for name, model in models.items():
    probas[name] = model.predict_proba(X_val)[:, 1]
    print(f"  {name}: predicted probabilities shape {probas[name].shape}")

# %%
# Figure 5: Precision-Recall curves (the primary visualisation per Section A)
print("Plotting Precision-Recall curves...")
fig, ax = plt.subplots(figsize=(9, 6))

colors = {"Logistic Regression": "steelblue", "Random Forest": "forestgreen", "XGBoost": "crimson"}

for name, y_proba in probas.items():
    precision, recall, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)
    ax.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})",
            color=colors[name], linewidth=2)

# Random baseline: precision = fraud rate (a model predicting at random)
random_baseline = y_val.mean()
ax.axhline(y=random_baseline, linestyle="--", color="gray", alpha=0.5,
           label=f"Random baseline ({random_baseline*100:.2f}%)")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves on Validation Set")
ax.legend(loc="lower left")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)

plt.tight_layout()
plt.savefig(figures_dir / "fig_05_pr_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: fig_05_pr_curves.png")

# %%
# Figure 6: ROC curves (secondary, for completeness — shows why ROC is misleading here)
print("Plotting ROC curves...")
fig, ax = plt.subplots(figsize=(9, 6))

for name, y_proba in probas.items():
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = roc_auc_score(y_val, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc_auc:.4f})",
            color=colors[name], linewidth=2)

# Diagonal: random classifier
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Random classifier")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves on Validation Set\n(Note: high values can be misleading under severe class imbalance)")
ax.legend(loc="lower right")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)

plt.tight_layout()
plt.savefig(figures_dir / "fig_06_roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: fig_06_roc_curves.png")

# %%
# Figure 7: Confusion matrices side by side at default threshold = 0.5
print("Plotting confusion matrices...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, y_proba) in zip(axes, probas.items()):
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    
    sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=ax,
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"],
                cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name}\n(threshold = 0.5)")

plt.tight_layout()
plt.savefig(figures_dir / "fig_07_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: fig_07_confusion_matrices.png")

# %%
# Figure 8: Threshold analysis - how do precision, recall, and F1 change with threshold?
print("Plotting threshold analysis...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

thresholds = np.linspace(0.01, 0.99, 99)

for ax, (name, y_proba) in zip(axes, probas.items()):
    precs, recs, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precs.append(precision_score(y_val, y_pred, zero_division=0))
        recs.append(recall_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred))
    
    ax.plot(thresholds, precs, label="Precision", color="steelblue", linewidth=2)
    ax.plot(thresholds, recs, label="Recall", color="forestgreen", linewidth=2)
    ax.plot(thresholds, f1s, label="F1", color="crimson", linewidth=2)
    
    # Mark the default threshold
    ax.axvline(x=0.5, linestyle="--", color="gray", alpha=0.5, label="Default (0.5)")
    
    ax.set_xlabel("Threshold")
    ax.set_title(name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9)

axes[0].set_ylabel("Score")
plt.tight_layout()
plt.savefig(figures_dir / "fig_08_threshold_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: fig_08_threshold_analysis.png")

# %%
# Compute richer metrics table
def recall_at_precision(y_true, y_proba, target_precision=0.90):
    """Find the maximum recall achievable at or above the target precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # The last point of precision_recall_curve is (precision=1, recall=0); ignore it
    valid = precision[:-1] >= target_precision
    if not valid.any():
        return 0.0, None
    valid_recalls = recall[:-1][valid]
    valid_thresholds = thresholds[valid]
    best_idx = np.argmax(valid_recalls)
    return valid_recalls[best_idx], valid_thresholds[best_idx]


def best_f1_threshold(y_true, y_proba):
    """Find the threshold that maximises F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    return f1_scores[best_idx], thresholds[best_idx]


# %%
# Cost-weighted confusion matrix per Section A
# Assumption (documented for the report): missed fraud (FN) is 5x as costly as
# a false alarm (FP), in a context where one is a real loss and the other is
# a customer-friction cost.
COST_FN = 5.0
COST_FP = 1.0


def cost_weighted_score(cm, cost_fn=COST_FN, cost_fp=COST_FP):
    """Lower is better. Counts misclassification cost from a confusion matrix."""
    fn = cm[1, 0]
    fp = cm[0, 1]
    return fn * cost_fn + fp * cost_fp


# %%
# Build the full evaluation table
print("\nBuilding evaluation table...")
results = []
for name, y_proba in probas.items():
    # Default threshold metrics
    y_pred_default = (y_proba >= 0.5).astype(int)
    cm_default = confusion_matrix(y_val, y_pred_default)
    
    # Best-F1 threshold
    best_f1, best_t = best_f1_threshold(y_val, y_proba)
    y_pred_optimal = (y_proba >= best_t).astype(int)
    cm_optimal = confusion_matrix(y_val, y_pred_optimal)
    
    # Recall at 90% precision (per Section A)
    rec_at_90, t_at_90 = recall_at_precision(y_val, y_proba, target_precision=0.90)
    
    results.append({
        "model": name,
        "pr_auc": average_precision_score(y_val, y_proba),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "f1_default_threshold": f1_score(y_val, y_pred_default),
        "best_f1": best_f1,
        "best_f1_threshold": best_t,
        "recall_at_90_precision": rec_at_90,
        "threshold_at_90_precision": t_at_90 if t_at_90 is not None else np.nan,
        "cost_default_threshold": cost_weighted_score(cm_default),
        "cost_optimal_threshold": cost_weighted_score(cm_optimal),
    })

results_df = pd.DataFrame(results)
print("\n" + "=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)
pd.set_option("display.float_format", "{:.4f}".format)
print(results_df.to_string(index=False))

results_df.to_csv(PROJECT_ROOT / "reports" / "evaluation_results.csv", index=False)
print(f"\nSaved: reports/evaluation_results.csv")

# %%
# Final summary print
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
for r in results:
    print(f"\n{r['model']}:")
    print(f"  PR-AUC:                     {r['pr_auc']:.4f}")
    print(f"  F1 at default threshold:    {r['f1_default_threshold']:.4f}")
    print(f"  Best F1 (optimal threshold): {r['best_f1']:.4f} at t={r['best_f1_threshold']:.3f}")
    print(f"  Recall @ 90% precision:     {r['recall_at_90_precision']:.4f}")
    print(f"  Threshold @ 90% precision:  {r['threshold_at_90_precision']:.4f}")
    print(f"  Cost (default threshold):   {r['cost_default_threshold']:,.0f}")
    print(f"  Cost (optimal threshold):   {r['cost_optimal_threshold']:,.0f}")
# %%
