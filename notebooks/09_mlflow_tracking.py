"""
Section E Phase 1: Log all trained models to MLflow.

Loads each of the 5 saved .pkl models, computes their validation-set
metrics (using already-saved CSVs for consistency), and logs each as a
separate MLflow run with parameters, metrics, and the model artifact.

After running, view the results with:
    mlflow ui --port 5000

Then open: http://localhost:5000
"""

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mobile_money_fraud"
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# %%
# Configure MLflow to use a local file-based tracking store.
# This stores all runs in mlruns/ at the project root.
mlflow.set_tracking_uri(f"file://{PROJECT_ROOT}/mlruns")
mlflow.set_experiment("mobile_money_fraud_detection")

print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment:   mobile_money_fraud_detection")

# %%
# Load validation data once
splits_dir = PROJECT_ROOT / "data" / "processed" / "splits"
X_val = pd.read_csv(splits_dir / "X_val.csv")
y_val = pd.read_csv(splits_dir / "y_val.csv").iloc[:, 0]
print(f"Validation set: {X_val.shape}, fraud rate: {y_val.mean()*100:.4f}%")

# %%
# Helper: compute the standard metric suite for a given model
def evaluate_for_logging(model, X_val, y_val):
    """Compute metrics dictionary suitable for mlflow.log_metrics()."""
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred)

    return {
        "pr_auc":    average_precision_score(y_val, y_proba),
        "roc_auc":   roc_auc_score(y_val, y_proba),
        "f1":        f1_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall":    recall_score(y_val, y_pred),
        "tp":        int(cm[1, 1]),
        "fp":        int(cm[0, 1]),
        "fn":        int(cm[1, 0]),
        "tn":        int(cm[0, 0]),
    }


def log_model_run(
    model_path: Path,
    run_name: str,
    extra_params: dict,
    extra_tags: dict,
    is_xgboost: bool = False,
    X_val_to_use=None,
):
    """Log one model as an MLflow run."""
    if X_val_to_use is None:
        X_val_to_use = X_val
    
    print(f"\n{'='*60}")
    print(f"Logging: {run_name}")
    print(f"{'='*60}")
    
    model = joblib.load(model_path)
    metrics = evaluate_for_logging(model, X_val_to_use, y_val)
    params = {**model.get_params(), **extra_params}
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for tag_key, tag_value in extra_tags.items():
            mlflow.set_tag(tag_key, tag_value)

        # Log the model artifact
        if is_xgboost:
            mlflow.xgboost.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(model, name="model")

        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Logged to MLflow.")


# %%
# Run 1: Logistic Regression baseline
log_model_run(
    model_path=PROJECT_ROOT / "models" / "baseline_logreg.pkl",
    run_name="baseline_logreg",
    extra_params={"feature_count": 9, "tuned": False, "phase": "2_baseline"},
    extra_tags={
        "model_family": "linear",
        "purpose": "interpretable baseline",
        "section": "D_phase_2",
    },
)

# %%
# Run 2: Random Forest baseline
log_model_run(
    model_path=PROJECT_ROOT / "models" / "baseline_rf.pkl",
    run_name="baseline_rf",
    extra_params={"feature_count": 9, "tuned": False, "phase": "2_baseline"},
    extra_tags={
        "model_family": "bagged_trees",
        "purpose": "non-linear baseline",
        "section": "D_phase_2",
    },
)

# %%
# Run 3: XGBoost baseline
log_model_run(
    model_path=PROJECT_ROOT / "models" / "baseline_xgb.pkl",
    run_name="baseline_xgb",
    extra_params={"feature_count": 9, "tuned": False, "phase": "2_baseline"},
    extra_tags={
        "model_family": "gradient_boosted",
        "purpose": "boosted baseline",
        "section": "D_phase_2",
    },
    is_xgboost=True,
)

# %%
# Run 4: Tuned Random Forest (the champion)
log_model_run(
    model_path=PROJECT_ROOT / "models" / "tuned_rf.pkl",
    run_name="tuned_rf",
    extra_params={
        "feature_count": 9,
        "tuned": True,
        "phase": "4_hyperparameter_tuning",
        "tuning_method": "RandomizedSearchCV",
        "tuning_n_iter": 30,
        "tuning_cv_folds": 3,
        "tuning_subsample_pct": 25,
    },
    extra_tags={
        "model_family": "bagged_trees",
        "purpose": "champion_model",
        "section": "D_phase_4",
    },
)

# %%
# Run 5: Tuned RF with recipient_count_24h (the failed feature experiment)
# This one needs a different X_val because the feature set is different
splits_with_recip_dir = PROJECT_ROOT / "data" / "processed" / "splits_with_recipient"

# Check if the with-recipient splits were saved separately
if not (splits_with_recip_dir / "X_val.csv").exists():
    # We need to regenerate the with-recipient validation set
    # Use the same logic as Phase 5
    print("\nRegenerating with-recipient validation set for logging...")
    from src.preprocessing import engineer_features, add_recipient_frequency
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "paysim_filtered.csv")
    df_with = engineer_features(df)
    df_with = add_recipient_frequency(df_with, window_hours=24)
    
    FEATURES_WITH_RECIP = [
        "amount_log", "drainage_ratio", "error_orig", "error_dest",
        "hour_of_day", "oldbalanceOrg", "oldbalanceDest",
        "type_encoded", "is_unusual_hour", "recipient_count_24h",
    ]
    
    X_with = df_with[FEATURES_WITH_RECIP].copy()
    y_with = df_with["isFraud"].copy()
    
    X_temp, X_test_recip, y_temp, y_test_recip = train_test_split(
        X_with, y_with, test_size=0.15, stratify=y_with, random_state=42,
    )
    X_train_recip, X_val_recip, y_train_recip, y_val_recip = train_test_split(
        X_temp, y_temp, test_size=15/85, stratify=y_temp, random_state=42,
    )
else:
    X_val_recip = pd.read_csv(splits_with_recip_dir / "X_val.csv")

log_model_run(
    model_path=PROJECT_ROOT / "models" / "tuned_rf_with_recipient.pkl",
    run_name="tuned_rf_with_recipient",
    extra_params={
        "feature_count": 10,
        "tuned": True,
        "phase": "5_feature_comparison",
        "additional_feature": "recipient_count_24h",
    },
    extra_tags={
        "model_family": "bagged_trees",
        "purpose": "feature_ablation_study",
        "section": "D_phase_5",
    },
    X_val_to_use=X_val_recip,
)

# %%
print("\n" + "="*60)
print("ALL 5 MODELS LOGGED TO MLFLOW")
print("="*60)
print(f"\nView the runs:")
print(f"  cd {PROJECT_ROOT}")
print(f"  mlflow ui --port 5000")
print(f"  Open: http://localhost:5000")
# %%
