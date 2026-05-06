"""Step 4: Evaluate on held-out test set with fraud-specific metrics."""
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Annotated[float, "test_pr_auc"]:
    """Compute classification metrics with focus on PR-AUC."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    for name, value in metrics.items():
        mlflow.log_metric(f"test_{name}", value)
    mlflow.log_metric("test_true_positives", tp)
    mlflow.log_metric("test_false_negatives", fn)
    mlflow.log_metric("test_false_positives", fp)
    
    print("=" * 60)
    print("TEST SET PERFORMANCE")
    print("=" * 60)
    for name, value in metrics.items():
        print(f"  {name:12s}: {value:.4f}")
    print(f"  Missed fraud (FN): {fn} | False alarms (FP): {fp}")
    print("=" * 60)
    
    return metrics["pr_auc"]
