"""Step 3: Train Random Forest with MLflow autologging."""
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_split: int = 2,
    class_weight: str = "balanced",
) -> Annotated[RandomForestClassifier, "tuned_rf_model"]:
    """Train Random Forest with hyperparameters from Section D tuning."""
    mlflow.sklearn.autolog()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    print(f"Trained RF: {n_estimators} trees, max_depth={max_depth}")
    return model
