"""Step 2: Apply preprocessing using the project's preprocessing module."""
import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from zenml import step

from src.preprocessing import prepare_dataset, NUMERIC_FEATURES, BINARY_FEATURES


@step
def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    use_recipient_frequency: bool = False,
) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
    Annotated[StandardScaler, "fitted_scaler"],
]:
    """Apply feature engineering and stratified train/test split."""
    X, y = prepare_dataset(df, use_recipient_frequency=use_recipient_frequency)
    
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    scaler = StandardScaler()
    X_train_scaled = X_train_df.copy()
    X_test_scaled = X_test_df.copy()
    
    X_train_scaled[NUMERIC_FEATURES] = scaler.fit_transform(X_train_df[NUMERIC_FEATURES])
    X_test_scaled[NUMERIC_FEATURES] = scaler.transform(X_test_df[NUMERIC_FEATURES])
    
    print(f"Train shape: {X_train_scaled.shape}, fraud rate: {y_train.mean():.4%}")
    print(f"Test shape:  {X_test_scaled.shape}, fraud rate: {y_test.mean():.4%}")
    
    return (
        X_train_scaled.values,
        X_test_scaled.values,
        y_train.values,
        y_test.values,
        scaler,
    )
