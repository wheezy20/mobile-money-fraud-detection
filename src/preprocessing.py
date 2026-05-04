"""
Mobile money fraud detection: preprocessing module.

This module provides reproducible feature engineering and preprocessing
for the PaySim mobile money dataset. It is designed to be imported by
training scripts, ZenML pipelines, and the FastAPI inference server.

Design principles:
- Pure functions where possible: same input always produces same output
- Stateless transformations are functions; stateful ones (scaling) are
  sklearn Transformers
- Backward-looking only: no feature uses information fromall clear, let's run it the future,
  so the same code works at training time and at inference time

Author: Eyram
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------

# Transaction types that contain fraud. Established in Section B EDA.
FRAUD_RELEVANT_TYPES = ["CASH_OUT", "TRANSFER"]

# Hours considered "unusual" — fraud rate spikes during these hours.
# Established in Section B EDA: rates of 19-58% vs baseline of 0.30%.
UNUSUAL_HOURS = {0, 1, 2, 3, 4, 5, 6}

# Columns dropped from the final feature set.
COLUMNS_TO_DROP = [
    "step",            # replaced by hour_of_day / is_unusual_hour
    "newbalanceOrig",  # captured in error_orig
    "newbalanceDest",  # captured in error_dest
    "nameOrig",        # near-unique cardinality, no signal
    "nameDest",        # used only for recipient_count_24h, then dropped
    "isFlaggedFraud",  # label leakage (Section A discussion)
]

# Final feature set, in the order the model expects.
NUMERIC_FEATURES = [
    "amount_log",
    "drainage_ratio",
    "error_orig",
    "error_dest",
    "hour_of_day",
    "oldbalanceOrg",
    "oldbalanceDest",
]

BINARY_FEATURES = [
    "type_encoded",
    "is_unusual_hour",
]


# -----------------------------------------------------------------------
# STAGE 1: FILTER TO FRAUD-RELEVANT TRANSACTION TYPES
# -----------------------------------------------------------------------

def filter_fraud_relevant_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict the dataset to transaction types where fraud actually occurs.

    Section B EDA established that fraud is exclusively present in
    CASH_OUT and TRANSFER transactions. Filtering reduces the dataset
    by ~56% with zero loss of positive signal.

    Args:
        df: Raw PaySim dataframe.

    Returns:
        Filtered dataframe (CASH_OUT and TRANSFER rows only).
    """
    mask = df["type"].isin(FRAUD_RELEVANT_TYPES)
    return df.loc[mask].copy()


# -----------------------------------------------------------------------
# STAGE 2: STATELESS FEATURE ENGINEERING
# -----------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features identified in Section B EDA.

    All features here are stateless: they depend only on the row being
    processed, not on any global state from training data. This means
    the same function works at both training and inference time.

    Engineered features:
        - type_encoded: 1 if TRANSFER else 0 (binary)
        - amount_log: log1p(amount) — handles heavy right skew
        - drainage_ratio: amount / (oldbalanceOrg + 1)
        - error_orig: amount - (oldbalanceOrg - newbalanceOrig)
        - error_dest: amount - (newbalanceDest - oldbalanceDest)
        - hour_of_day: step % 24
        - is_unusual_hour: 1 if hour in {0..6} else 0

    Args:
        df: Filtered dataframe.

    Returns:
        New dataframe with engineered columns added.
    """
    df = df.copy()

    df["type_encoded"] = (df["type"] == "TRANSFER").astype(int)
    df["amount_log"] = np.log1p(df["amount"])
    df["drainage_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["error_orig"] = df["amount"] - (df["oldbalanceOrg"] - df["newbalanceOrig"])
    df["error_dest"] = df["amount"] - (df["newbalanceDest"] - df["oldbalanceDest"])
    df["hour_of_day"] = df["step"] % 24
    df["is_unusual_hour"] = df["hour_of_day"].isin(UNUSUAL_HOURS).astype(int)

    return df


# -----------------------------------------------------------------------
# STAGE 3: OPTIONAL — RECIPIENT FREQUENCY (BACKWARD-LOOKING)
# -----------------------------------------------------------------------

def add_recipient_frequency(
    df: pd.DataFrame,
    window_hours: int = 24,
) -> pd.DataFrame:
    """
    Add a backward-looking count of how many times each recipient has
    appeared in the previous N hours up to (but not including) this row.

    LABEL LEAKAGE WARNING: A naive count using the full dataset would
    leak future information. This function uses only data from BEFORE
    each transaction, so it produces values consistent with what would
    be available at inference time.

    Computational cost: this is the most expensive step in preprocessing
    (O(n*k) where k = avg recipient appearances per window). For 2.77M
    rows, expect ~30 seconds. The simpler features above are O(n).

    Trade-off: this feature adds non-trivial complexity but captures the
    mule-account signal identified in Section B (low-frequency recipients
    are disproportionately likely to receive fraudulent funds).

    Args:
        df: Dataframe with at minimum 'step' and 'nameDest' columns.
        window_hours: Backward-looking window size in hours. Default 24.

    Returns:
        Dataframe with new column 'recipient_count_24h'.
    """
    df = df.copy()
    df = df.sort_values("step").reset_index(drop=True)

    # For each row, count occurrences of nameDest where step is within
    # the past `window_hours` and strictly before the current step.
    counts = np.zeros(len(df), dtype=int)
    history = {}  # nameDest -> list of step values it appeared at

    for i, (step, name) in enumerate(zip(df["step"].values, df["nameDest"].values)):
        # Get historical appearances of this recipient
        past_steps = history.get(name, [])
        # Count those within the window
        cutoff = step - window_hours
        count = sum(1 for s in past_steps if s >= cutoff and s < step)
        counts[i] = count
        # Record this transaction for future rows
        history.setdefault(name, []).append(step)

    df["recipient_count_24h"] = counts
    return df


# -----------------------------------------------------------------------
# STAGE 4: BUILD THE SKLEARN PREPROCESSING PIPELINE
# -----------------------------------------------------------------------

def build_preprocessing_pipeline(use_recipient_frequency: bool = False) -> Pipeline:
    """
    Construct the sklearn Pipeline that transforms raw filtered data
    into model-ready features.

    The pipeline performs:
        1. Stateless feature engineering (engineer_features)
        2. Optional recipient frequency (add_recipient_frequency)
        3. Column scaling (StandardScaler on numeric features)
        4. Pass-through for binary features (no scaling)
        5. Drop columns not in the final feature set

    Args:
        use_recipient_frequency: If True, include recipient_count_24h.
            Default False (slower; trade-off documented above).

    Returns:
        sklearn Pipeline ready to .fit_transform() on a dataframe.
    """
    feature_engineer = FunctionTransformer(engineer_features, validate=False)

    transformers = [
        ("scale_numeric", StandardScaler(), NUMERIC_FEATURES),
        ("passthrough_binary", "passthrough", BINARY_FEATURES),
    ]

    if use_recipient_frequency:
        transformers.append(
            ("scale_recipient", StandardScaler(), ["recipient_count_24h"])
        )

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # everything else (raw step, names, etc.) is dropped
    )

    pipeline = Pipeline([
        ("engineer", feature_engineer),
        ("preprocess", column_transformer),
    ])

    return pipeline


# -----------------------------------------------------------------------
# STAGE 5: CONVENIENCE — END-TO-END PREPARATION
# -----------------------------------------------------------------------

def prepare_dataset(
    df: pd.DataFrame,
    use_recipient_frequency: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run the full preprocessing flow up to (but not including) scaling.
    Returns features X and target y separately.

    Use this for quick exploration. For training, use the Pipeline
    object from build_preprocessing_pipeline() instead — it integrates
    cleanly with scikit-learn's training and cross-validation tools.

    Args:
        df: Raw PaySim dataframe.
        use_recipient_frequency: Add the rolling-count feature.

    Returns:
        (X, y) tuple where X is the engineered feature matrix and
        y is the isFraud target.
    """
    df = filter_fraud_relevant_types(df)
    df = engineer_features(df)

    if use_recipient_frequency:
        df = add_recipient_frequency(df)

    feature_cols = NUMERIC_FEATURES + BINARY_FEATURES
    if use_recipient_frequency:
        feature_cols = feature_cols + ["recipient_count_24h"]

    X = df[feature_cols].copy()
    y = df["isFraud"].copy()

    return X, y


# -----------------------------------------------------------------------
# Imports needed for FunctionTransformer (deferred to avoid circular deps)
# -----------------------------------------------------------------------

from sklearn.preprocessing import FunctionTransformer  # noqa: E402