"""
Test script: verify src/preprocessing.py works end-to-end.

Runs the preprocessing pipeline on the filtered PaySim dataset, validates
the output structure, and times the slow steps.

Run cells with VS Code's `# %%` cell markers, or run as a script:
    python notebooks/02_test_preprocessing.py
"""

# %%
# Setup: import the preprocessing module
# Note: we add the project root to sys.path so we can import from src/
import sys
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mobile_money_fraud"
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import time

from src.preprocessing import (
    filter_fraud_relevant_types,
    engineer_features,
    add_recipient_frequency,
    build_preprocessing_pipeline,
    prepare_dataset,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
)

print("Imports successful — preprocessing module loaded")

# %%
# Load the filtered dataset (skips re-running EDA filtering)
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "paysim_filtered.csv"
df = pd.read_csv(DATA_PATH)

print(f"Loaded: {df.shape}")
print(f"Fraud rate: {df['isFraud'].mean() * 100:.4f}%")

# %%
# Test 1: stateless feature engineering on a small sample
sample = df.sample(1000, random_state=42)
sample_engineered = engineer_features(sample)

print("Original columns:", list(sample.columns))
print()
print("After engineering, new columns:", 
      [c for c in sample_engineered.columns if c not in sample.columns])
print()
print("Shape before:", sample.shape)
print("Shape after: ", sample_engineered.shape)

# %%
# Test 2: spot-check the engineered feature values
print("Sample of engineered features:")
print(sample_engineered[
    ["amount", "amount_log", "drainage_ratio", "error_orig", 
     "hour_of_day", "is_unusual_hour", "type", "type_encoded"]
].head(10))

# %%
# Test 3: validate engineered feature ranges and types
checks_passed = []
checks_failed = []

# amount_log should be non-negative for non-negative amounts
if (sample_engineered["amount_log"] >= 0).all():
    checks_passed.append("amount_log non-negative")
else:
    checks_failed.append("amount_log has negative values")

# is_unusual_hour should only be 0 or 1
if set(sample_engineered["is_unusual_hour"].unique()).issubset({0, 1}):
    checks_passed.append("is_unusual_hour is binary")
else:
    checks_failed.append("is_unusual_hour has non-binary values")

# type_encoded should only be 0 or 1
if set(sample_engineered["type_encoded"].unique()).issubset({0, 1}):
    checks_passed.append("type_encoded is binary")
else:
    checks_failed.append("type_encoded has non-binary values")

# hour_of_day should be in [0, 23]
if (sample_engineered["hour_of_day"].between(0, 23)).all():
    checks_passed.append("hour_of_day in [0, 23]")
else:
    checks_failed.append("hour_of_day out of range")

print("Validation checks:")
for c in checks_passed:
    print(f"  ✓ {c}")
for c in checks_failed:
    print(f"  ✗ {c}")

# %%
# Test 4: build and run the full pipeline
print("Building preprocessing pipeline (without recipient frequency)...")
pipeline = build_preprocessing_pipeline(use_recipient_frequency=False)
print(pipeline)

# %%
# Run pipeline on small sample to verify shape
print("Fitting pipeline on 10,000-row sample...")
start = time.time()
sample_large = df.sample(10000, random_state=42)
X_transformed = pipeline.fit_transform(sample_large)
elapsed = time.time() - start

print(f"Done in {elapsed:.2f} seconds")
print(f"Output shape: {X_transformed.shape}")
print(f"Expected feature count: {len(NUMERIC_FEATURES) + len(BINARY_FEATURES)}")
print()
print("First 5 transformed rows:")
print(X_transformed[:5])

# %%
# Test 5: end-to-end on the full dataset (no recipient frequency)
print("Running prepare_dataset() on full filtered dataset...")
start = time.time()
X, y = prepare_dataset(df, use_recipient_frequency=False)
elapsed = time.time() - start

print(f"Done in {elapsed:.2f} seconds")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y fraud rate: {y.mean() * 100:.4f}%")

# %%
# Test 6: time the optional recipient frequency feature on a sample
# (Skip the full dataset for now — we'll only run that once we commit)
print("Testing recipient frequency on 100,000-row sample...")
sample_for_recip = df.sample(100000, random_state=42).copy()
start = time.time()
sample_with_recip = add_recipient_frequency(sample_for_recip)
elapsed = time.time() - start

print(f"Done in {elapsed:.2f} seconds for 100K rows")
print(f"Estimated full-dataset time: ~{elapsed * 27.7:.0f} seconds for 2.77M rows")
print()
print("Sample of recipient_count_24h values:")
print(sample_with_recip[["step", "nameDest", "recipient_count_24h"]].head(20))
print()
print("Distribution of recipient_count_24h:")
print(sample_with_recip["recipient_count_24h"].describe())

# %%
# Save the transformed dataset for Section D modelling
print("Saving prepared X and y for Section D...")
X.to_csv(PROJECT_ROOT / "data" / "processed" / "X_features.csv", index=False)
y.to_csv(PROJECT_ROOT / "data" / "processed" / "y_target.csv", index=False)
print(f"Saved X to: data/processed/X_features.csv ({X.shape})")
print(f"Saved y to: data/processed/y_target.csv ({y.shape})")
# %%
