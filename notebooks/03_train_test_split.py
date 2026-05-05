"""
Section D Phase 1: Stratified train/validation/test split.

Splits the prepared feature matrix into 70/15/15 train/val/test sets,
preserving the fraud rate in each split (stratification).

Saves splits to disk so all subsequent modelling work uses identical data.

Run as:
    python notebooks/03_train_test_split.py
"""

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "mobile_money_fraud"
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# %%
# Load the prepared features and target
print("Loading prepared X and y from Section C output...")
X = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_features.csv")
y = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "y_target.csv").iloc[:, 0]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Overall fraud rate: {y.mean() * 100:.4f}%")

# %%
# Stratified split: first separate test (15%), then split rest into train (70%) and val (15%)
RANDOM_SEED = 42

# Step 1: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,
    stratify=y,
    random_state=RANDOM_SEED,
)

# Step 2: split remaining into train (70/85 = ~82.4%) and val (15/85 = ~17.6%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=15 / 85,
    stratify=y_temp,
    random_state=RANDOM_SEED,
)

# %%
# Verify the splits
print("Split sizes:")
print(f"  Train: {len(X_train):>8,} rows ({len(X_train) / len(X) * 100:.1f}%)")
print(f"  Val:   {len(X_val):>8,} rows ({len(X_val) / len(X) * 100:.1f}%)")
print(f"  Test:  {len(X_test):>8,} rows ({len(X_test) / len(X) * 100:.1f}%)")
print()
print("Fraud counts and rates per split:")
print(f"  Train: {y_train.sum():>5,} fraud ({y_train.mean() * 100:.4f}%)")
print(f"  Val:   {y_val.sum():>5,} fraud ({y_val.mean() * 100:.4f}%)")
print(f"  Test:  {y_test.sum():>5,} fraud ({y_test.mean() * 100:.4f}%)")

# %%
# Save splits to disk
print("Saving splits to disk...")

splits_dir = PROJECT_ROOT / "data" / "processed" / "splits"
splits_dir.mkdir(parents=True, exist_ok=True)

X_train.to_csv(splits_dir / "X_train.csv", index=False)
X_val.to_csv(splits_dir / "X_val.csv", index=False)
X_test.to_csv(splits_dir / "X_test.csv", index=False)
y_train.to_csv(splits_dir / "y_train.csv", index=False, header=True)
y_val.to_csv(splits_dir / "y_val.csv", index=False, header=True)
y_test.to_csv(splits_dir / "y_test.csv", index=False, header=True)

print(f"Saved to {splits_dir}/")
print("Files:", sorted(p.name for p in splits_dir.iterdir()))
# %%
