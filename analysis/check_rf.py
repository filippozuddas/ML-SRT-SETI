#!/usr/bin/env python3
"""Check RF dimensions and compare with test data."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import joblib
from pathlib import Path

RF_PATH = Path(__file__).parent.parent / "results" / "large_scale_training" / "random_forest.joblib"
rf = joblib.load(RF_PATH)

print("RF Info:")
print(f"  n_estimators: {rf.n_estimators}")
print(f"  n_features_in_: {rf.n_features_in_}")
print(f"  classes_: {rf.classes_}")

# Check feature importances
print(f"\nFeature importances (top 10):")
importances = rf.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]
for idx in top_indices:
    print(f"  Feature {idx}: {importances[idx]:.4f}")

# Check if features are balanced between classes
print(f"\nTotal sum of importances: {importances.sum():.4f}")
