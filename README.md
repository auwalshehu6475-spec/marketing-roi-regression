# -*- coding: utf-8 -*-
"""
K-Nearest Neighbor (KNN) Classification Project
Dataset: Iris Dataset (scikit-learn)

Project Workflow:
1. Import libraries
2. Load and explore data
3. Perform preprocessing
4. Split data
5. Scale features
6. Hyperparameter tuning
7. Train final model
8. Evaluate performance
9. Interpret results
10. Predict new samples
"""

# ==============================================================================
# 1. IMPORT LIBRARIES
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Plot settings
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================

print("=" * 60)
print("DATA LOADING")
print("=" * 60)

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

print(f"Dataset Shape: {df.shape}")

print("\nFirst 5 rows")