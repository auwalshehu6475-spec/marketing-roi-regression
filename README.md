# ==========================================
# KNN Classification Mini Project
# Dataset: Breast Cancer Wisconsin (Diagnostic)
# ==========================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Set style for plotting
sns.set_theme(style="whitegrid")

# ------------------------------------------
# Task 1: Load and Explore Dataset
# ------------------------------------------
print("--- Task 1: Loading Data ---")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print(f"Dataset Shape: {X.shape}")
print(f"Target Distribution:\n{y.value_counts(normalize=True)}")
print(f"Target Names: 0 = {data.target_names[0]}, 1 = {data.target_names[1]}\n")

# ------------------------------------------
# Task 2: Handle Missing Values & Visualize
# ------------------------------------------
print("--- Task 2: Data Preprocessing & EDA ---")
print(f"Missing values in dataset: {X.isnull().sum().sum()}")

# Visualizing distributions of the first 4 features as a sample
sample_features = X.columns[:4]
plt.figure(figsize=(12, 8))
for i, col in enumerate(sample_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(
        data=X,
        x=col,
        hue=y,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# ------------------------------------------
# Task 3: Train/Test Split (80/20)
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}\n")

# ------------------------------------------
# Task 4, 5, 6: Feature Scaling + Hyperparameter Tuning (k) via CV
# ------------------------------------------
# IMPORTANT: Use a pipeline to avoid data leakage –
# scaling is refitted on each CV fold.
print("--- Tasks 4, 5, 6: Hyperparameter Tuning (k) ---")
k_values = [1, 3, 5, 7, 9, 11]
cv_scores = []

for k in k_values:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )
    # 5-Fold Cross Validation on raw X_train
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())
    print(f"k = {k:2d} | 5-Fold CV Mean Accuracy: {scores.mean():.4f}")

# ------------------------------------------
# Task 7: Plot k vs. Accuracy
# ------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_scores, marker="o", linestyle="--", color="b")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("KNN: Tuning parameter $k$")
plt.xticks(k_values)
plt.show()

optimal_k = k_values[np.argmax(cv_scores)]
print(f"\nOptimal k identified via CV: {optimal_k}\n")

# ------------------------------------------
# Task 8: Final Model Evaluation (no leakage)
# ------------------------------------------
print("--- Task 8: Final Model Evaluation ---")
# Build the final pipeline with the optimal k
final_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=optimal_k)),
    ]
)
final_pipe.fit(X_train, y_train)
y_pred = final_pipe.predict(X_test)

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Final Test Accuracy:  {accuracy:.4f}")
print(f"Final Test Precision: {precision:.4f}")
print(f"Final Test Recall:    {recall:.4f}")
print(f"Final Test F1-Score:  {f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names,
)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title(f"Confusion Matrix (k={optimal_k})")
plt.show()

# ------------------------------------------
# Task 9: Test Predictions on New Sample (using raw features)
# ------------------------------------------
print("--- Task 9: Real-time Inference Example ---")
# Create a realistic new sample from raw (unscaled) data
# For demonstration, we take the mean of the training set features
raw_new_sample = np.mean(X_train, axis=0).values.reshape(1, -1)

prediction = final_pipe.predict(raw_new_sample)
prediction_proba = final_pipe.predict_proba(raw_new_sample)

print(
    f"New Sample Prediction Class: {prediction[0]} ({data.target_names[prediction[0]]})"
)
print(f"Prediction Probability [Malignant, Benign]: {prediction_proba[0]}")