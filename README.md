# ======================================================
# KNN IRIS CLASSIFICATION PROJECT
# ======================================================

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ======================================================
# 1. LOAD DATASET
# ======================================================

print("========== LOADING DATASET ==========")

iris = load_iris(as_frame=True)

df = iris.frame

# Create readable species names
df["species"] = df["target"].map(
{
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}
)

# Remove numeric target column
df.drop("target", axis=1, inplace=True)

print(df.head())

# ======================================================
# 2. DATA DISCOVERY
# ======================================================

print("\n========== DATA EXPLORATION ==========")

print("\nDataset Shape:")

print(df.shape)

print("\nDataset Information:")

print(df.info())

print("\nSummary Statistics:")

print(df.describe())

print("\nMissing Values:")

print(df.isnull().sum())

# ======================================================
# 3. VISUALIZATION
# ======================================================

print("\n========== DATA VISUALIZATION ==========")

sns.pairplot(
    df,
    hue="species",
    diag_kind="hist"
)

plt.suptitle(
    "Iris Dataset Feature Relationships",
    y=1.02
)

plt.show()

# ======================================================
# 4. SPLIT FEATURES AND TARGET
# ======================================================

X = df.drop(
    "species",
    axis=1
)

y = df["species"]

# 80% training
# 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Size:", X_train.shape)

print("Testing Size:", X_test.shape)

# ======================================================
# 5. FEATURE SCALING
# ======================================================

print("\n========== FEATURE SCALING ==========")

# KNN relies on distance calculations
# Therefore scaling is important

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# ======================================================
# 6. HYPERPARAMETER TUNING
# ======================================================

print("\n========== HYPERPARAMETER TUNING ==========")

k_values = range(1, 21)

cv_scores = []

for k in k_values:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    scores = cross_val_score(
        knn,
        X_train_scaled,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    cv_scores.append(
        scores.mean()
    )

    print(
        f"k={k} : Accuracy={scores.mean():.4f}"
    )

# Select best k

optimal_k = k_values[
    np.argmax(cv_scores)
]

print("\nBest k =", optimal_k)

# Plot tuning graph

plt.figure(figsize=(8,5))

plt.plot(
    k_values,
    cv_scores,
    marker="o"
)

plt.xlabel("k value")

plt.ylabel("Cross Validation Accuracy")

plt.title("Selecting Optimal k")

plt.grid(True)

plt.show()

# ======================================================
# 7. TRAIN FINAL MODEL
# ======================================================

print("\n========== TRAINING FINAL MODEL ==========")

final_knn = KNeighborsClassifier(
    n_neighbors=optimal_k
)

final_knn.fit(
    X_train_scaled,
    y_train
)

# ======================================================
# 8. MAKE PREDICTIONS
# ======================================================

y_pred = final_knn.predict(
    X_test_scaled
)

# ======================================================
# 9. EVALUATE MODEL
# ======================================================

print("\n========== MODEL EVALUATION ==========")

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ======================================================
# 10. CONFUSION MATRIX
# ======================================================

labels = np.unique(y)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ======================================================
# 11. TEST NEW SAMPLE
# ======================================================

print("\n========== NEW PREDICTION ==========")

sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

sample_scaled = scaler.transform(
    sample
)

prediction = final_knn.predict(
    sample_scaled
)

print(
    "Sample:",
    sample.values[0]
)

print(
    "Predicted Species:",
    prediction[0]
)

# ======================================================
# 12. RESULT INTERPRETATION
# ======================================================

print("\n========== INTERPRETATION ==========")

print(
"""
The KNN classifier successfully classified
the Iris flower species.

Feature scaling improved performance because
KNN depends on distance calculations.

Cross-validation was used to select the
best value of k.

The confusion matrix and classification
report show that the model performs
very well on this dataset.
"""
)