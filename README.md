"""
=============================================================================
K-NEAREST NEIGHBOR (KNN) CLASSIFICATION - PATTERN RECOGNITION MINI PROJECT
=============================================================================
Dataset : Iris Dataset (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width,
          Species). 150 samples, 3 balanced classes (setosa, versicolor,
          virginica).

This script walks through a complete, reproducible KNN workflow:
loading -> EDA -> cleaning -> scaling -> splitting -> baseline model ->
hyperparameter tuning -> final model -> evaluation -> new predictions ->
theory notes -> summary report.

Run with:  python knn_iris_classification.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

DATA_PATH = "iris.csv"   # CSV with columns: Sepal.Length, Sepal.Width,
                         # Petal.Length, Petal.Width, Species


# =============================================================================
# 1. LOAD DATASET
# =============================================================================
def load_dataset(path: str) -> pd.DataFrame:
    """Load the Iris dataset from CSV into a tidy DataFrame."""
    df = pd.read_csv(path)
    # Standardize column names so the rest of the code reads cleanly
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    return df


def explore_basic_info(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SECTION 1: DATASET OVERVIEW")
    print("=" * 70)

    print("\nFirst 5 rows:\n", df.head())
    print("\nLast 5 rows:\n", df.tail())
    print("\nDataset shape (rows, columns):", df.shape)
    print("\nColumn names:", list(df.columns))
    print("\nData types:\n", df.dtypes)
    print("\nDescriptive statistics:\n", df.describe())
    print("\nClass distribution:\n", df["species"].value_counts())

    # Interpretation:
    # The dataset has 150 samples and 4 numeric features describing flower
    # measurements (in cm), plus one categorical target (species). The
    # classes are perfectly balanced (50 each), which means accuracy is a
    # trustworthy metric here (no class imbalance to correct for).


# =============================================================================
# 2. DATA EXPLORATION (EDA)
# =============================================================================
def explore_data(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SECTION 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print("\n--- df.info() ---")
    df.info()

    print("\n--- df.describe() ---")
    print(df.describe())

    print("\n--- Class frequencies ---")
    print(df["species"].value_counts(normalize=True))

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Correlation matrix + heatmap
    corr = df[numeric_cols].corr()
    print("\n--- Correlation matrix ---\n", corr)

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Iris Features")
    plt.tight_layout()
    plt.savefig("plot_correlation_heatmap.png")
    plt.close()

    # Pairplot (dataset is small enough -> appropriate here)
    pp = sns.pairplot(df, hue="species", corner=True)
    pp.fig.suptitle("Pairplot of Iris Features by Species", y=1.02)
    pp.savefig("plot_pairplot.png")
    plt.close()

    # Histograms
    df[numeric_cols].hist(figsize=(8, 6), bins=15, color="steelblue", edgecolor="black")
    plt.suptitle("Feature Distributions (Histograms)")
    plt.tight_layout()
    plt.savefig("plot_histograms.png")
    plt.close()

    # Boxplots (per feature, split by species)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.boxplot(data=df, x="species", y=col, ax=ax)
        ax.set_title(f"Boxplot of {col} by species")
    plt.tight_layout()
    plt.savefig("plot_boxplots.png")
    plt.close()

    # Interpretation:
    # Petal length and petal width are strongly correlated with each other
    # and are the most discriminative features for separating the three
    # species (setosa is clearly separated; versicolor/virginica overlap
    # more, especially in sepal measurements). This visual separability is
    # exactly what makes KNN -- a distance-based method -- effective here.


# =============================================================================
# 3. HANDLE MISSING VALUES
# =============================================================================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("SECTION 3: MISSING VALUE CHECK")
    print("=" * 70)

    missing = df.isnull().sum()
    print("\nMissing values per column:\n", missing)

    if missing.sum() > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        print("\nMissing numeric values were filled with the column mean.")
        print("Reason: the mean preserves the overall distribution and avoids")
        print("dropping rows, which would shrink an already small dataset.")
    else:
        print("\nNo missing values found -- no imputation needed.")

    return df


# =============================================================================
# 4. FEATURE SCALING
# =============================================================================
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    KNN classifies a point based on the distance to its nearest neighbors.
    If features are on different scales (e.g. one column in cm ranging
    0-10 and another in mm ranging 0-100), the larger-scale feature will
    dominate the distance calculation regardless of its actual importance.

    StandardScaler transforms each feature to have mean 0 and standard
    deviation 1, so every feature contributes proportionally to the
    Euclidean distance used by KNN.

    IMPORTANT: the scaler is fit ONLY on the training data, then applied
    to the test data, to avoid leaking test-set information into training.
    """
    print("\n" + "=" * 70)
    print("SECTION 4: FEATURE SCALING (StandardScaler)")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nFeatures scaled to mean=0, std=1.")
    print("Why this matters for KNN: distance metrics (Euclidean/Manhattan)")
    print("treat all features as equally weighted by raw magnitude. Without")
    print("scaling, a feature measured in larger units would dominate the")
    print("distance calculation and bias neighbor selection.")

    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# 5. TRAIN-TEST SPLIT
# =============================================================================
def split_data(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("SECTION 5: TRAIN-TEST SPLIT (80/20, stratified)")
    print("=" * 70)

    X = df.drop(columns=["species"])
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTraining size: {X_train.shape[0]} samples")
    print(f"Testing size:  {X_test.shape[0]} samples")
    print("\nStratified sampling was used so the class proportions in")
    print("train/test sets match the original (balanced) dataset.")

    return X_train, X_test, y_train, y_test


# =============================================================================
# 6. BUILD INITIAL KNN MODEL
# =============================================================================
def build_initial_model(X_train_scaled, y_train, X_test_scaled):
    print("\n" + "=" * 70)
    print("SECTION 6: INITIAL KNN MODEL (k=5)")
    print("=" * 70)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("\nInitial model trained with k=5.")
    print("Predictions on test set:", list(y_pred))

    return model, y_pred


# =============================================================================
# 7. HYPERPARAMETER TUNING (cross-validation over k)
# =============================================================================
def tune_k(X_train_scaled, y_train, k_values=(1, 3, 5, 7, 9, 11)):
    print("\n" + "=" * 70)
    print("SECTION 7: HYPERPARAMETER TUNING (5-fold cross-validation)")
    print("=" * 70)

    results = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring="accuracy")
        results.append({"k": k, "Cross Validation Accuracy": scores.mean()})

    cv_df = pd.DataFrame(results)
    print("\n", cv_df)

    best_row = cv_df.loc[cv_df["Cross Validation Accuracy"].idxmax()]
    best_k = int(best_row["k"])
    print(f"\nBest k found via cross-validation: k={best_k} "
          f"(CV accuracy={best_row['Cross Validation Accuracy']:.4f})")

    return cv_df, best_k


# =============================================================================
# 8. VISUALIZATION: k vs ACCURACY
# =============================================================================
def plot_k_vs_accuracy(cv_df: pd.DataFrame, best_k: int) -> None:
    print("\n" + "=" * 70)
    print("SECTION 8: k vs ACCURACY PLOT")
    print("=" * 70)

    plt.figure(figsize=(7, 5))
    plt.plot(cv_df["k"], cv_df["Cross Validation Accuracy"], marker="o", linestyle="-", color="steelblue")

    best_acc = cv_df.loc[cv_df["k"] == best_k, "Cross Validation Accuracy"].values[0]
    plt.scatter([best_k], [best_acc], color="red", s=120, zorder=5, label=f"Best k={best_k}")

    plt.title("KNN: Cross-Validation Accuracy vs k")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_k_vs_accuracy.png")
    plt.close()

    print("\nPlot saved as plot_k_vs_accuracy.png")
    print(f"Best k={best_k} is highlighted in red.")


# =============================================================================
# 9. TRAIN FINAL MODEL
# =============================================================================
def train_final_model(X_train_scaled, y_train, X_test_scaled, best_k: int):
    print("\n" + "=" * 70)
    print(f"SECTION 9: FINAL MODEL (k={best_k})")
    print("=" * 70)

    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_train_scaled, y_train)
    y_pred_final = final_model.predict(X_test_scaled)

    print(f"\nFinal model trained using optimal k={best_k}.")
    return final_model, y_pred_final


# =============================================================================
# 10. MODEL EVALUATION
# =============================================================================
def evaluate_model(y_test, y_pred, labels):
    print("\n" + "=" * 70)
    print("SECTION 10: MODEL EVALUATION")
    print("=" * 70)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision (macro avg): {prec:.4f}")
    print(f"Recall (macro avg):    {rec:.4f}")
    print(f"F1-score (macro avg):  {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion Matrix:\n", cm)

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig("plot_confusion_matrix.png")
    plt.close()

    # Interpretation:
    # Accuracy tells us overall correctness. Precision asks "of everything
    # predicted as class X, how much was actually X?" Recall asks "of all
    # true class X samples, how many did we catch?" On Iris, setosa is
    # almost always perfectly separated; most confusion (if any) tends to
    # occur between versicolor and virginica, which overlap slightly in
    # petal measurements.

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


# =============================================================================
# 11. PREDICT NEW SAMPLES
# =============================================================================
def predict_new_samples(model, scaler, feature_names):
    print("\n" + "=" * 70)
    print("SECTION 11: PREDICTING NEW / UNSEEN SAMPLES")
    print("=" * 70)

    # A few realistic hand-crafted flower measurements (cm)
    new_samples = pd.DataFrame(
        [
            [5.0, 3.4, 1.5, 0.2],   # looks like setosa
            [6.0, 2.7, 5.1, 1.6],   # looks like versicolor/virginica
            [6.7, 3.3, 5.7, 2.1],   # looks like virginica
        ],
        columns=feature_names,
    )

    new_samples_scaled = scaler.transform(new_samples)
    predictions = model.predict(new_samples_scaled)
    probabilities = model.predict_proba(new_samples_scaled)

    print("\nNew samples:\n", new_samples)
    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        prob_str = ", ".join(f"{cls}: {p:.2f}" for cls, p in zip(model.classes_, proba))
        print(f"\nSample {i + 1}: Predicted species = {pred}")
        print(f"  Class probabilities -> {prob_str}")


# =============================================================================
# 12-14. THEORY (printed as documentation, not executed logic)
# =============================================================================
KNN_THEORY = """
SECTION 13: KNN THEORY
-----------------------
What is KNN?
  K-Nearest Neighbors is a supervised classification algorithm that assigns
  a label to a new point based on the majority label among its 'k' closest
  points in the training data.

Lazy learning / Instance-based learning:
  KNN does not build an explicit internal model during training -- it just
  stores the training data ("lazy learning"). All the computation happens
  at prediction time, when distances to every stored point are calculated
  ("instance-based learning").

Distance metrics:
  - Euclidean distance: straight-line distance between two points,
    sqrt(sum((x_i - y_i)^2)). Most common default for KNN.
  - Manhattan distance: sum of absolute differences, sum(|x_i - y_i|).
    Useful when movement is grid-like or features are less correlated.

Why scaling matters:
  Distance metrics combine all features. A feature with a larger numeric
  range will dominate the distance value, effectively giving it more
  "voting power" even if it isn't more predictive. Scaling equalizes that.

Advantages of KNN:
  - Simple, intuitive, no training phase.
  - Naturally handles multi-class problems.
  - Non-linear decision boundaries without complex tuning.

Disadvantages of KNN:
  - Slow at prediction time on large datasets (must compare to all points).
  - Sensitive to irrelevant features and feature scale.
  - Struggles in high dimensions ("curse of dimensionality").

Choosing k:
  - Small k -> model is sensitive to noise (overfitting), jagged boundaries.
  - Large k -> model is smoother but may blur class boundaries
    (underfitting), and is slower.
  - Cross-validation (as done in Section 7) is the standard way to pick k.

Underfitting vs Overfitting:
  - Overfitting (k too small): the model memorizes noise in training data
    and generalizes poorly.
  - Underfitting (k too large): the model is too generalized and misses
    real distinctions between classes.
"""


def print_theory():
    print("\n" + "=" * 70)
    print(KNN_THEORY)
    print("=" * 70)


# =============================================================================
# 17. FINAL SUMMARY REPORT
# =============================================================================
def print_summary(df, X_train, X_test, best_k, metrics):
    print("\n" + "=" * 70)
    print("SECTION 17: FINAL SUMMARY REPORT")
    print("=" * 70)

    summary = f"""
Dataset used        : Iris Dataset (CSV)
Number of samples    : {df.shape[0]}
Number of features   : {df.shape[1] - 1}
Training set size    : {X_train.shape[0]}
Testing set size     : {X_test.shape[0]}
Optimal k            : {best_k}
Test Accuracy        : {metrics['accuracy']:.4f}
Precision (macro)    : {metrics['precision']:.4f}
Recall (macro)       : {metrics['recall']:.4f}
F1-score (macro)     : {metrics['f1']:.4f}

Key observations:
  - Petal length/width are the most discriminative features.
  - Setosa is almost perfectly separable from the other two species.
  - Versicolor and virginica show minor overlap, which is the main source
    of any misclassification.
  - Feature scaling was essential since KNN relies purely on distance.

Practical applications of KNN:
  - Recommendation systems (find similar users/items)
  - Image and handwriting recognition
  - Medical diagnosis (classify based on similar past patient cases)
  - Anomaly/fraud detection
"""
    print(summary)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    df = load_dataset(DATA_PATH)
    explore_basic_info(df)
    explore_data(df)
    df = handle_missing_values(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Initial baseline model
    build_initial_model(X_train_scaled, y_train, X_test_scaled)

    # Hyperparameter tuning
    cv_df, best_k = tune_k(X_train_scaled, y_train)
    plot_k_vs_accuracy(cv_df, best_k)

    # Final model trained on best k
    final_model, y_pred_final = train_final_model(X_train_scaled, y_train, X_test_scaled, best_k)

    # Evaluation
    labels = sorted(df["species"].unique())
    metrics = evaluate_model(y_test, y_pred_final, labels)

    # Predict new unseen samples
    predict_new_samples(final_model, scaler, X_train.columns)

    # Theory + summary
    print_theory()
    print_summary(df, X_train, X_test, best_k, metrics)

    print("\nAll plots saved in the current directory:")
    print("  plot_correlation_heatmap.png, plot_pairplot.png, plot_histograms.png,")
    print("  plot_boxplots.png, plot_k_vs_accuracy.png, plot_confusion_matrix.png")


if __name__ == "__main__":
    main()
