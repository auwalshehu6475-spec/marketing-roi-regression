# -*- coding: utf-8 -*-
"""
K-Nearest Neighbor Classification – Pattern Recognition Mini Project
Dataset: Iris (from scikit-learn)
"""

# ==============================================================================
# 1. IMPORTS & SETUP
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
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Set global aesthetic styles for visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# ==============================================================================
# 2. DATA DISCOVERY & PREPROCESSING
# ==============================================================================
print("=" * 60)
print("STAGE 1: DATA DISCOVERY & PREPROCESSING")
print("=" * 60)

# Load the built-in Iris dataset
iris = load_iris()
X = iris.data          # Features: sepal length, sepal width, petal length, petal width
y = iris.target        # Target labels: 0=setosa, 1=versicolor, 2=virginica
feature_names = iris.feature_names
target_names = iris.target_names

# Convert to a Pandas DataFrame for structured exploration
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

# Print exploratory structural findings
print(f"• Dataset dimensions: {df.shape[0]} samples, {df.shape[1] - 1} features")
print("\n• First 5 rows of the dataset:")
print(df.head())

print("\n• Data types and structural info:")
df.info()

print("\n• Missing values check (per column):")
print(df.isnull().sum())

print("\n• Descriptive statistical summary:")
print(df.describe())


# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS (VISUALIZATIONS)
# ==============================================================================
print("\nGenerating exploratory visualizations...")

# Pairplot to evaluate feature separation and distributions
pair_plot = sns.pairplot(df, hue="species", diag_kind="kde", palette="Set2")
pair_plot.fig.suptitle("Pairwise Feature Distributions by Species", y=1.02)
plt.show()
plt.close()

# Boxplots to evaluate individual feature ranges and potential outliers
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
for i, feature in enumerate(feature_names):
    sns.boxplot(x="species", y=feature, data=df, ax=axes[i], palette="Set2")
    axes[i].set_title(f"Distribution of {feature}")
plt.tight_layout()
plt.show()
plt.close()


# ==============================================================================
# 4. DATA SPLITTING & FEATURE SCALING
# ==============================================================================
# Stratified Train-Test Split (80% Train, 20% Test) to ensure balanced classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n• Training set size: {X_train.shape[0]} samples")
print(f"• Test set size: {X_test.shape[0]} samples")

# Feature Scaling: Essential for distance-based algorithms like KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n• Feature scaling completed using StandardScaler.")
print(f"  Scaled Training Mean (Target ~0): {np.mean(X_train_scaled, axis=0).round(2)}")
print(f"  Scaled Training Std  (Target ~1): {np.std(X_train_scaled, axis=0).round(2)}")


# ==============================================================================
# 5. HYPERPARAMETER TUNING (CROSS-VALIDATION FOR k)
# ==============================================================================
print("\n" + "=" * 60)
print("STAGE 2: HYPERPARAMETER TUNING")
print("=" * 60)

# Evaluate odd k-values to prevent tie-voting issues
k_values = list(range(1, 16, 2))  
cv_scores = []

print("Running 5-Fold Cross-Validation for optimal 'k' identification...")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Perform a 5-fold CV on the scaled training dataset
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"  k = {k:2d} | Mean CV Accuracy: {scores.mean():.4f}")

# Plot validation curves for hyperparameter analysis
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_scores, marker='o', linestyle='--', color='teal', linewidth=2)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('5-Fold Cross-Validated Accuracy')
plt.title('Hyperparameter Tuning: k-Value Optimization Curve')
plt.xticks(k_values)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
plt.close()

# Identify the absolute best performing parameter
best_k = k_values[np.argmax(cv_scores)]
print(f"\n=> Optimal parameter chosen: k = {best_k} (CV Accuracy: {max(cv_scores):.4f})")


# ==============================================================================
# 6. KNN MODEL IMPLEMENTATION & TRAINING
# ==============================================================================
print("\n" + "=" * 60)
print("STAGE 3: MODEL IMPLEMENTATION & TESTING")
print("=" * 60)

# Instantiate the optimal model architecture
knn_final = KNeighborsClassifier(n_neighbors=best_k)

# Train/Fit the model onto the processed training arrays
print(f"Training final KNeighborsClassifier using k={best_k}...")
knn_final.fit(X_train_scaled, y_train)

# Generate final classifications on unseen test matrices
y_pred = knn_final.predict(X_test_scaled)
print("Predictions successfully generated for the test partition.")


# ==============================================================================
# 7. PERFORMANCE EVALUATION & INTERPRETATION
# ==============================================================================
print("\n" + "=" * 60)
print("STAGE 4: PERFORMANCE EVALUATION & INTERPRETATION")
print("=" * 60)

# Calculate core performance metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Overall Test Accuracy : {acc:.4f}")
print(f"Macro Precision       : {prec:.4f}")
print(f"Macro Recall          : {rec:.4f}")
print(f"Macro F1-Score        : {f1:.4f}")

# Detailed Scikit-Learn Class-by-Class Breakdown Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Compute and plot the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names, cbar=False)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix: Final Evaluation Predictions', fontsize=14)
plt.show()
plt.close()


# ==============================================================================
# 8. PREDICTION DEPLOYMENT TEST ON A NEW SAMPLE
# ==============================================================================
print("\n" + "=" * 60)
print("STAGE 5: INFERENCE ON NEW PRODUCTION DATA")
print("=" * 60)

# Simulating an unseen flower vector entry: [SepalL, SepalW, PetalL, PetalW]
new_sample = np.array([[6.1, 3.1, 4.7, 1.4]])
print(f"Incoming sample characteristics: {new_sample[0]}")

# Transform vector utilizing the historical scaler configuration
new_sample_scaled = scaler.transform(new_sample)

# Execute predictions and pull probability arrays
prediction = knn_final.predict(new_sample_scaled)
pred_proba = knn_final.predict_proba(new_sample_scaled)

print(f"Predicted Taxonomy Class: {target_names[prediction[0]].upper()}")
print("Raw confidence vector mapping:")
for i, name in enumerate(target_names):
    print(f"  * {name:<10}: {pred_proba[0][i] * 100:.2f}%")


# ==============================================================================
# 9. SUMMARY OBSERVATIONS & DOCUMENTATION NOTES
# ==============================================================================
print("\n" + "=" * 60)
print("PROJECT SUMMARY & DOCUMENTATION ARCHITECTURE")
print("=" * 60)
print("1. PREPROCESSING: Missing values checked (none found). Features normalized utilizing Z-score scaling.")
print("   This steps prevents features with larger measurement limits (e.g., Sepal Length) from dominating distance calcs.")
print("2. SEPARATION: Stratified splits were leveraged to maintain strict balance distribution controls across splits.")
print("3. OPTIMIZATION: Cross-Validation evaluated historical variance. Selecting an odd k-value avoids multi-class draw-ties.")
print("4. SUMMARY: The confusion matrix verifies complete separation of 'setosa' and high confidence boundary maps.")