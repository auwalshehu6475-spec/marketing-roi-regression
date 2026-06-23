
import time
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set plotting style
sns.set_theme(style='darkgrid', font_scale=1.1)

# =====================================================================
# 1. DATA DISCOVERY & PREPROCESSING
# =====================================================================
print("=== 1. Data Discovery & Preprocessing ===")

# Load the dataset directly from sklearn
raw_data = load_breast_cancer()
X = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
y = pd.Series(raw_data.target, name='diagnosis')  # 0: Malignant, 1: Benign

print(f"Dataframe shape: {X.shape}")
print(f"Class distribution:\n{y.value_counts().rename({0: 'Malignant (0)', 1: 'Benign (1)'})}\n")

# Split dataset into training and testing data (80/20 split)
# stratify=y ensures equal representation of classes in both splits
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4, stratify=y
)

# Feature Scaling (Crucial for distance-based algorithms like kNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print("Data successfully split and scaled.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}\n")


# =====================================================================
# 2. CUSTOM KNN MODEL IMPLEMENTATION
# =====================================================================
class CustomKNN:
    """k-Nearest Neighbors classifier implemented from scratch."""
    def __init__(self, k=3, metric='euclidean', p=2):
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None
    
    def euclidean(self, v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))
    
    def manhattan(self, v1, v2):
        return np.sum(np.abs(v1 - v2))
    
    def minkowski(self, v1, v2, p=2):
        return np.sum(np.abs(v1 - v2) ** p) ** (1 / p)
        
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        
    def predict(self, X_test):
        X_test = np.array(X_test)
        preds = []
        for test_row in X_test:
            nearest_neighbours = self._get_neighbours(test_row)
            # Replaced scipy.stats.mode with Counter for reliable speed and version safety
            majority = Counter(nearest_neighbours).most_common(1)[0][0]
            preds.append(majority)
        return np.array(preds)
    
    def _get_neighbours(self, test_row):
        distances = []
        for train_row, train_class in zip(self.X_train, self.y_train):
            if self.metric == 'euclidean':
                dist = self.euclidean(train_row, test_row)
            elif self.metric == 'manhattan':
                dist = self.manhattan(train_row, test_row)
            elif self.metric == 'minkowski':
                dist = self.minkowski(train_row, test_row, self.p)
            else:
                raise NameError('Supported metrics are euclidean, manhattan and minkowski')
            distances.append((dist, train_class))
            
        distances.sort(key=lambda x: x[0])
        neighbours = [distances[i][1] for i in range(self.k)]
        return neighbours


# =====================================================================
# 3. HYPERPARAMETER TUNING (k-Value Testing)
# =====================================================================
print("=== 2. Hyperparameter Tuning (Cross-Validation) ===")

k_values = range(1, 21)
cv_scores = []

# Validate hyperparameter spaces systematically using 5-Fold Cross Validation
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k value found via 5-fold CV: k = {optimal_k}")
print(f"Best CV Accuracy: {max(cv_scores):.4f}\n")


# =====================================================================
# 4. PERFORMANCE EVALUATION & INTERPRETATION
# =====================================================================
print("=== 3. Performance Evaluation & Comparison ===")

# Train and evaluate Scikit-Learn implementation with optimal k
sklearn_clf = KNeighborsClassifier(n_neighbors=optimal_k)
sklearn_clf.fit(X_train, y_train)
sklearn_preds = sklearn_clf.predict(X_test)
sklearn_acc = accuracy_score(y_test, sklearn_preds)

# Train and evaluate Custom Implementation with optimal k
custom_clf = CustomKNN(k=optimal_k, metric='euclidean')
custom_clf.fit(X_train, y_train)
custom_preds = custom_clf.predict(X_test)
custom_acc = accuracy_score(y_test, custom_preds)

print(f"Scikit-Learn kNN (k={optimal_k}) Test Accuracy: {sklearn_acc * 100:.2f}%")
print(f"Custom Scratch kNN (k={optimal_k}) Test Accuracy : {custom_acc * 100:.2f}%\n")

print("Classification Report (Scikit-Learn Model):")
print(classification_report(y_test, sklearn_preds, target_names=raw_data.target_names))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, sklearn_preds)
print(cm)

# Optional Plotting of Hyperparameter Curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o', linestyle='dashed', color='royalblue', markersize=8)
plt.title('kNN Hyperparameter Tuning: Cross-Validation Accuracy vs. k-value')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean CV Accuracy')
plt.xticks(k_values)
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k ({optimal_k})')
plt.legend()
# plt.show() # Uncomment to view the plot during local execution
