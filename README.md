import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')
df = X.copy()
df['species'] = y.map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"Dataset Shape: {df.shape}")
print(f"Features: {iris.feature_names}")
print(f"Classes: {list(iris.target_names)}")
print(f"\nClass Distribution:\n{df['species'].value_counts()}")
print(f"\nMissing Values: {df.isnull().sum().sum()}")
print(f"\nStatistical Summary:\n{df.describe().round(3)}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
for idx, col in enumerate(iris.feature_names):
    sns.histplot(data=df, x=col, hue='species', kde=True, ax=axes[idx], palette='viridis')
    axes[idx].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

pairplot = sns.pairplot(df, hue='species', palette='viridis', diag_kind='kde', height=2.5)
pairplot.fig.suptitle('Iris Feature Pairwise Relationships', y=1.02, fontsize=14)
plt.savefig('pairplot.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
corr = X.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training class distribution:\n{y_train.value_counts().sort_index()}")
print(f"Test class distribution:\n{y_test.value_counts().sort_index()}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nScaled Training Data Summary:\n{pd.DataFrame(X_train_scaled, columns=iris.feature_names).describe().round(3)}")

knn_baseline = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_baseline.fit(X_train_scaled, y_train)
y_pred_baseline = knn_baseline.predict(X_test_scaled)
baseline_acc = accuracy_score(y_test, y_pred_baseline)

print(f"\nBaseline KNN (k=5) Accuracy: {baseline_acc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_baseline, target_names=iris.target_names)}")

k_values = [1, 3, 5, 7, 9, 11]
cv_scores = []
cv_stds = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    cv_scores.append(scores.mean())
    cv_stds.append(scores.std())
    print(f'k={k:2d}: CV Accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})')

optimal_idx = np.argmax(cv_scores)
optimal_k = k_values[optimal_idx]
print(f'\nOptimal k: {optimal_k} (CV Accuracy: {cv_scores[optimal_idx]:.4f})')

plt.figure(figsize=(10, 6))
plt.errorbar(k_values, cv_scores, yerr=cv_stds, fmt='o-', capsize=5, color='steelblue', ecolor='lightcoral', linewidth=2, markersize=8)
plt.axvline(x=optimal_k, color='green', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
plt.xlabel('Number of Neighbors (k)', fontsize=12)
plt.ylabel('Cross-Validation Accuracy', fontsize=12)
plt.title('KNN Hyperparameter Tuning: k vs. Accuracy', fontsize=14)
plt.xticks(k_values)
plt.ylim([min(cv_scores) - 0.05, 1.05])
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('k_vs_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()

knn_final = KNeighborsClassifier(n_neighbors=optimal_k, metric='minkowski', p=2)
knn_final.fit(X_train_scaled, y_train)
y_pred = knn_final.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n===== FINAL MODEL EVALUATION =====")
print(f'Optimal k: {optimal_k}')
print(f'Accuracy:  {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1-Score:  {f1:.4f}')

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, square=True, linewidths=0.5)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'Confusion Matrix (k={optimal_k})', fontsize=14)
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names, digits=4))

new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3],
    [5.9, 3.0, 4.2, 1.5],
    [4.9, 3.1, 1.5, 0.1],
])

new_samples_scaled = scaler.transform(new_samples)
predictions = knn_final.predict(new_samples_scaled)
probabilities = knn_final.predict_proba(new_samples_scaled)

print("\n===== NEW SAMPLE PREDICTIONS =====")
for i, (sample, pred, prob) in enumerate(zip(new_samples, predictions, probabilities)):
    species = iris.target_names[pred]
    confidence = prob[pred]
    print(f'Sample {i+1}: {sample}')
    print(f'  Predicted: {species} (confidence: {confidence:.4f})')
    print(f'  Class probabilities: {dict(zip(iris.target_names, prob.round(4)))}')
    print()

results = []

knn_raw = KNeighborsClassifier(n_neighbors=optimal_k)
knn_raw.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, knn_raw.predict(X_test))
results.append(('No Scaling', acc_raw))

knn_std = KNeighborsClassifier(n_neighbors=optimal_k)
knn_std.fit(X_train_scaled, y_train)
acc_std = accuracy_score(y_test, knn_std.predict(X_test_scaled))
results.append(('StandardScaler', acc_std))

minmax = MinMaxScaler()
X_train_mm = minmax.fit_transform(X_train)
X_test_mm = minmax.transform(X_test)
knn_mm = KNeighborsClassifier(n_neighbors=optimal_k)
knn_mm.fit(X_train_mm, y_train)
acc_mm = accuracy_score(y_test, knn_mm.predict(X_test_mm))
results.append(('MinMaxScaler', acc_mm))

for metric in ['euclidean', 'manhattan', 'chebyshev']:
    knn_dist = KNeighborsClassifier(n_neighbors=optimal_k, metric=metric)
    knn_dist.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn_dist.predict(X_test_scaled))
    results.append((f'{metric.capitalize()} Distance', acc))

comparison_df = pd.DataFrame(results, columns=['Configuration', 'Test Accuracy'])
print("===== SCALING & DISTANCE METRIC COMPARISON =====")
print(comparison_df.to_string(index=False))

plt.figure(figsize=(10, 5))
colors = ['coral' if 'Scaling' in cfg or 'Distance' in cfg else 'steelblue' for cfg in comparison_df['Configuration']]
bars = plt.bar(comparison_df['Configuration'], comparison_df['Test Accuracy'], color=colors, edgecolor='black')
plt.ylim([0.8, 1.05])
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Impact of Scaling and Distance Metrics on KNN Performance', fontsize=14)
plt.xticks(rotation=15, ha='right')
for bar, acc in zip(bars, comparison_df['Test Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('scaling_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
