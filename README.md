import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import StringIO

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# STEP 1: LOAD DATA
# ==========================================

columns = [
    'age', 'menopause', 'tumor-size', 'inv-nodes',
    'node-caps', 'deg-malig', 'breast', 'breast-quad',
    'irradiat', 'Class'
]

df = pd.read_csv(
    StringIO(raw_data),
    header=None,
    names=columns,
    skipinitialspace=True
)

# Clean quotes + missing values
df = df.apply(lambda col: col.str.replace("'", "", regex=False))
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Convert numeric column properly
df['deg-malig'] = df['deg-malig'].astype(int)

# ==========================================
# STEP 2: SPLIT FEATURES / TARGET
# ==========================================

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# STEP 3: PREPROCESSING PIPELINE
# ==========================================

categorical_features = [
    'age', 'menopause', 'tumor-size', 'inv-nodes',
    'node-caps', 'breast', 'breast-quad', 'irradiat'
]

numeric_features = ['deg-malig']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# ==========================================
# STEP 4: MODEL + PIPELINE
# ==========================================

def create_model(k):
    return Pipeline(steps=[
        ('preprocess', preprocessor),
        ('knn', KNeighborsClassifier(
            n_neighbors=k,
            metric='hamming',
            weights='distance'
        ))
    ])

# ==========================================
# STEP 5: CROSS-VALIDATION FOR BEST K
# ==========================================

k_values = [1, 3, 5, 7, 9, 11]
cv_scores = []

for k in k_values:
    model = create_model(k)
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='f1_weighted'
    )
    cv_scores.append(scores.mean())

optimal_k = k_values[int(np.argmax(cv_scores))]

print("=== Hyperparameter Tuning Results ===")
for k, score in zip(k_values, cv_scores):
    print(f"k = {k}: F1-score = {score:.4f}")

print(f"\nOptimal k selected: {optimal_k}\n")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_scores, marker='o')
plt.title("K vs Cross-Validated F1 Score")
plt.xlabel("K Value")
plt.ylabel("F1 Score")
plt.grid(True)
plt.show()

# ==========================================
# STEP 6: FINAL MODEL TRAINING
# ==========================================

final_model = create_model(optimal_k)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

# ==========================================
# STEP 7: EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='recurrence-events', zero_division=0)
recall = recall_score(y_test, y_pred, pos_label='recurrence-events', zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label='recurrence-events', zero_division=0)

print("=== Final Model Performance ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# STEP 8: SAFE INFERENCE (NEW DATA)
# ==========================================

sample = X_test.iloc[[0]].copy()   # safe structure-preserving sample

prediction = final_model.predict(sample)

print("\n=== Inference ===")
print("Predicted class:", prediction[0])