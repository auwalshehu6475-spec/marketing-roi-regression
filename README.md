# ============================================================
# Lumina HealthPath Capstone Project
# Binary Classification using Logistic Regression
# ============================================================

# ==========================
# 1. Import Libraries
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================
# 2. Load Dataset
# ==========================
# Replace "patients.csv" with your actual dataset filename
df = pd.read_csv("patients.csv")

# ==========================
# 3. Explore the Dataset
# ==========================
print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# ==========================
# 4. Handle Missing Values
# ==========================
imputer = SimpleImputer(strategy="median")

df[df.columns] = imputer.fit_transform(df)

# ==========================
# 5. Correlation Heatmap
# ==========================
plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.show()

# ==========================
# 6. Separate Features and Target
# ==========================
# Replace "Risk" with your target column if different
X = df.drop("Risk", axis=1)
y = df["Risk"]

# ==========================
# 7. Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# 8. Feature Scaling
# ==========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# 9. Build Logistic Regression Model
# ==========================
model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)

# ==========================
# 10. Make Predictions
# ==========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# ==========================
# 11. Evaluate Model
# ==========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n==============================")
print("Model Performance")
print("==============================")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================
# 12. Confusion Matrix
# ==========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stable", "High Risk"],
    yticklabels=["Stable", "High Risk"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==========================
# 13. Model Coefficients
# ==========================
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nFeature Importance (Coefficients)")
print(coefficients)

# ==========================
# 14. Business Check
# ==========================
if recall >= 0.85:
    print("\n✅ Business Requirement Met: Recall is at least 0.85")
else:
    print("\n❌ Business Requirement NOT Met: Recall is below 0.85")

# ==========================
# 15. Predict New Patient (Optional)
# ==========================
# Replace the values below with actual patient measurements
# Ensure the order matches your feature columns.

# new_patient = pd.DataFrame([[
#     45,     # Age
#     28.5,   # BMI
#     145,    # Glucose
#     5000    # Activity
# ]], columns=X.columns)

# new_patient_scaled = scaler.transform(new_patient)

# prediction = model.predict(new_patient_scaled)
# probability = model.predict_proba(new_patient_scaled)

# print("\nPrediction:", prediction[0])
# print("Probability:", probability)