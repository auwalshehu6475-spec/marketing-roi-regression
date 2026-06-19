import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score
)

# =====================================================
# 1. Load Dataset
# =====================================================

df = pd.read_csv("617ec7a0-b7f1-423e-b810-23f59803ffb6.csv")

# =====================================================
# 2. Clean Column Names
# =====================================================

df.columns = df.columns.str.strip()

# =====================================================
# 3. Remove Unnecessary ID Columns (if present)
# =====================================================

for col in ['Unnamed: 0', 'id']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# =====================================================
# 4. Inspect Dataset
# =====================================================

print("Dataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nOriginal Satisfaction Distribution:")
print(df['satisfaction'].value_counts())

# =====================================================
# 5. Handle Missing Values
# =====================================================

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# =====================================================
# 6. Encode Target Variable
# =====================================================

df['satisfaction'] = (
    df['satisfaction']
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        'satisfied': 1,
        'neutral or dissatisfied': 0
    })
)

# Remove rows with unmapped values if any
df = df.dropna(subset=['satisfaction'])

df['satisfaction'] = df['satisfaction'].astype(int)

print("\nEncoded Satisfaction Distribution:")
print(df['satisfaction'].value_counts(normalize=True))

# =====================================================
# 7. Encode Categorical Variables
# =====================================================

categorical_cols = ['Customer Type', 'Type of Travel', 'Class']

df_encoded = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

# =====================================================
# 8. Separate Features and Target
# =====================================================

X = df_encoded.drop(columns='satisfaction')
y = df_encoded['satisfaction']

# Ensure all columns are numeric
X = X.astype(float)

# Save feature names
feature_names = X.columns

# =====================================================
# 9. Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# 10. Scale Features
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 11. Build Logistic Regression Model
# =====================================================

log_reg = LogisticRegression(
    max_iter=3000,
    random_state=42,
    solver='lbfgs'
)

log_reg.fit(X_train_scaled, y_train)

# =====================================================
# 12. Make Predictions
# =====================================================

y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# =====================================================
# 13. Evaluate Model
# =====================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n--- Model Performance Metrics ---")

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print("\nClassification Report:")

print(classification_report(y_test, y_pred))

# =====================================================
# 14. Confusion Matrix
# =====================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Dissatisfied/Neutral', 'Satisfied'],
    yticklabels=['Dissatisfied/Neutral', 'Satisfied']
)

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()

# =====================================================
# 15. Feature Importance (Coefficients)
# =====================================================

coefficients = log_reg.coef_[0]

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Odds Ratio': np.exp(coefficients)
})

coef_df = coef_df.sort_values(
    by='Coefficient',
    ascending=False
)

print("\n--- Top Drivers of Customer Satisfaction ---")

print(coef_df)