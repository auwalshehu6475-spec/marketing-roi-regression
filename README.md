import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load data
df = pd.read_csv("data.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop unnecessary columns
df_cleaned = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Normalize target
df_cleaned['diagnosis'] = df_cleaned['diagnosis'].astype(str).str.strip().str.upper()
df_cleaned['diagnosis'] = df_cleaned['diagnosis'].map({'M': 1, 'B': 0})

# Remove rows where target became NaN
df_cleaned = df_cleaned.dropna(subset=['diagnosis'])

# Features and target
X = df_cleaned.drop(columns=['diagnosis'])
X = X.select_dtypes(include=[np.number])  # keep numeric only
y = df_cleaned['diagnosis']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
log_reg = LogisticRegression(max_iter=2000, solver='liblinear')
log_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred = log_reg.predict(X_test_scaled)

# Metrics
print("--- Model Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix")
plt.show()