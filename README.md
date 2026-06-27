# ============================================================
# TikTok Capstone Project
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

from imblearn.over_sampling import SMOTE

# ==========================
# 2. Load Dataset
# ==========================
df = pd.read_csv("tiktok_dataset.csv")

# ==========================
# 3. Explore Dataset
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
# 4. Drop Missing Values
# ==========================
df = df.dropna()

print("\nDataset Shape After Dropping Missing Values:")
print(df.shape)

# ==========================
# 5. Feature Engineering
# Create transcription length
# ==========================
df["transcription_length"] = (
    df["video_transcription_text"]
    .astype(str)
    .str.len()
)

# ==========================
# 6. One-Hot Encode Categorical Variables
# ==========================
df = pd.get_dummies(
    df,
    columns=["claim_status", "author_ban_status"],
    drop_first=True
)

# ==========================
# 7. Encode Target Variable
# ==========================
df["verified_status"] = df["verified_status"].map({
    "verified": 1,
    "not verified": 0
})

# ==========================
# 8. Correlation Heatmap
# ==========================
plt.figure(figsize=(12,10))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# ==========================
# 9. Prepare Features & Target
# ==========================
columns_to_drop = [
    "verified_status",
    "video_transcription_text"
]

# Drop ID columns if they exist
for col in ["#", "video_id"]:
    if col in df.columns:
        columns_to_drop.append(col)

X = df.drop(columns=columns_to_drop)
y = df["verified_status"]

# ==========================
# 10. Handle Class Imbalance
# ==========================
smote = SMOTE(random_state=42)

X, y = smote.fit_resample(X, y)

print("\nClass Distribution After SMOTE")
print(y.value_counts())

# ==========================
# 11. Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# 12. Feature Scaling
# ==========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# 13. Build Logistic Regression Model
# ==========================
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)

# ==========================
# 14. Predictions
# ==========================
y_pred = model.predict(X_test)

# ==========================
# 15. Evaluation
# ==========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n==============================")
print("Model Performance")
print("==============================")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================
# 16. Confusion Matrix
# ==========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Verified","Verified"],
    yticklabels=["Not Verified","Verified"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# ==========================
# 17. Logistic Regression Coefficients
# ==========================
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nFeature Coefficients")
print(coefficients)

# ==========================
# 18. Business Interpretation
# ==========================
print("\nBusiness Insights")
print("- Longer video transcriptions may indicate more detailed content.")
print("- Positive coefficients increase the likelihood of a creator being verified.")
print("- Negative coefficients decrease the likelihood of verification.")
print("- Videos containing factual claims may require additional moderation.")
print("- Engagement-related features can help TikTok prioritize creator verification.")
print("- SMOTE balanced the 94/6 class distribution, improving the model's ability to identify verified creators.")
print("- F1-score is emphasized because the original dataset is highly imbalanced, making it a more reliable metric than accuracy.")
print("- This model can help TikTok reduce verification review time and better prioritize moderation resources.")