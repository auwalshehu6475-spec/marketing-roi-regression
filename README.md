# ==============================
# Import libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42

# ==============================
# Load data
# ==============================
DATA_PATH = "../data/engineered_nba_players.csv"
TARGET_COL = "target_5yrs"
df = pd.read_csv(DATA_PATH)

# Display first rows and info
print("First five rows:")
display(df.head())
print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names: {list(df.columns)}")
print("\nData types:")
print(df.dtypes)

# Check missing and duplicates
print("\nMissing values per column:")
print(df.isnull().sum())
n_duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {n_duplicates}")

# Summary statistics
display(df.describe().T)

# Class distribution
class_counts = df[TARGET_COL].value_counts().sort_index()
class_pct = (class_counts / len(df) * 100).round(1)
summary = pd.DataFrame({"count": class_counts, "percentage": class_pct})
summary.index = ["0 (left league < 5 yrs)", "1 (stayed >= 5 yrs)"]
display(summary)

# ==============================
# Prepare features and target
# ==============================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != TARGET_COL]
print(f"Detected {len(feature_cols)} continuous numeric features:")
print(feature_cols)
X = df[feature_cols].copy()
y = df[TARGET_COL].copy()

# Impute missing (if any)
missing_total = X.isnull().sum().sum()
if missing_total > 0:
    print(f"Imputing {missing_total} missing values with column medians.")
    X = X.fillna(X.median())
else:
    print("No missing values detected in the feature set.")

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"Training set: {X_train.shape[0]} rows")
print(f"Testing set:  {X_test.shape[0]} rows")
print("\nClass balance, training set:")
print(y_train.value_counts(normalize=True).round(3))
print("\nClass balance, testing set:")
print(y_test.value_counts(normalize=True).round(3))

# ==============================
# Train Gaussian Naive Bayes
# ==============================
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
print("First 10 predictions: ", y_pred[:10])
print("First 10 actuals:     ", y_test.values[:10])

# ==============================
# Evaluation
# ==============================
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
cm_df = pd.DataFrame(
    cm,
    index=["Actual: < 5 yrs (0)", "Actual: >= 5 yrs (1)"],
    columns=["Predicted: < 5 yrs (0)", "Predicted: >= 5 yrs (1)"],
)
print("Confusion Matrix:")
display(cm_df)
print(f"\nTrue Negatives  (correctly flagged short career): {tn}")
print(f"False Positives (predicted bust risk... predicted long, was short): {fp}")
print(f"False Negatives (predicted short, was actually long career): {fn}")
print(f"True Positives  (correctly flagged long career): {tp}")

# Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", cbar=True,
    xticklabels=["Predicted < 5 yrs", "Predicted >= 5 yrs"],
    yticklabels=["Actual < 5 yrs", "Actual >= 5 yrs"],
)
plt.title("Confusion Matrix — Gaussian Naive Bayes", fontsize=13, fontweight="bold")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("../images/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# Classification report
report = classification_report(y_test, y_pred, target_names=["< 5 yrs (0)", ">= 5 yrs (1)"])
print(report)

# Metrics DataFrame
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Value": [acc, prec, rec, f1],
})
metrics_df["Value"] = metrics_df["Value"].round(3)
display(metrics_df)

# ==============================
# Additional EDA visualizations
# ==============================

# Correlation heatmap
corr = df[feature_cols + [TARGET_COL]].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap of Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../images/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# Top correlated pairs
corr_pairs = (
    corr.where(~np.eye(corr.shape[0], dtype=bool))
    .stack()
    .reset_index()
)
corr_pairs.columns = ["feature_1", "feature_2", "correlation"]
corr_pairs["abs_correlation"] = corr_pairs["correlation"].abs()
corr_pairs = corr_pairs.sort_values("abs_correlation", ascending=False)
corr_pairs = corr_pairs[corr_pairs["feature_1"] < corr_pairs["feature_2"]]
display(corr_pairs.head(10).reset_index(drop=True))

# Class means and variances
class_means = df.groupby(TARGET_COL)[feature_cols].mean().T
class_means.columns = ["mean_class_0", "mean_class_1"]
class_vars = df.groupby(TARGET_COL)[feature_cols].var().T
class_vars.columns = ["var_class_0", "var_class_1"]
stats_df = pd.concat([class_means, class_vars], axis=1)
display(stats_df)

# Standardized mean difference (effect size)
pooled_std = np.sqrt((stats_df["var_class_0"] + stats_df["var_class_1"]) / 2)
mean_diff = (stats_df["mean_class_1"] - stats_df["mean_class_0"])
stats_df["standardized_mean_difference"] = (mean_diff / pooled_std).round(3)
ranked_features = (
    stats_df[["mean_class_0", "mean_class_1", "standardized_mean_difference"]]
    .sort_values("standardized_mean_difference", key=lambda s: s.abs(), ascending=False)
)
display(ranked_features)

# Class distribution bar plot
plt.figure(figsize=(6, 5))
ax = sns.countplot(x=TARGET_COL, data=df, hue=TARGET_COL, palette="Set2", legend=False)
ax.set_xticklabels(["< 5 yrs (0)", ">= 5 yrs (1)"])
for container in ax.containers:
    ax.bar_label(container)
plt.title("Class Distribution: target_5yrs", fontsize=13, fontweight="bold")
plt.xlabel("Outcome")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.savefig("../images/class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# Feature distributions by class (histograms + KDE)
ncols = 3
nrows = int(np.ceil(len(feature_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.8))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    sns.histplot(
        data=df, x=col, hue=TARGET_COL, kde=True, ax=axes[i],
        palette={0: "#d62728", 1: "#1f77b4"}, element="step", stat="density", common_norm=False,
    )
    axes[i].set_title(f"Distribution of {col} by Class", fontsize=10)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")
for j in range(len(feature_cols), len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("Feature Distributions by Outcome Class", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../images/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# Boxplots for top features
top_features = ranked_features.index[:6].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(top_features):
    sns.boxplot(x=TARGET_COL, y=col, data=df, hue=TARGET_COL, palette="Set2", legend=False, ax=axes[i])
    axes[i].set_xticklabels(["< 5 yrs", ">= 5 yrs"])
    axes[i].set_title(f"{col} by Outcome", fontsize=11)
    axes[i].set_xlabel("")
fig.suptitle("Top Informative Features: Boxplots by Class", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../images/boxplots_by_class.png", dpi=150, bbox_inches="tight")
plt.show()