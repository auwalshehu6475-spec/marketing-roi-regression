# %%
# ============================================
# Cell 1: Imports and Setup
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve, auc,
    make_scorer,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42

# %%
# ============================================
# Cell 2: Load and Explore Data
# ============================================
DATA_PATH = "../data/engineered_nba_players.csv"
TARGET_COL = "target_5yrs"
df = pd.read_csv(DATA_PATH)

print("First five rows:")
display(df.head())
print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names: {list(df.columns)}")
print("\nData types:")
print(df.dtypes)

# Check missing values
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

# %%
# ============================================
# Cell 3: Feature/Target Preparation
# ============================================
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

# %%
# ============================================
# Cell 4: Train-Test Split
# ============================================
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

# %%
# ============================================
# Cell 5: Model Training (Baseline GaussianNB)
# ============================================
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("First 10 predictions: ", y_pred[:10])
print("First 10 actuals:     ", y_test.values[:10])

# %%
# ============================================
# Cell 6: Evaluation Metrics
# ============================================
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

# %%
# ============================================
# Cell 7: Markdown – The "Naive" Assumption
# ============================================
# %% [markdown]
# ### The “Naive” Assumption of Feature Independence
#
# Naive Bayes classifiers assume that all features are conditionally independent given the class label. In basketball data, this is unrealistic because statistics like points, rebounds, and assists are naturally correlated. For example, players who score more often tend to play more minutes, which increases their rebound and assist numbers.
#
# #### Why Does Naive Bayes Still Work?
#
# Even when the independence assumption is violated, Naive Bayes often performs well because:
# - It focuses on **relative probabilities** rather than exact estimates.
# - The simplification reduces overfitting, especially with small datasets.
# - The Gaussian variant handles continuous features naturally.
#
# However, correlated features can distort probability estimates. In our dataset, `total_points` and `efficiency` are correlated (r=0.59), and `total_points` is correlated with `tov` (0.84) and `reb` (0.68). These high correlations violate the assumption. To mitigate this, we could apply feature selection, PCA, or switch to models that explicitly handle dependencies (e.g., logistic regression, random forests). For now, we accept the limitation and interpret results with this caveat.

# %%
# ============================================
# Cell 8: Markdown – Precision/Recall Analysis
# ============================================
# %% [markdown]
# ### Precision vs. Recall – The Scouting Dilemma
#
# | Metric         | Class 0 (< 5 yrs) | Class 1 (≥ 5 yrs) |
# |----------------|-------------------|-------------------|
# | **Precision**  | 0.52              | 0.80              |
# | **Recall**     | 0.77              | 0.55              |
# | **F1‑score**   | 0.62              | 0.65              |
#
# **Interpretation for NBA decision‑makers:**
#
# - **Precision for class 1 (long career):** 0.80 → When the model predicts a player will stay ≥5 years, it is correct 80% of the time. This means we can confidently invest resources (draft picks, development) in players flagged as “safe bets.”
# - **Recall for class 1:** 0.55 → The model misses 45% of the players who actually go on to have long careers. These are **False Negatives (FN)** – potential stars that we overlook.
# - **Precision for class 0 (short career):** 0.52 → When the model predicts a short career, it’s only right about half the time. Many players who are actually long‑term contributors are misclassified as busts.
# - **Recall for class 0:** 0.77 → We correctly identify 77% of the players who will wash out early – good for avoiding wasted draft picks.
#
# #### The Trade‑off: Missing a Star vs. Over‑investing in a Bust
#
# - **False Negative (FN = 74):** Players we think will leave early but actually have long careers. Cost: losing out on future All‑Stars or solid rotation players – the **“hidden gem”** error.
# - **False Positive (FP = 23):** Players we predict to be successful but who fade quickly. Cost: wasted draft capital, salary, and development time – the **“bust”** error.
#
# The model currently leans toward **higher recall for the short‑career class** (0.77) at the expense of **lower recall for the long‑career class** (0.55). This means we are more cautious about predicting success – we prefer to call a player a “bust” rather than over‑promise. In practice, this approach may be safer for a team that values stability and wants to avoid expensive mistakes, but it risks missing undervalued talent.
#
# **Recommendation:** Adjust the decision threshold. If we lower the threshold for class 1 (predict long career more easily), we will increase recall for class 1 (catch more stars) but also increase false positives (more busts). The optimal balance depends on the team’s risk appetite and budget.

# %%
# ============================================
# Cell 9: Cross-Validation and Model Robustness
# ============================================
# Use stratified k-fold cross-validation to get stable estimates
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring=make_scorer(recall_score, pos_label=1))
print(f"CV Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
print(f"CV Recall (class 1): {cv_recall.mean():.3f} ± {cv_recall.std():.3f}")

# %%
# ============================================
# Cell 10: Threshold Tuning for Business Trade-off
# ============================================
# We can adjust the decision threshold to favor recall or precision
# Using the predicted probabilities, we can evaluate different thresholds
thresholds = np.linspace(0.1, 0.9, 9)
metrics_df_thresh = []
for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    prec_thresh = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec_thresh = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1_thresh = f1_score(y_test, y_pred_thresh, zero_division=0)
    metrics_df_thresh.append({"Threshold": thresh, "Precision": prec_thresh, "Recall": rec_thresh, "F1": f1_thresh})

thresh_df = pd.DataFrame(metrics_df_thresh)
print("Threshold tuning results:")
display(thresh_df)

# Plot threshold trade-off
plt.figure(figsize=(8,5))
plt.plot(thresh_df["Threshold"], thresh_df["Precision"], label="Precision", marker='o')
plt.plot(thresh_df["Threshold"], thresh_df["Recall"], label="Recall", marker='o')
plt.plot(thresh_df["Threshold"], thresh_df["F1"], label="F1", marker='o')
plt.xlabel("Decision Threshold (class 1)")
plt.ylabel("Score")
plt.title("Precision/Recall Trade-off vs. Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../images/threshold_tradeoff.png", dpi=150, bbox_inches="tight")
plt.show()

# Choose a threshold that maximizes F1 (or a business-defined metric)
best_idx = thresh_df["F1"].idxmax()
best_thresh = thresh_df.loc[best_idx, "Threshold"]
print(f"Optimal threshold for F1: {best_thresh:.2f}")
# Evaluate with that threshold
y_pred_opt = (y_pred_proba >= best_thresh).astype(int)
print(classification_report(y_test, y_pred_opt, target_names=["< 5 yrs (0)", ">= 5 yrs (1)"]))

# %%
# ============================================
# Cell 11: Compare with Other Models (Logistic Regression, Random Forest)
# ============================================
# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
}
results = {}
for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)
    y_pred_cv = clf.predict(X_test_scaled)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred_cv),
        "Precision": precision_score(y_test, y_pred_cv),
        "Recall": recall_score(y_test, y_pred_cv),
        "F1": f1_score(y_test, y_pred_cv),
    }
results_df = pd.DataFrame(results).T
print("Comparison with other models:")
display(results_df)

# %%
# ============================================
# Cell 12: Stakeholder Summary
# ============================================
# %% [markdown]
# ## Stakeholder Summary – Actionable Insights for NBA Decision‑Makers
#
# ### Key Takeaways
#
# 1. **Model Performance:**  
#    The Gaussian Naive Bayes model correctly identifies 64% of all players (accuracy). It is **better at recognising players who will have short careers** (recall 77%) than those who will have long careers (recall 55%). This means the model is a useful **screening tool** to filter out likely busts, but it should not be the sole basis for investment in a prospect.
#
# 2. **Actionable Recommendations:**
#
#    - **Use as a First Filter:** Apply the model to draft candidates and free agents. Players predicted as “short career” should be examined more closely – they may still be undervalued gems that the model misclassifies (the 74 false negatives). Scouts should focus extra resources on these “red‑flagged” prospects.
#
#    - **Adjust the Threshold for Risk Tolerance:**  
#      *Risk‑averse strategy (avoid busts):* Keep the current threshold or even raise it to increase precision for class 1 – you’ll sign fewer players, but those you sign are more likely to pan out.  
#      *Risk‑seeking strategy (find stars):* Lower the threshold to increase recall for class 1 – you’ll sign more players, but accept more misses. Use this if your team has ample cap space and development resources.
#
#    - **Incorporate Additional Data:** The model currently uses only basic box‑score statistics. To improve, include:
#      - Advanced metrics (PER, WS/48, BPM).
#      - College/overseas performance and scouting grades.
#      - Injury history and physical measurements.
#      - Work ethic and psychological assessments (qualitative).
#
# 3. **Model Limitations:**
#    - The independence assumption is violated due to correlations between stats. Consider using more flexible models (e.g., logistic regression with interaction terms, random forests, or XGBoost) to capture these relationships.
#    - Class imbalance (62% long‑career, 38% short‑career) may bias the model. Future work could apply **oversampling (SMOTE)** or **class‑weighting** to improve recall for the minority class.
#
# 4. **Next Steps:**
#    - **Cross‑validation:** We already applied it; the model is stable (accuracy ~0.63, recall ~0.55).
#    - **Feature engineering:** Create derived features like usage rate, efficiency ratios, or positional averages.
#    - **Compare models:** Logistic regression and random forest already show similar or better performance; consider ensemble methods.
#    - **Calibrate probabilities:** Use Platt scaling or isotonic regression to make predicted probabilities more reliable for decision‑making.
#
# **Final Word:** This model is a solid starting point. Its strength lies in its simplicity and interpretability. By combining it with scouting expertise and additional data, it can become a powerful decision‑support tool for roster construction.

# %%
# ============================================
# Cell 13: Additional EDA (Correlations, Distributions, Boxplots)
# ============================================
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