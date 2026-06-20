# %% [markdown]
# # Airline Customer Satisfaction – Random Forest Optimization
# 
# This notebook follows the required structure:
# 1. Load & clean data  
# 2. Three‑way split (Train/Val/Test)  
# 3. Hyperparameter tuning with `GridSearchCV` and `PredefinedSplit`  
# 4. Final evaluation on the test set  
# 5. Comparison with a baseline Decision Tree  
# 6. Feature importance visualisation
# 
# ## 1. Imports & Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# %%
# 2. Load the dataset
df = pd.read_csv('966bb80f-b8a8-4a15-acd0-e5fa46ae7b6b.csv')
print(f"Original shape: {df.shape}")

# %%
# 3. Data cleaning
# Drop columns that are clearly identifiers (if they exist)
df.drop(columns=['Unnamed: 0', 'id'], errors='ignore', inplace=True)

# Drop rows where the target is missing
df.dropna(subset=['satisfaction'], inplace=True)

# Impute missing values in numeric columns (e.g., Arrival Delay in Minutes)
num_cols = df.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Impute missing values in categorical columns (except target) with mode
cat_cols = df.select_dtypes(include='object').columns.difference(['satisfaction'])
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print(f"Shape after cleaning: {df.shape}")
print("Remaining missing values per column:")
print(df.isnull().sum().sum())  # should be 0

# %%
# 4. Encode target and categorical features
df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'dissatisfied': 0})
# One‑hot encode categorical predictors (drop first to avoid multicollinearity)
df_encoded = pd.get_dummies(
    df,
    columns=['Customer Type', 'Type of Travel', 'Class'],
    drop_first=True
)

# Separate features and target
X = df_encoded.drop(columns=['satisfaction'])
y = df_encoded['satisfaction']
print(f"Features shape: {X.shape}, Target distribution:\n{y.value_counts(normalize=True)}")

# %%
# 5. Three‑Way Data Split (60% Train, 20% Validation, 20% Test)
# First split off the test set (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
# Then split the remaining 80% into train (60% of total) and validation (20% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print(f"Train size: {X_train.shape[0]}")
print(f"Validation size: {X_val.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# %%
# 6. Create PredefinedSplit for hyperparameter tuning
# Stack train and validation together (GridSearchCV will split them according to test_fold)
X_grid = pd.concat([X_train, X_val]).reset_index(drop=True)
y_grid = pd.concat([y_train, y_val]).reset_index(drop=True)

# test_fold: -1 for training, 0 for validation
test_fold = np.concatenate([
    -np.ones(X_train.shape[0]),          # -1 -> training indices
    np.zeros(X_val.shape[0])             #  0 -> validation fold
])
ps = PredefinedSplit(test_fold)

# %%
# 7. Define parameter grid and execute GridSearchCV
param_grid = {
    'max_depth': [10, 20, None],
    'n_estimators': [50, 100],
    'min_samples_leaf': [1, 4]
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=ps,
    scoring='f1',          # F1-score for imbalanced classes
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_grid, y_grid)

print("\nBest hyperparameters:", grid_search.best_params_)
print(f"Best cross‑validation F1 (on validation fold): {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_

# %%
# 8. Evaluate the optimized Random Forest on the test set
y_pred_rf = best_rf.predict(X_test)

rf_accuracy  = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall    = recall_score(y_test, y_pred_rf)
rf_f1        = f1_score(y_test, y_pred_rf)

print("Random Forest – Test Set Performance:")
print(f"  Accuracy : {rf_accuracy:.4f}")
print(f"  Precision: {rf_precision:.4f}")
print(f"  Recall   : {rf_recall:.4f}")
print(f"  F1‑score : {rf_f1:.4f}")

# %%
# 9. Baseline Decision Tree for comparison
dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt.fit(X_train, y_train)

y_train_pred_dt = dt.predict(X_train)
y_test_pred_dt  = dt.predict(X_test)

dt_train_acc = accuracy_score(y_train, y_train_pred_dt)
dt_test_acc  = accuracy_score(y_test, y_test_pred_dt)
dt_test_f1   = f1_score(y_test, y_test_pred_dt)

print("Decision Tree (baseline) – Performance:")
print(f"  Train Accuracy : {dt_train_acc:.4f}")
print(f"  Test Accuracy  : {dt_test_acc:.4f}")
print(f"  Test F1‑score  : {dt_test_f1:.4f}")

# %%
# 10. Feature Importances (Random Forest)
importances = best_rf.feature_importances_
fi_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

# Plot top 15 features
top_n = 15
fi_df_top = fi_df.tail(top_n)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(fi_df_top['Feature'], fi_df_top['Importance'], color='skyblue')
ax.set_xlabel('Importance Score')
ax.set_title(f'Top {top_n} Random Forest Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=150)
plt.show()

# %%
# 11. Summary statistics for comparison (can be printed or displayed)
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
print(f"{'Model':<20} {'Test Accuracy':>13} {'Test F1‑score':>13}")
print("-"*50)
print(f"{'Random Forest (tuned)':<20} {rf_accuracy:>13.4f} {rf_f1:>13.4f}")
print(f"{'Decision Tree (baseline)':<20} {dt_test_acc:>13.4f} {dt_test_f1:>13.4f}")