import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("0f484464-3cc8-4bfc-968b-b1a9fc4d4b1d.csv")

# Drop useless columns
drop_cols = [c for c in ['Unnamed: 0', 'name'] if c in df.columns]
df = df.drop(columns=drop_cols)

# Ensure target exists
assert 'target_5yrs' in df.columns, "target_5yrs column not found"

# -----------------------------
# 2. Basic Feature Engineering (safe + leakage-free)
# -----------------------------
def feature_engineering(data):
    data = data.copy()

    # Avoid division errors
    if 'min' in data.columns and 'pts' in data.columns:
        data['min'] = data['min'].replace(0, np.nan)
        data['pts_per_min'] = data['pts'] / data['min']

    # Efficiency metric
    cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov']
    if all(c in data.columns for c in cols):
        data['efficiency_rating'] = (
            data['pts'] + data['reb'] + data['ast'] +
            data['stl'] + data['blk']
        ) - data['tov']

    return data

df = feature_engineering(df)

# -----------------------------
# 3. Split Data
# -----------------------------
X = df.drop(columns=['target_5yrs'])
y = df['target_5yrs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 4. Preprocessing Pipeline
# -----------------------------
numeric_features = X.select_dtypes(include=[np.number]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features)
    ],
    remainder="drop"
)

# -----------------------------
# 5. Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# 6. Full Pipeline (IMPORTANT PART)
# -----------------------------
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# -----------------------------
# 7. Train
# -----------------------------
clf.fit(X_train, y_train)

# -----------------------------
# 8. Evaluate
# -----------------------------
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\n--- MODEL PERFORMANCE ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))