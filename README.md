import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ==========================================================
# 1. LOAD DATA
# ==========================================================
df = pd.read_csv("0f484464-3cc8-4bfc-968b-b1a9fc4d4b1d.csv")

drop_cols = [c for c in ['Unnamed: 0', 'name'] if c in df.columns]
df_cleaned = df.drop(columns=drop_cols)

# ==========================================================
# 2. TARGET DISTRIBUTION
# ==========================================================
plt.figure(figsize=(6, 4))
sns.countplot(data=df_cleaned, x='target_5yrs', palette='viridis')
plt.title("Target Distribution")
plt.show()

print(df_cleaned['target_5yrs'].value_counts())

# ==========================================================
# 3. MISSING VALUES
# ==========================================================
print("\nMissing values:")
print(df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0])

# ==========================================================
# 4. FEATURE ENGINEERING (SAFE)
# ==========================================================
if 'min' in df_cleaned.columns:
    df_cleaned['min'] = df_cleaned['min'].replace(0, np.nan)

if 'pts' in df_cleaned.columns:
    df_cleaned['pts_per_min'] = df_cleaned['pts'] / df_cleaned['min']

df_cleaned['efficiency_rating'] = (
    df_cleaned[['pts','reb','ast','stl','blk']].sum(axis=1)
    - df_cleaned['tov']
)

df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cleaned['pts_per_min'] = df_cleaned['pts_per_min'].fillna(0)

# ==========================================================
# 5. CORRELATION (FIXED)
# ==========================================================
features_only = df_cleaned.drop(columns=['target_5yrs'])
corr_matrix = features_only.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.show()

upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

redundant_features = [
    col for col in upper_tri.columns
    if any(upper_tri[col] > 0.90)
]

print("\nHighly correlated features:", redundant_features)

df_reduced = df_cleaned.drop(columns=redundant_features)

# ==========================================================
# 6. FEATURE MATRIX (NO LEAKAGE FIX)
# ==========================================================
X = df_reduced.drop(columns=['target_5yrs'])
y = df_reduced['target_5yrs']

# numeric-only safety
X = X.select_dtypes(include=[np.number])

# Imputer + scaler (OK for now, but ideally inside pipeline later)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_final = pd.DataFrame(X_scaled, columns=X.columns)

print("\nFinal shape:", X_final.shape)
print(X_final.head())