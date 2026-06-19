import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ==========================================
# 1. DATA EXPLORATION
# ==========================================
df = pd.read_csv("0f484464-3cc8-4bfc-968b-b1a9fc4d4b1d.csv")

# Drop noise columns
columns_to_drop = [col for col in ['Unnamed: 0', 'name'] if col in df.columns]
df_cleaned = df.drop(columns=columns_to_drop)

# Class distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df_cleaned, x='target_5yrs', palette='viridis')
plt.title('Distribution of Target Class (>= 5 Years Career)')
plt.xlabel('Target (0 = <5 yrs, 1 = >=5 yrs)')
plt.ylabel('Count')
plt.show()

# Class balance print
class_counts = df_cleaned['target_5yrs'].value_counts()
print("--- Target Class Distribution ---")
print(class_counts)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

# Safe division
if 'min' in df_cleaned.columns:
    df_cleaned['min'] = df_cleaned['min'].replace(0, np.nan)

if 'pts' in df_cleaned.columns and 'min' in df_cleaned.columns:
    df_cleaned['pts_per_min'] = df_cleaned['pts'] / df_cleaned['min']
    df_cleaned['pts_per_min'] = df_cleaned['pts_per_min'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Efficiency rating (only if columns exist)
cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov']
if all(c in df_cleaned.columns for c in cols):
    df_cleaned['efficiency_rating'] = (
        df_cleaned['pts'] +
        df_cleaned['reb'] +
        df_cleaned['ast'] +
        df_cleaned['stl'] +
        df_cleaned['blk']
    ) - df_cleaned['tov']

# ==========================================
# 3. CORRELATION ANALYSIS (FIXED)
# ==========================================

features_only = df_cleaned.drop(columns=['target_5yrs'])

corr = features_only.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.show()