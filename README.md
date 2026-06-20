import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# 1. Load dataset
df = pd.read_csv('966bb80f-b8a8-4a15-acd0-e5fa46ae7b6b.csv')

# 2. Drop identifier/index columns
df.drop(columns=['Unnamed: 0', 'id'], errors='ignore', inplace=True)

# 3. Drop rows with missing target
df.dropna(subset=['satisfaction'], inplace=True)

# 4. Handle missing values in all numeric columns
num_cols = df.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])

# (If any categorical columns have NaN after encoding, fill with mode before get_dummies)
cat_cols = df.select_dtypes(include='object').columns.difference(['satisfaction'])
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 5. Encode target and categorical features
df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'dissatisfied': 0})
df_encoded = pd.get_dummies(df, columns=['Customer Type', 'Type of Travel', 'Class'], drop_first=True)

# Rest of the code ... (same as original)

# After training dt, actually print metrics
dt_train_acc = accuracy_score(y_train, y_train_pred_dt)
dt_test_acc  = accuracy_score(y_test, y_test_pred_dt)
print(f"Decision Tree - Train Accuracy: {dt_train_acc:.3f}, Test Accuracy: {dt_test_acc:.3f}")

# For cleaner plot, show top-15 features
fi_df_top = fi_df.tail(15)
ax.barh(fi_df_top['Feature'], fi_df_top['Importance'], color='skyblue')