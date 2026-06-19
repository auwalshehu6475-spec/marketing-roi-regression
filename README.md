# ==========================================
# Multiple Linear Regression Marketing Analysis
# ==========================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==========================================
# 2. Load and Clean Dataset
# ==========================================

df = pd.read_csv("c107fa55-45be-4f9c-8f61-088e88d1fb0c.csv")

# Clean column names (strip whitespace and replace spaces)
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Remove missing values
df_clean = df.dropna()

print(f"Dataset Shape: {df_clean.shape}")
print(f"\nColumns: {list(df_clean.columns)}")
print("\nFirst 5 Rows:")
print(df_clean.head())

# ==========================================
# 3. Exploratory Data Analysis & Correlation
# ==========================================

print("\nSummary Statistics:")
print(df_clean.describe())

plt.figure(figsize=(8, 6))
# Filter numeric columns for correlation matrix
numeric_cols = ['TV', 'Radio', 'Social_Media', 'Sales']
sns.heatmap(
    df_clean[numeric_cols].corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)
plt.title('Correlation Matrix')
plt.show()

# ==========================================
# 4. Multicollinearity Check (VIF)
# ==========================================

# Select the required independent variables as continuous features
features = ['TV', 'Radio', 'Social_Media']
X_vif = df_clean[features]

# Add constant for VIF calculation and cast to float for stability
X_vif_const = sm.add_constant(X_vif).astype(float)

vif = pd.DataFrame()
vif["Feature"] = X_vif_const.columns
vif["VIF"] = [
    variance_inflation_factor(X_vif_const.values, i)
    for i in range(X_vif_const.shape[1])
]

print("\n========== VIF RESULTS ==========")
print(vif.round(4))
print("\nInterpretation:\nVIF < 5 = acceptable\nVIF > 5 = moderate multicollinearity\nVIF > 10 = severe multicollinearity")

# ==========================================
# 5. Build Multiple Linear Regression Model
# ==========================================

# Define features (X) and target variable (y)
X = df_clean[features]
y = df_clean['Sales']

# Add an intercept term (OBLIGATORY for statsmodels OLS)
X = sm.add_constant(X)

# Fit the Multiple Linear Regression model
mlr_model = sm.OLS(y, X).fit()

print("\n========== MLR SUMMARY ==========")
print(mlr_model.summary())

# ==========================================
# 6. Adjusted R-Squared & F-Statistic Evaluation
# ==========================================

r2 = mlr_model.rsquared
adj_r2 = mlr_model.rsquared_adj
f_stat = mlr_model.fvalue
f_pval = mlr_model.f_pvalue

print("\n========== MODEL PERFORMANCE ==========")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")
print(f"F-statistic: {f_stat:.4f} (p-value: {f_pval:.4e})")
print(f"\nInterpretation: The model explains {adj_r2*100:.2f}% of the variation in Sales after accounting for multiple predictors.")

# ==========================================
# 7. Assumption Checks (Residual Diagnostics)
# ==========================================

residuals = mlr_model.resid
fitted = mlr_model.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residual plot (Homoscedasticity & Linearity Check)
sns.scatterplot(x=fitted, y=residuals, alpha=0.6, ax=axes[0])
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_title('Residuals vs Fitted')
axes[0].set_xlabel('Predicted Sales')
axes[0].set_ylabel('Residuals')

# Q-Q Plot (Normality Check - scaled correctly)
sm.qqplot(residuals, line='s', ax=axes[1])
axes[1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()

# ==========================================
# 8. Final Regression Equation
# ==========================================

intercept = mlr_model.params['const']
b_tv = mlr_model.params['TV']
b_radio = mlr_model.params['Radio']
b_social = mlr_model.params['Social_Media']

print("\n========== FINAL EQUATION ==========")
equation = (
    f"Sales = {intercept:.4f} "
    f"+ ({b_tv:.4f} × TV) "
    f"+ ({b_radio:.4f} × Radio) "
    f"+ ({b_social:.4f} × Social_Media)"
)
print(equation)

# ==========================================
# 9. Coefficient & Partial Effect Interpretation
# ==========================================

print("\n========== BUSINESS INTERPRETATION ==========")
for col in features:
    coef = mlr_model.params[col]
    p_val = mlr_model.pvalues[col]
    sig_text = "is statistically significant" if p_val < 0.05 else "is NOT statistically significant"
    
    print(
        f"Holding all other variables constant, a 1-unit increase in {col} investment "
        f"is associated with a {coef:.4f} unit change in Sales ({sig_text}, p-value = {p_val:.4f})."
    )

# ==========================================
# 10. Business Recommendation
# ==========================================

print("\n========== FINAL RECOMMENDATION ==========")
print(
"""
1. Allocate marketing budget to channels prioritizing those with the highest, 
   statistically significant positive partial coefficients (p < 0.05).
2. Look at the final MLR summary: If a channel's p-value is greater than 0.05, 
   consider reducing budget there as it does not have a reliable relationship with Sales.
3. Use Adjusted R² instead of regular R² because it penalizes adding variables 
   that do not improve model quality, giving us a true picture of model performance.
"""
)