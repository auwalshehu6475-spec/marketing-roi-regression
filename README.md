# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# Set visualization styles
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ## 1. Load and Clean Dataset
# Def file = '85334965-5736-457a-b8d4-a077e6872f84.csv'

# Load data
df = pd.read_csv(file)

# Display initial metadata
print("--- Initial Dataset Summary ---")
print(df.info())
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# Drop missing values to ensure clean modeling
df_clean = df.dropna().copy()
print(f"\nShape after dropping missing values: {df_clean.shape}")

# ## 2. Exploratory Data Analysis (EDA) & Variable Selection

# Calculate correlation matrix
corr_matrix = df_clean.corr()
print("--- Correlation Matrix ---")
print(corr_matrix)

# Visualize relationships with sales
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.regplot(data=df_clean, x='TV', y='Sales', ax=axes[0], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[0].set_title(f"TV vs Sales (r = {corr_matrix.loc['TV', 'Sales']:.4f})")

sns.regplot(data=df_clean, x='Radio', y='Sales', ax=axes[1], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[1].set_title(f"Radio vs Sales (r = {corr_matrix.loc['Radio', 'Sales']:.4f})")

sns.regplot(data=df_clean, x='Social_Media', y='Sales', ax=axes[2], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[2].set_title(f"Social Media vs Sales (r = {corr_matrix.loc['Social_Media', 'Sales']:.4f})")

plt.tight_layout()
plt.savefig('eda_plots.png')
print("\n[INFO] EDA plots saved successfully as 'eda_plots.png'")

# **Analytical Justification:** Based on the correlation matrix, **TV** has a near-perfect linear relationship with Sales ($r = 0.9995$), outperforming Radio ($0.8686$) and Social Media ($0.5274$). Therefore, TV is selected as our independent variable.

# ## 3. Build the OLS Regression Model

# Define variables
X = df_clean['TV']
y = df_clean['Sales']

# Add constant to independent variable for intercept calculation
X_with_constant = sm.add_constant(X)

# Fit OLS Model
model = sm.OLS(y, X_with_constant).fit()

# Display results
print(model.summary())

# ## 4. Regression Diagnostic Plots

# Extract residuals and fitted values
residuals = model.resid
fitted_values = model.fittedvalues

# Create diagnostics grid
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Linearity & Homoscedasticity Check
sns.scatterplot(x=fitted_values, y=residuals, ax=axes[0], alpha=0.3)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_title('Residuals vs Fitted (Homoscedasticity)')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')

# 2. Normality Check: Q-Q Plot
sm.qqplot(residuals, line='s', ax=axes[1])
axes[1].set_title('Normal Q-Q Plot')

# 3. Normality Check: Residuals Distribution Histogram
sns.histplot(residuals, kde=True, ax=axes[2])
axes[2].set_title('Residuals Distribution')

plt.tight_layout()
plt.savefig('diagnostic_plots.png')
print("[INFO] Diagnostic plots saved successfully as 'diagnostic_plots.png'")

# Simple Linear Regression – Marketing ROI Analysis

## Project Overview
This project performs an end-to-end Simple Linear Regression analysis on an enterprise marketing dataset. Using Python, `pandas`, and `statsmodels`, we explore which marketing channel (TV, Radio, or Social Media) yields the strongest return on investment (ROI) for Sales, fit an Ordinary Least Squares (OLS) model, and validate core regression assumptions.

## Key Findings
- **Top Channel:** **TV Advertising** exhibits an extraordinary correlation of $r = 0.9995$ with sales volume.
- **Model Efficiency ($R^2$):** **99.9%** of the variance in sales can be explained solely by TV marketing allocation.
- **ROI Impact:** For every **\$1.00** increase in TV advertising expenditure, Sales increase predictably by **\$3.56**.
- **Model Validity:** Assumptions of Linearity, Homoscedasticity, and Normality of residuals were fully met via diagnostic visualization metrics.

## Project Structure
- `regression_analysis.ipynb`: The complete executed data workspace.
- `85334965-5736-457a-b8d4-a077e6872f84.csv`: Cleaned source marketing dataset.
- `eda_plots.png`: Distribution and regression scatter plots for marketing channels.
- `diagnostic_plots.png`: Residual, Q-Q, and homoscedasticity verification figures.

## Environment Setup Instructions
To run the notebook locally, configure your Python environment using the instructions below:

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
   cd YOUR_REPOSITORY

jupyter notebook

pip install pandas numpy matplotlib seaborn statsmodels scipy jupyter
