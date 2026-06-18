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