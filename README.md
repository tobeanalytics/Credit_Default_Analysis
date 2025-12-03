## Bank Customer Default Prediction

A complete end-to-end machine-learning project covering data cleaning, exploratory analysis, model development, evaluation, explainability, and business insights for financial decision-making.

### 📌 Project Overview

This project predicts whether a customer is likely to default on a loan using structured bank customer data.
It includes:

- Data cleaning & preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Baseline & advanced machine-learning models
- Evaluation & interpretation
- SHAP explainability
- Practical lender-facing recommendations

The goal is to help lenders reduce credit losses while approving more quality customers.

### 🧹 Data Cleaning Summary

**Key cleaning steps performed:**

- Removed/handled missing values
- Transformed categorical features
- Standardized/normalized numerical variables
- Removed outliers where necessary
- Fixed inconsistent entries
- Exported cleaned dataset as:
"data/processed/bank_cleaned.csv"

**📊 Exploratory Data Analysis (EDA)**

**This analysis covered:**

- Customer demographic patterns
- Income, age, and credit behavior
- Default rate distribution
- Correlations between features & target
- Feature interactions visualized through histograms, KDE plots, boxplots, heatmaps, etc.


### 🧪 Models & Baseline Summary

I evaluated multiple models. Baseline results included:

**Baseline Model Evaluation Notes** 
- The baseline model provides a “starting point” performance.
- It helps measure whether more advanced models are actually improving predictions.
- Metrics such as accuracy, precision, recall, F1-score, and ROC-AUC were recorded.
- These scores were saved in:
"models/baseline_results.json"

### Final Model Summary 
- The final model significantly improved ability to detect likely defaulters.
- Higher recall means fewer risky customers slip through.
- Higher precision means fewer good customers are wrongly flagged.
- ROC-AUC shows the model is far better than random guessing.
- Final results saved in:
"models/model_summary.json"


### 🧠 SHAP Explainability

I generated:

- SHAP summary plot
- Feature importance ranking
- Force plots for individual predictions

Interpretation revealed:

- Top drivers of default
- Why certain customers are predicted risky or safe

### 🏦 Business-Level Insights

My findings help lenders:

- Improve loan approval decisions
- Reduce default losses
- Identify key risk drivers (income, credit history, age, repayment pattern, etc.)
- Build segmented lending strategies
- Estimate revenue & loss impact if recommendations are applied


### 🚀 How to Use This Project

1. Install requirements

pip install -r requirements.txt

2. Run the notebook

Open the full workflow:

notebooks/bank_default_analysis.ipynb

3. Train the model (optional script version)

python src/train_model.py

4. Load the trained model

import joblib
model = joblib.load("models/final_model.pkl")


### 📈 Future Enhancements

- Deploy model as interactive dashboard
- Add API endpoint for real-time scoring
- Implement automated monitoring for model drift
- Build credit risk simulation tools


**👤 Author**

- Tobechukwu Edwin
- Data Analyst & Data Scientist
- LinkedIn: [coming soon]
