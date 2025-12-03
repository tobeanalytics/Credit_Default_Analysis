## Credit Card Default Prediction – Machine Learning Analysis

### 📌 Project Overview

This project analyzes the UCI Credit Card Default Dataset and builds machine-learning models to predict whether a customer will default on their next monthly payment.
The goal is to help financial institutions identify high-risk customers, reduce losses, and improve lending decisions.

### ⭐ Business Summary 

**Why This Analysis Matters**

Banks and lenders lose money when customers fail to repay loans or credit card balances.
This project helps answer:

- Who is more likely to default next month?
- What customer behaviors signal higher risk?
- How can lenders reduce losses using data?

### Key Findings

- A clear difference exists between customers who default and those who don’t.
- Payment history variables (late payments) were the strongest predictors of default.
- Random Forest was the best-performing model, giving the most accurate predictions.
- SHAP analysis showed that the model’s decisions are explainable and transparent.

### What Lenders Can Do

- Flag high-risk customers early
- Offer targeted repayment plans
- Adjust credit limits
- Reduce financial losses through proactive intervention


### 🔍 Technical Summary (For Data Scientists)

Dataset

- 30,000 customers
- 24 features
- Binary target: default payment next month

### Steps Performed
1. Data Cleaning & Preparation
2. Exploratory Data Analysis (EDA)
3. Model Training:
   - Logistic Regression
   - Random Forest
4. Model Evaluation:
   - Confusion Matrix
   - ROC-AUC
   - Precision–Recall Curve
   - KS Curve
   - Lift & Gain Charts
5. Explainability with SHAP
6. Business Insights & Recommendations

**Best Model**

- Random Forest
- Highest AUC
- Strong recall
- Best separation on KS statistic
- Most robust performance overall


### 🚀 How to Run the Project

1. Install dependencies:

pip install -r requirements.txt

2. Open the notebooks:

jupyter notebook notebooks/

3. Run steps in order from 01 → 05.


### 🏁 Conclusion

This project provides a complete, transparent, and business-ready approach to understanding and predicting credit default risk. The models and insights can directly support risk management teams, financial analysts, and lending institutions.

### 📈 Future Enhancements

- Deploy model as interactive dashboard
- Add API endpoint for real-time scoring
- Implement automated monitoring for model drift
- Build credit risk simulation tools

**👤 Author**

- Tobechukwu Edwin
- Data Analyst & Data Scientist
- LinkedIn: [coming soon]
