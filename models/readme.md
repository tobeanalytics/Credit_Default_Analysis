# Models Folder — Credit Default Analysis

This folder contains serialized model artifacts used in the *Credit Default Analysis* project.

## Files

- scaler.pkl  
  - (Optional) StandardScaler instance used to scale features before feeding them to the Logistic Regression model.
  - Load with joblib.load("models/scaler.pkl").

- logistic_regression_model.pkl  
  - Trained LogisticRegression model used as a baseline/classification model.

- random_forest_model.pkl  
  - Trained RandomForestClassifier model — the preferred production model due to stronger performance.

## Purpose

These files allow anyone to load the trained models and reproduce predictions without re-training:
- Run batch predictions on a cleaned dataset.
- Power a simple API or a demo web app.
- Validate model outputs during review.

## How to load (example snippet)
```python
import joblib

scaler = joblib.load("models/scaler.pkl")                    
lr = joblib.load("models/logistic_regression_model.pkl")
rf = joblib.load("models/random_forest_model.pkl")