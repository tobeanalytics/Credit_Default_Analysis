### Baseline Model Results (baseline_model_results.csv)

This report shows the performance of my very first, simple model before any tuning or advanced modeling.

- What the baseline tells us:

The baseline model gives us a starting point to compare future models against.

Its accuracy and recall are moderate, meaning:

It is correct sometimes but not very strong.

It misses many real defaulters.

It is mainly used to check if more advanced models actually improved things.

#### Takeaway:
This is my “reference model.” It’s not great, but it tells us what a simple model can do. Everything else should perform better than this.


### Model Performance Summary (model_performance_summary.csv)

This file compares my main working models side-by-side:

Logistic Regression

Random Forest

It shows metrics like:

Accuracy

Precision

Recall

F1-score

ROC-AUC


- What the summary shows:

Logistic Regression

Performs okay but not very strong.

Misses more defaulters than Random Forest.

Works better as an explanatory model, not the best performer.

#### Takeaway:
Good for understanding general patterns, but not the best at catching risky customers.


### Random Forest

Outperforms Logistic Regression in almost every metric.

Finds more actual defaulters (high recall).

Makes fewer wrong high-risk predictions (better precision).

Highest overall accuracy and ROC score.


- Takeaway:
The Random Forest model is the most reliable and accurate for predicting default. It should be the preferred model.


### Overall Interpretation of the Report Files

The baseline model was weak, as expected.

Logistic Regression improved performance, but still had limitations.

Random Forest clearly performed the best, showing:

Better ability to separate risky vs low-risk customers

More accurate predictions

Stronger recall (very important for risk detection)


### Final Summary:
This reports show that Random Forest is the most effective model and should be used for predicting credit card default. The baseline model and Logistic Regression help us confirm that Random Forest is truly better.