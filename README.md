# Task 1: Credit Scoring Model 💳

## Objective
Predict an individual's **creditworthiness** using past financial data.

## Approach
Classification algorithms: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.

## Features Used
| Feature | Description |
|---|---|
| `age` | Applicant age |
| `income` | Annual income |
| `loan_amount` | Requested loan amount |
| `loan_duration_months` | Loan tenure |
| `num_credit_cards` | Number of active credit cards |
| `existing_debts` | Current outstanding debts |
| `payment_history` | Historical payment behavior |
| `employment_years` | Years in current employment |
| `savings_balance` | Current savings |
| `num_dependents` | Number of dependents |
| `debt_to_income` | *(engineered)* Debt / Income ratio |
| `loan_to_income` | *(engineered)* Loan / Income ratio |
| `savings_to_income` | *(engineered)* Savings / Income ratio |

## Evaluation Metrics
- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **ROC-AUC** (primary metric for imbalanced credit data)
- **Confusion Matrix**

## How to Run
```bash
pip install -r requirements.txt
python credit_scoring.py
```

## Real Datasets
- [UCI Credit Card Default](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- [Kaggle Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)

## Output
- Console: classification report for each model
- `credit_scoring_results.png`: ROC curves, model comparison, confusion matrix, feature importance
