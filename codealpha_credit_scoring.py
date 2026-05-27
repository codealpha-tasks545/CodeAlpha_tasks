"""
CodeAlpha Internship - Task 1: Credit Scoring Model
Objective: Predict an individual's creditworthiness using past financial data.
Author: [Your Name]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. GENERATE / LOAD DATASET
# ──────────────────────────────────────────────

def generate_credit_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic credit dataset.
    Replace this function with pd.read_csv('your_data.csv') for real data.
    Recommended real datasets:
      - UCI Credit Card Default: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
      - Kaggle Give Me Some Credit: https://www.kaggle.com/c/GiveMeSomeCredit
    """
    np.random.seed(random_state)
    n = n_samples

    age              = np.random.randint(18, 75, n)
    income           = np.random.normal(55000, 25000, n).clip(10000, 200000)
    loan_amount      = np.random.normal(15000, 10000, n).clip(1000, 80000)
    loan_duration    = np.random.choice([12, 24, 36, 48, 60], n)
    num_credit_cards = np.random.randint(0, 8, n)
    existing_debts   = np.random.normal(8000, 5000, n).clip(0, 50000)
    payment_history  = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n,
                                         p=[0.3, 0.4, 0.2, 0.1])
    employment_years = np.random.randint(0, 35, n)
    savings_balance  = np.random.normal(10000, 8000, n).clip(0, 100000)
    num_dependents   = np.random.randint(0, 5, n)

    # Encode payment history for target generation
    ph_map = {'Excellent': 0, 'Good': 1, 'Fair': 2, 'Poor': 3}
    ph_num = np.array([ph_map[p] for p in payment_history])

    # Creditworthy = 1 (good), 0 (bad) — based on weighted features
    score = (
        0.3 * (income / income.max()) +
        0.2 * (1 - existing_debts / existing_debts.max()) +
        0.2 * (savings_balance / savings_balance.max()) +
        0.15 * (employment_years / 35) +
        0.15 * (1 - ph_num / 3)
    )
    noise = np.random.normal(0, 0.08, n)
    creditworthy = (score + noise > 0.45).astype(int)

    df = pd.DataFrame({
        'age': age,
        'income': income.round(2),
        'loan_amount': loan_amount.round(2),
        'loan_duration_months': loan_duration,
        'num_credit_cards': num_credit_cards,
        'existing_debts': existing_debts.round(2),
        'payment_history': payment_history,
        'employment_years': employment_years,
        'savings_balance': savings_balance.round(2),
        'num_dependents': num_dependents,
        'creditworthy': creditworthy
    })
    return df


# ──────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Encode categorical
    le = LabelEncoder()
    df['payment_history_encoded'] = le.fit_transform(df['payment_history'])
    df.drop(columns=['payment_history'], inplace=True)

    # Feature engineering
    df['debt_to_income']    = df['existing_debts'] / (df['income'] + 1)
    df['loan_to_income']    = df['loan_amount']    / (df['income'] + 1)
    df['savings_to_income'] = df['savings_balance'] / (df['income'] + 1)

    X = df.drop(columns=['creditworthy'])
    y = df['creditworthy']
    return X, y


# ──────────────────────────────────────────────
# 3. TRAIN & EVALUATE MODELS
# ──────────────────────────────────────────────

def evaluate_model(name, model, X_test, y_test, results):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'f1_score':  f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_prob),
        'y_pred':    y_pred,
        'y_prob':    y_prob,
    }
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Not Creditworthy', 'Creditworthy']))
    print(f"  ROC-AUC : {results[name]['roc_auc']:.4f}")
    return results


def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Decision Tree': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    DecisionTreeClassifier(max_depth=6, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    GradientBoostingClassifier(n_estimators=200, random_state=42))
        ]),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        results = evaluate_model(name, model, X_test, y_test, results)

    return trained, results


# ──────────────────────────────────────────────
# 4. VISUALISATIONS
# ──────────────────────────────────────────────

def plot_results(results, X_test, y_test, feature_names, rf_model):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Credit Scoring Model — Evaluation Dashboard', fontsize=16, fontweight='bold')

    # --- ROC Curves ---
    ax = axes[0, 0]
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curves')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Model Comparison Bar Chart ---
    ax = axes[0, 1]
    names = list(results.keys())
    metrics = ['accuracy', 'f1_score', 'roc_auc']
    x = np.arange(len(names))
    width = 0.25
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[n][metric] for n in names]
        ax.bar(x + i * width, vals, width, label=metric.replace('_', ' ').title(), color=color, alpha=0.85)
    ax.set(xticks=x + width, xticklabels=[n.replace(' ', '\n') for n in names],
           ylim=(0.5, 1.05), title='Model Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # --- Confusion Matrix (Best Model) ---
    ax = axes[1, 0]
    best_name = max(results, key=lambda n: results[n]['roc_auc'])
    cm = confusion_matrix(y_test, results[best_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Creditworthy', 'Creditworthy'],
                yticklabels=['Not Creditworthy', 'Creditworthy'])
    ax.set(title=f'Confusion Matrix — {best_name}', xlabel='Predicted', ylabel='Actual')

    # --- Feature Importance (Random Forest) ---
    ax = axes[1, 1]
    rf_clf = rf_model.named_steps['clf']
    importances = pd.Series(rf_clf.feature_importances_, index=feature_names).sort_values(ascending=True)
    importances.tail(10).plot(kind='barh', ax=ax, color='#9C27B0', alpha=0.85)
    ax.set(title='Top 10 Feature Importances (Random Forest)', xlabel='Importance')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('credit_scoring_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[✓] Plot saved → credit_scoring_results.png")


# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  CodeAlpha — Task 1: Credit Scoring Model")
    print("=" * 60)

    # Load data
    df = generate_credit_dataset(n_samples=2000)
    print(f"\n[✓] Dataset shape : {df.shape}")
    print(f"[✓] Class balance :\n{df['creditworthy'].value_counts(normalize=True).round(3)}")

    # Preprocess
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n[✓] Train samples : {len(X_train)}")
    print(f"[✓] Test  samples : {len(X_test)}")

    # Train
    trained_models, results = train_models(X_train, X_test, y_train, y_test)

    # Best model summary
    best_name = max(results, key=lambda n: results[n]['roc_auc'])
    print(f"\n{'='*60}")
    print(f"  🏆 Best Model: {best_name}")
    print(f"     ROC-AUC  : {results[best_name]['roc_auc']:.4f}")
    print(f"     Accuracy : {results[best_name]['accuracy']:.4f}")
    print(f"     F1-Score : {results[best_name]['f1_score']:.4f}")
    print(f"{'='*60}")

    # Plot
    plot_results(results, X_test, y_test,
                 feature_names=X.columns.tolist(),
                 rf_model=trained_models['Random Forest'])
