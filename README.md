# Churn Scoring Pipeline
### AI for Fintech | [RISK-001]

A production-grade pipeline for predicting customer churn in banking, with automated feature engineering, Bayesian hyperparameter optimization, and SHAP-based explainability.

Built to work with any tabular churn dataset. Swap the CSV, adjust the target column, and the pipeline handles the rest from raw data to a scored, explainable model.

![Architecture Churn Scoring Pipeline](Architecture%20Churn%20Scoring%20Pipeline.png)

---

## What This Solves

Banks lose customers every day without knowing who is at risk or why. This pipeline scores every customer by churn probability, ranks them by risk, and explains the exact factors driving each prediction so retention teams know who to prioritize and what to address.

---

## Project Structure

```
churn-scoring-pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── xgb_params.py
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── target_analysis.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── training.py
│   ├── evaluation.py
│   └── explainability.py
├── data/
│   ├── bank_customer_churn.csv
│   ├── churn_clean.parquet
│   ├── churn_preprocessed.parquet
│   ├── churn_features.parquet
│   └── churn_selected.parquet
├── models/
│   ├── xgb_churn.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   ├── polynomial.pkl
│   └── selected_features.pkl
├── outputs/
│   ├── ks_curve.png
│   ├── lift_curve.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── shap_waterfall.png
└── notebooks/
    ├── quickstart.ipynb
    └── test_predict.ipynb
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/junidepieri-design/risk-001-customer-churn-scoring.git
cd risk-001-customer-churn-scoring
```

### 2. Create a virtual environment

```bash
# Windows
py -3.13 -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your dataset

Place your churn CSV in the `data/` folder. The pipeline expects a target column indicating churn (1) or not (0), and any identifier columns you want excluded.

If you are re-running the pipeline with a new dataset, delete any existing files inside `data/`, `models/`, and `outputs/` first, keeping only your new CSV in `data/`. This avoids mixing artifacts from a previous run with the new one.

### 5. Run the pipeline

Open `notebooks/quickstart.ipynb` and run all cells in order.

```python
from src.eda import EDA

eda = EDA(
    file_path='../data/your_dataset.csv',
    target_column='your_target_column',
    columns_to_exclude=['customer_id']
)
eda.run()
```

Each subsequent cell runs the next pipeline step, reading the output of the previous one.

### 6. Test predictions on a new customer

Open `notebooks/test_predict.ipynb` to see how to score a single new customer using the saved model and preprocessing artifacts.

---

## How It Works

```
Raw CSV
     │
     ▼ 01 EDA
Summary report + churn_clean.parquet
     │
     ▼ 02 Target Analysis
Distribution and balance check
     │
     ▼ 03 Preprocessing
Label encoding + standard scaling
     │
     ▼ 04 Feature Engineering
PolynomialFeatures — automatic interactions and powers
     │
     ▼ 05 Feature Selection
RandomForest importance — top N features selected
     │
     ▼ 06 Training
XGBoost + Optuna Bayesian optimization
     │
     ▼ 07 Evaluation
AUC-ROC, Gini, KS statistic, LIFT curve by decile
     │
     ▼ 08 Explainability
SHAP values — global importance and per-customer drivers
```

Every step saves its output to `data/`, `models/`, or `outputs/`, so the pipeline can be run in full or resumed from any step.

---

## Key Design Decisions

**Automatic feature engineering**: Instead of hand-crafting features for one specific dataset, the pipeline uses PolynomialFeatures to generate candidate interactions automatically, then lets RandomForest importance decide which ones matter. This is what makes the pipeline reusable across different churn datasets without code changes.

**Bayesian optimization over grid search**: Optuna finds strong hyperparameters in far fewer trials than exhaustive search. The search space lives in `config/xgb_params.py`, so tuning depth is a configuration change, not a code change.

**KS and LIFT over accuracy**: Churn datasets are inherently imbalanced. Accuracy is misleading when 80% of customers do not churn. KS measures how well the model separates churners from non-churners. LIFT tells the business exactly how much better than random the model performs at each decile — the metric retention teams actually use to prioritize outreach.

**Saved artifacts, not just the model**: The label encoders, scaler, polynomial transformer, and selected feature list are all saved during training. This is what allows `test_predict.ipynb` to score new customers using the exact same transformations applied during training, without retraining or refitting anything.

---

## Configuration Reference

All hyperparameter tuning is controlled from `config/xgb_params.py`.

| Parameter | Description |
|-----------|-------------|
| OPTUNA_TRIALS | Number of optimization trials. Higher values find better parameters but take longer. |
| max_depth | Maximum depth of each tree. |
| learning_rate | Step size for each boosting round. |
| n_estimators | Number of boosting rounds. |
| subsample | Fraction of samples used per tree. |
| colsample_bytree | Fraction of features used per tree. |
| min_child_weight | Minimum sum of instance weight needed in a child. |
| gamma | Minimum loss reduction required to make a split. |

---

## Metrics Explained

| Metric | What It Means |
|--------|---------------|
| AUC-ROC | Overall discrimination power. 0.87+ is strong for churn. |
| Gini | Derived from AUC. 0.70+ is considered very good in banking. |
| KS | Maximum separation between churner and non-churner distributions. 0.50+ is a strong model. |
| LIFT | How many times more churners are found in a decile compared to random selection. A decile 10 LIFT of 4x means targeting the top 10% of scores finds 4 times more churners than reaching out randomly. |

---

## What v2 Would Look Like

This pipeline covers the full path from raw data to an explainable, validated model. Natural next steps include a scoring service for real-time inference, model monitoring for drift detection, integration with CRM systems for automated retention triggers, and fairness testing across protected groups for regulatory compliance.

---

## Dataset

This project uses the publicly available Bank Customer Churn dataset from Kaggle, licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0.

No proprietary data is required to run the pipeline. Any tabular churn dataset with a binary target column works with minimal configuration changes.

---

## Author

Built by [Odemir Depieri Jr](https://www.linkedin.com/in/odemir-depieri-jr/) — Data and AI specialist focused on production systems for financial institutions.

Part of the AI for Fintech applied research hub.
