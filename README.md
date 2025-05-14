Project Title: Flu Shot Learning

Aim: Predict H1N1 and Seasonal Flu Vaccine Uptake from Survey Data

Description

This project builds and evaluates models to predict whether individuals received the H1N1 and/or seasonal flu vaccines, based on demographic, behavioral, and opinion survey features. It follows the CRISP‑DM process—Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment—to deliver robust probability estimates for both vaccine targets 


Table of Contents

1. Installation

2. Usage

3. Data

4. Pipeline Overview (CRISP‑DM)

    4.1 Business Understanding

    4.2 Data Understanding

    4.3 Data Preparation

    4.4 Modeling

    4.5 Evaluation

    4.6 Deployment

5. Results & Interpretation

6. Next Steps

7. License


Installation

>> In git bash run the following commands
# Clone repository
git clone https://github.com/edgarsoudie/flu-shot-learning.git
cd flu-shot-learning

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Dependencies include: pandas, scikit-learn, lightgbm 
DrivenData

Usage

Place data files (`training_set_features.csv`, `training_set_labels.csv`, `test_set_features.csv`, `submission_format.csv`) in the data/ folder.

Run the main pipeline:

bash

python src/run_pipeline.py
Inspect submission.csv for model predictions ready for submission.

Data
Source: National 2009 H1N1 Flu Survey via DrivenData 
DrivenData
.

Training: 26,707 respondents × 35 features + 2 targets.

Test: 26,708 respondents × 35 features.

Features: Demographics (age, sex), behaviors (mask use, handwashing), opinions (vaccine concern/knowledge), and doctor recommendations.

Pipeline Overview (CRISP‑DM)
4.1 Business Understanding
Goal: Maximize predictive accuracy for H1N1 and seasonal flu vaccination uptake to practice end‑to‑end modeling skills.

4.2 Data Understanding
We merged features and labels, examined target balance (~20% H1N1, ~50% seasonal), and checked missingness (up to 10% in employment fields) 
DrivenData
.

4.3 Data Preparation
Missing Values: Imputed categorical missings as "missing"—retains information rather than dropping rows.

Encoding: Used LabelEncoder on combined train+test values for simplicity and compatibility with LightGBM 
Wikipedia
.

4.4 Modeling
Algorithm: LightGBM (binary objective, AUC metric).

Validation: 5‑fold StratifiedKFold to preserve class ratios across folds.

Ensembling: Averaged out‑of‑fold (OOF) predictions from each fold to reduce variance and boost generalization.

python
Copy
Edit
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr, va) in enumerate(kf.split(X, y_h1n1)):
    dtrain = lgb.Dataset(X.iloc[tr], label=y_h1n1.iloc[tr])
    dval   = lgb.Dataset(X.iloc[va], label=y_h1n1.iloc[va])
    model  = lgb.train(params, dtrain, valid_sets=[dval],
                       early_stopping_rounds=50, verbose_eval=False)
    # store OOF and models…
Stratified folds ensure robust AUC evaluation on imbalanced data 
Wikipedia
.

4.5 Evaluation
Metric: ROC AUC (macro-average across both targets).

OOF AUC: ≈0.88 for H1N1, ≈0.92 for seasonal—strong discrimination between classes.

python
Copy
Edit
from sklearn.metrics import roc_auc_score
print("H1N1 OOF AUC:", roc_auc_score(y_h1n1, oof_h1n1))
print("Seasonal OOF AUC:", roc_auc_score(y_seasonal, oof_seasonal))
4.6 Deployment
Ensemble‑average predictions across folds, then write to submission.csv.

Results & Interpretation
High AUCs demonstrate the model’s ability to rank vaccine takers above non‑takers with >88% accuracy.

Top Features: Doctor recommendations, age group, and vaccine concern consistently drove predictions—aligning with domain expectations.

Next Steps
Hyperparameter Tuning via Optuna for extra gains 
datascience-pm.com
.

Feature Engineering: Composite behavioral scores and interaction terms.

Model Blending: Include CatBoost or classifier chains to exploit label correlation.

Threshold Optimization: Convert probabilities to class labels where needed.

License
This project is licensed under the MIT License. See LICENSE for details.

