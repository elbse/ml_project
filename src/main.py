import pandas as pd
from preprocess import load_and_prepare_data
from models import run_baseline, run_smote_models, run_ensemble
from evaluate import evaluate_model
from explain import run_shap

# Load data
X_train, X_test, y_train, y_test = load_and_prepare_data("data/AndMal2017.csv")

# Phase 1
baseline_models = run_baseline(X_train, y_train)

# Phase 2
smote_models, X_res, y_res = run_smote_models(X_train, y_train)

# Phase 3
ensemble_model = run_ensemble(X_res, y_res)

# Evaluation
evaluate_model(ensemble_model, X_test, y_test)

# Explainability
run_shap(ensemble_model, X_test)