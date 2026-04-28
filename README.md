# Enhancing Android Malware Detection with Explainable ML
### SMOTE-Tomek · RF–XGBoost Ensemble · SHAP Interpretability

A research project that improves upon the baseline study by Palma et al. (2024) using the **CICAndMal2017** dataset. We address class imbalance, boost malware-class recall, and replace static feature selection with genuine post-hoc explainability.

---

## Overview

| Component | Approach |
|---|---|
| Dataset | CICAndMal2017 (5,491 records, 80 features) |
| Resampling | SMOTE-Tomek (training set only) |
| Base Learners | Random Forest + XGBoost |
| Ensemble | Soft Voting |
| Explainability | SHAP (TreeExplainer) |
| Baseline Accuracy | 79.14% (RF, no resampling) |

---

## Setup

````bash
pip install scikit-learn xgboost imbalanced-learn shap pandas matplotlib seaborn
````

Download the dataset from the [CIC website](https://www.unb.ca/cic/datasets/andmal2017.html) and place the CSV in the project root.

---

## Usage

````bash
# Run full pipeline (baseline replication → SMOTE-Tomek → ensemble + SHAP)
python main.py
````

Or open the notebook in Google Colab:

````
notebooks/malware_detection.ipynb
````

---

## Project Structure

````
├── data/                  # CICAndMal2017 CSV
├── notebooks/             # Colab experiments
├── src/
│   ├── preprocess.py      # Cleaning, normalization, train-test split
│   ├── resampling.py      # SMOTE-Tomek
│   ├── models.py          # RF, XGBoost, ensemble
│   └── explainability.py  # SHAP plots
├── results/               # Confusion matrices, SHAP figures
└── main.py
````

---

## Results

| Model | Accuracy | Recall (Malware) | F1 |
|---|---|---|---|
| RF — baseline \[3\] | 79.14% | — | — |
| RF + SMOTE-Tomek | TBD | TBD | TBD |
| XGBoost + SMOTE-Tomek | TBD | TBD | TBD |
| **RF–XGBoost Ensemble** | **TBD** | **TBD** | **TBD** |

---

## Reference

Palma, C., Ferreira, A., & Figueiredo, M. (2024). *Explainable Machine Learning for Malware Detection on Android Applications.* Information, 15(1), 25. https://doi.org/10.3390/info15010025
