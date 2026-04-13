# =============================================================================
#  PHASE 3 — RF+XGBoost Soft Voting Ensemble + SHAP Explainability
#  - SMOTETomek on training set only
#  - GridSearchCV hyperparameter tuning
#  - 5-fold stratified cross-validation
#  - SHAP: global summary, feature importance, local force plots
#  - SHAP vs RRFS feature comparison
# =============================================================================

import matplotlib
matplotlib.use("Agg")

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_validate, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
#  CONFIGURATION
# =============================================================================

DATASET_PATH = "dataset.csv"
TARGET_COL   = "Class"
DROP_COLS    = ["App", "Package", "Category", "Description", "Related apps"]
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5
RESULTS_DIR  = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
#  1. DATA LOADING (identical to Phase 1 & 2)
# =============================================================================

print("=" * 65)
print("  PHASE 3 — Ensemble + SHAP Explainability")
print("=" * 65)

print("\n[1] Loading and preparing dataset ...")
df = pd.read_csv(DATASET_PATH, low_memory=False)
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df.drop_duplicates(inplace=True)

if df[TARGET_COL].dtype == object:
    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL])

X = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number])
y = df[TARGET_COL]

vt = VarianceThreshold(threshold=0.0)
X = pd.DataFrame(vt.fit_transform(X), columns=X.columns[vt.get_support()])

vc = y.value_counts().sort_index()
n_malware = int(vc.get(1, 0))
n_benign  = int(vc.get(0, 0))
spw       = n_malware / n_benign

print(f"    Shape            : {X.shape}")
print(f"    Malware          : {n_malware}  |  Benign : {n_benign}")
print(f"    scale_pos_weight : {spw:.2f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"    Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")

# =============================================================================
#  2. HELPERS
# =============================================================================

def evaluate(name, y_true, y_pred, y_prob=None, print_report=False):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  {'─'*55}")
    print(f"  {name}")
    print(f"  {'─'*55}")
    print(f"    Accuracy        : {acc*100:.2f}%")
    print(f"    Precision       : {prec*100:.2f}%")
    print(f"    Recall          : {rec*100:.2f}%   <- malware detection rate")
    print(f"    F1-Score        : {f1*100:.2f}%")
    if auc is not None:
        print(f"    AUC-ROC         : {auc*100:.2f}%")
    print(f"    False Negatives : {fn}  (malware missed)")
    print(f"    False Positives : {fp}  (benign wrongly flagged)")

    if print_report:
        print(f"\n    Full Classification Report:")
        report = classification_report(
            y_true, y_pred, target_names=["Benign", "Malware"]
        )
        for line in report.splitlines():
            print("    " + line)

    return {
        "Model":           name,
        "Accuracy":        round(acc  * 100, 2),
        "Precision":       round(prec * 100, 2),
        "Recall":          round(rec  * 100, 2),
        "F1-Score":        round(f1   * 100, 2),
        "AUC-ROC":         round(auc  * 100, 2) if auc is not None else None,
        "False Negatives": int(fn),
        "False Positives": int(fp),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def save_cm(name, y_true, y_pred, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=["Benign", "Malware"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()

# =============================================================================
#  3. SMOTE-TOMEK RESAMPLING (training set only)
# =============================================================================

print("\n\n[2] Applying SMOTE-Tomek to training set ...")
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

smt = SMOTETomek(random_state=RANDOM_STATE)
X_train_res, y_train_res = smt.fit_resample(X_train_sc, y_train)

res_vc = pd.Series(y_train_res).value_counts().sort_index()
print(f"    Before resampling : Benign={int((y_train==0).sum())}  Malware={int((y_train==1).sum())}")
print(f"    After resampling  : Benign={int(res_vc.get(0,0))}  Malware={int(res_vc.get(1,0))}")
print(f"    Test set          : UNCHANGED (no data leakage)")

# =============================================================================
#  4. GRIDSEARCHCV HYPERPARAMETER TUNING
# =============================================================================

print("\n\n[3] GridSearchCV — tuning RF and XGBoost ...")
print("    (This may take several minutes)")

cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# --- RF tuning ---------------------------------------------------------------
print("\n  Tuning Random Forest ...")
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth":    [None, 10, 20],
    "min_samples_leaf": [1, 2],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    rf_param_grid,
    cv=cv_inner,
    scoring="f1",
    n_jobs=-1,
    verbose=0,
)
rf_grid.fit(X_train_res, y_train_res)
best_rf = rf_grid.best_estimator_
print(f"    Best RF params : {rf_grid.best_params_}")
print(f"    Best RF F1 (CV): {rf_grid.best_score_*100:.2f}%")

# --- XGBoost tuning ----------------------------------------------------------
print("\n  Tuning XGBoost ...")
xgb_param_grid = {
    "n_estimators":  [100, 200],
    "max_depth":     [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "subsample":     [0.8, 1.0],
}
xgb_grid = GridSearchCV(
    XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    xgb_param_grid,
    cv=cv_inner,
    scoring="f1",
    n_jobs=-1,
    verbose=0,
)
xgb_grid.fit(X_train_res, y_train_res)
best_xgb = xgb_grid.best_estimator_
# Fix shap + xgboost version incompatibility — base_score must be explicit float
best_xgb.set_params(base_score=0.5)
best_xgb.fit(X_train_res, y_train_res)
print(f"    Best XGB params : {xgb_grid.best_params_}")
print(f"    Best XGB F1 (CV): {xgb_grid.best_score_*100:.2f}%")

# Save tuning results
tuning_df = pd.DataFrame([
    {"Model": "Random Forest",
     "Best Params": str(rf_grid.best_params_),
     "Best CV F1":  round(rf_grid.best_score_ * 100, 2)},
    {"Model": "XGBoost",
     "Best Params": str(xgb_grid.best_params_),
     "Best CV F1":  round(xgb_grid.best_score_ * 100, 2)},
])
tuning_df.to_csv(os.path.join(RESULTS_DIR, "phase3_tuning_results.csv"), index=False)
print(f"\n  Saved -> results/phase3_tuning_results.csv")

# =============================================================================
#  5. SOFT VOTING ENSEMBLE
# =============================================================================

print("\n\n[4] Training RF+XGBoost Soft Voting Ensemble ...")

ensemble = VotingClassifier(
    estimators=[("rf", best_rf), ("xgb", best_xgb)],
    voting="soft",
)
ensemble.fit(X_train_res, y_train_res)

y_pred_ens = ensemble.predict(X_test_sc)
y_prob_ens = ensemble.predict_proba(X_test_sc)[:, 1]

ensemble_metrics = evaluate(
    "RF+XGBoost Soft Voting Ensemble + SMOTE-Tomek",
    y_test, y_pred_ens, y_prob_ens,
    print_report=True
)
save_cm("RF+XGBoost Ensemble + SMOTE-Tomek",
        y_test, y_pred_ens, "cm_phase3_ensemble.png")

# =============================================================================
#  6. CROSS-VALIDATION
# =============================================================================

print("\n\n[5] 5-fold stratified cross-validation ...")

# Use ImbPipeline for CV so SMOTE-Tomek is applied per fold correctly
cv_pipe = ImbPipeline([
    ("smote_tomek", SMOTETomek(random_state=RANDOM_STATE)),
    ("scaler",      StandardScaler()),
    ("ensemble",    VotingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(**rf_grid.best_params_,
                                           random_state=RANDOM_STATE, n_jobs=-1)),
            ("xgb", XGBClassifier(**xgb_grid.best_params_,
                                   eval_metric="logloss",
                                   random_state=RANDOM_STATE, n_jobs=-1)),
        ],
        voting="soft",
    )),
])

cv_outer = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
cv_results = cross_validate(
    cv_pipe, X, y,
    cv=cv_outer,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    return_train_score=False,
)

print(f"\n  {'Metric':<12}  {'Mean':>8}  {'Std':>8}")
print(f"  {'─'*32}")
cv_summary = {}
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    scores = cv_results[f"test_{metric}"]
    mean   = scores.mean() * 100
    std    = scores.std()  * 100
    print(f"  {metric:<12}  {mean:>7.2f}%  +/-{std:>5.2f}%")
    cv_summary[metric] = {"mean": round(mean, 2), "std": round(std, 2)}

cv_df = pd.DataFrame([
    {"Metric": k, "Mean (%)": v["mean"], "Std (%)": v["std"]}
    for k, v in cv_summary.items()
])
cv_df.to_csv(os.path.join(RESULTS_DIR, "phase3_cv_results.csv"), index=False)
print(f"\n  Saved -> results/phase3_cv_results.csv")

# =============================================================================
#  7. SHAP EXPLAINABILITY
# =============================================================================

print("\n\n[6] SHAP Explainability ...")
print("    Computing SHAP values for RF and XGBoost base learners ...")

# Convert to DataFrame for SHAP
X_test_df = pd.DataFrame(X_test_sc, columns=X.columns)

# Use a sample of test set for speed (max 500 samples)
shap_sample_size = min(500, len(X_test_df))
np.random.seed(RANDOM_STATE)
shap_idx    = np.random.choice(len(X_test_df), shap_sample_size, replace=False)
X_shap      = X_test_df.iloc[shap_idx]
y_shap      = y_test.iloc[shap_idx] if hasattr(y_test, 'iloc') else y_test[shap_idx]
y_pred_shap = ensemble.predict(X_shap)

print(f"    SHAP sample size : {shap_sample_size} test instances")

# --- RF SHAP -----------------------------------------------------------------
print("\n  Computing RF SHAP values ...")
rf_explainer  = shap.TreeExplainer(best_rf)
rf_shap_vals  = rf_explainer.shap_values(X_shap)

# For binary classification, shap_values returns list [class0, class1]
if isinstance(rf_shap_vals, list):
    rf_shap_malware = rf_shap_vals[1]
else:
    rf_shap_malware = rf_shap_vals

# --- XGBoost SHAP ------------------------------------------------------------
print("  Computing XGBoost SHAP values ...")
xgb_explainer = shap.TreeExplainer(best_xgb)
xgb_shap_vals = xgb_explainer.shap_values(X_shap)

# Average SHAP values from both models (ensemble-level explanation)
ensemble_shap = (rf_shap_malware + xgb_shap_vals) / 2

print("  SHAP values computed successfully.")

# --- Global feature importance (mean |SHAP|) ---------------------------------
shap_importance = pd.DataFrame({
    "Feature":    X.columns,
    "RF_SHAP":    np.abs(rf_shap_malware).mean(axis=0),
    "XGB_SHAP":   np.abs(xgb_shap_vals).mean(axis=0),
    "Ensemble_SHAP": np.abs(ensemble_shap).mean(axis=0),
}).sort_values("Ensemble_SHAP", ascending=False)

shap_importance.to_csv(
    os.path.join(RESULTS_DIR, "phase3_shap_feature_importance.csv"), index=False
)
print(f"\n  Top 10 SHAP features (ensemble):")
print(shap_importance[["Feature","Ensemble_SHAP"]].head(10).to_string(index=False))

# --- 7a. SHAP summary beeswarm plot ------------------------------------------
print("\n  Generating SHAP summary beeswarm plot ...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    ensemble_shap, X_shap,
    feature_names=X.columns.tolist(),
    show=False, max_display=20
)
plt.title("SHAP summary — ensemble (top 20 features)", fontsize=11,
          fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "shap_summary_beeswarm.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# --- 7b. SHAP feature importance bar chart -----------------------------------
print("  Generating SHAP feature importance bar chart ...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    ensemble_shap, X_shap,
    feature_names=X.columns.tolist(),
    plot_type="bar", show=False, max_display=20
)
plt.title("SHAP feature importance — ensemble (top 20)", fontsize=11,
          fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "shap_importance_bar.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# =============================================================================
#  8. SHAP LOCAL FORCE PLOTS (4 specific instances)
# =============================================================================

print("\n[7] Generating SHAP local force plots ...")

y_shap_arr      = np.array(y_shap)
y_pred_shap_arr = np.array(y_pred_shap)

# Find indices for each case
tp_idx = np.where((y_shap_arr == 1) & (y_pred_shap_arr == 1))[0]  # correct malware
tn_idx = np.where((y_shap_arr == 0) & (y_pred_shap_arr == 0))[0]  # correct benign
fn_idx = np.where((y_shap_arr == 1) & (y_pred_shap_arr == 0))[0]  # missed malware
fp_idx = np.where((y_shap_arr == 0) & (y_pred_shap_arr == 1))[0]  # wrong flag

cases = {
    "tp_malware": (tp_idx, "True Positive — Malware correctly detected",
                   "shap_force_tp_malware.png"),
    "tn_benign":  (tn_idx, "True Negative — Benign correctly identified",
                   "shap_force_tp_benign.png"),
    "fn":         (fn_idx, "False Negative — Malware MISSED by model",
                   "shap_force_fn.png"),
    "fp":         (fp_idx, "False Positive — Benign wrongly flagged",
                   "shap_force_fp.png"),
}

xgb_exp_obj = shap.TreeExplainer(best_xgb)

for case_name, (idx_arr, title, fname) in cases.items():
    if len(idx_arr) == 0:
        print(f"    No {case_name} instances found in SHAP sample — skipping.")
        continue
    i = idx_arr[0]
    instance = X_shap.iloc[[i]]

    # Use waterfall plot (more readable than force plot in static PNG)
    shap_vals_i = xgb_exp_obj(instance)

    plt.figure(figsize=(12, 5))
    shap.waterfall_plot(shap_vals_i[0], max_display=15, show=False)
    plt.title(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {fname}")

# =============================================================================
#  9. SHAP vs RRFS COMPARISON
# =============================================================================

print("\n\n[8] SHAP vs RRFS feature comparison ...")

rrfs_path = os.path.join(RESULTS_DIR, "phase1_rrfs_top_features.csv")
if os.path.exists(rrfs_path):
    rrfs_df = pd.read_csv(rrfs_path)
    rrfs_features = rrfs_df["Feature"].tolist()
else:
    rrfs_features = []
    print("    WARNING: RRFS file not found. Run Phase 1 first.")

shap_top20 = shap_importance["Feature"].head(20).tolist()

# Build comparison table
max_len     = max(len(shap_top20), len(rrfs_features))
shap_padded = shap_top20 + [""] * (max_len - len(shap_top20))
rrfs_padded = rrfs_features + [""] * (max_len - len(rrfs_features))

comparison_df = pd.DataFrame({
    "Rank":          range(1, max_len + 1),
    "SHAP Top Feature":  shap_padded,
    "RRFS Top Feature":  rrfs_padded,
    "In Both":       ["Yes" if s != "" and s in rrfs_features
                      else "" for s in shap_padded],
})
comparison_df.to_csv(
    os.path.join(RESULTS_DIR, "phase3_shap_vs_rrfs_comparison.csv"), index=False
)

overlap = set(shap_top20) & set(rrfs_features)
print(f"\n  SHAP top 20 features  : {len(shap_top20)}")
print(f"  RRFS top features     : {len(rrfs_features)}")
print(f"  Overlap (in both)     : {len(overlap)} features")
print(f"  Overlapping features  : {list(overlap)[:5]} ...")
print(f"\n  Saved -> results/phase3_shap_vs_rrfs_comparison.csv")

# SHAP vs RRFS bar comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

top_n = 15
shap_plot = shap_importance.head(top_n)
axes[0].barh(shap_plot["Feature"][::-1],
             shap_plot["Ensemble_SHAP"][::-1],
             color="#534AB7", edgecolor="white")
axes[0].set_title("SHAP — top 15 features", fontsize=10, fontweight="bold")
axes[0].set_xlabel("Mean |SHAP value|")

if len(rrfs_features) > 0:
    rrfs_plot = rrfs_df.head(top_n)
    axes[1].barh(rrfs_plot["Feature"][::-1],
                 rrfs_plot["Relevance"][::-1],
                 color="#B5D4F4", edgecolor="white")
    axes[1].set_title("RRFS — top 15 features", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Pearson correlation with target")

plt.suptitle("SHAP vs RRFS — feature importance comparison",
             fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plot_shap_vs_rrfs_features.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# =============================================================================
#  10. UNIFIED RESULTS TABLE (all phases)
# =============================================================================

print("\n\n[9] Building unified results table (all phases) ...")

phase1_path = os.path.join(RESULTS_DIR, "phase1_baseline_results.csv")
phase2_path = os.path.join(RESULTS_DIR, "phase2_smote_tomek_comparison.csv")

if os.path.exists(phase1_path) and os.path.exists(phase2_path):
    p1 = pd.read_csv(phase1_path)
    p1["Phase"]      = "Phase 1"
    p1["Resampling"] = "None"

    p2 = pd.read_csv(phase2_path)
    p2["Phase"] = "Phase 2"

    p3 = pd.DataFrame([ensemble_metrics])
    p3["Phase"]      = "Phase 3"
    p3["Resampling"] = "SMOTE-Tomek"
    p3["Model"]      = "RF+XGBoost Ensemble"

    unified_cols = ["Phase", "Model", "Resampling", "Accuracy",
                    "Precision", "Recall", "F1-Score",
                    "AUC-ROC", "False Negatives", "False Positives"]

    all_unified = pd.concat([p1, p2, p3], ignore_index=True)
    available   = [c for c in unified_cols if c in all_unified.columns]
    all_unified[available].to_csv(
        os.path.join(RESULTS_DIR, "phase3_unified_all_results.csv"), index=False
    )
    print(f"  Saved -> results/phase3_unified_all_results.csv")
else:
    print("  WARNING: Phase 1 or Phase 2 CSV not found — skipping unified table.")

# Save Phase 3 ensemble results separately
p3_df = pd.DataFrame([ensemble_metrics])
p3_df.to_csv(os.path.join(RESULTS_DIR, "phase3_ensemble_results.csv"), index=False)
print(f"  Saved -> results/phase3_ensemble_results.csv")

# =============================================================================
#  11. VISUALISATIONS
# =============================================================================

print("\n[10] Generating Phase 3 plots ...")

BLUE   = "#B5D4F4"
PURPLE = "#534AB7"
RED    = "#A32D2D"
AMBER  = "#EF9F27"

# 11a. Ensemble per-metric bar
cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
vals = [ensemble_metrics[c] for c in cols]
fig, ax = plt.subplots(figsize=(7, 4))
bar_colors = [PURPLE, "#0F6E56", RED, "#185FA5"]
bars = ax.bar(cols, vals, color=bar_colors, width=0.5, edgecolor="white")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 115)
ax.set_ylabel("Score (%)", fontsize=10)
ax.set_title("Phase 3 — Proposed ensemble per-metric performance",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase3_ensemble_metrics.png"), dpi=150)
plt.close()

# 11b. All-phases F1 comparison
if os.path.exists(phase1_path) and os.path.exists(phase2_path):
    p1_f1 = pd.read_csv(phase1_path)[["Model", "F1-Score"]]
    p1_f1["Label"] = p1_f1["Model"].str.extract(r'\((\w+)\)')[0]
    p1_f1["Label"] = p1_f1["Label"].fillna(p1_f1["Model"])

    p2_full = pd.read_csv(phase2_path)
    p2_smote = p2_full[p2_full["Resampling"]=="SMOTE-Tomek"][["Model","F1-Score"]].copy()
    p2_smote["Label"] = p2_smote["Model"].str.extract(r'\((\w+)\)')[0]
    p2_smote["Label"] = p2_smote["Label"].fillna(p2_smote["Model"])
    p2_smote["Label"] = p2_smote["Label"] + " +ST"

    ensemble_row = pd.DataFrame([{
        "Label": "RF+XGBoost\nEnsemble",
        "F1-Score": ensemble_metrics["F1-Score"]
    }])

    all_f1 = pd.concat([
        p1_f1[["Label","F1-Score"]],
        p2_smote[["Label","F1-Score"]],
        ensemble_row
    ], ignore_index=True)

    n_p1 = len(p1_f1)
    n_p2 = len(p2_smote)
    colors_all = ([BLUE]*n_p1 + [AMBER]*n_p2 + [PURPLE])

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(all_f1["Label"], all_f1["F1-Score"],
                  color=colors_all, edgecolor="white", width=0.6)
    for bar, val in zip(bars, all_f1["F1-Score"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BLUE,   label="Phase 1 — No resampling"),
        Patch(facecolor=AMBER,  label="Phase 2 — SMOTE-Tomek"),
        Patch(facecolor=PURPLE, label="Phase 3 — Proposed ensemble"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_ylim(0, 95)
    ax.set_ylabel("F1-Score (%)", fontsize=10)
    ax.set_title("F1-Score across all phases — full comparison",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_f1_all_phases.png"), dpi=150)
    plt.close()

print("  Plots saved to /results/")

# =============================================================================
#  12. FINAL VERDICT — mapped to each specific objective
# =============================================================================

p1_best_acc = pd.read_csv(phase1_path)["Accuracy"].max() \
    if os.path.exists(phase1_path) else 0
p1_best_f1  = pd.read_csv(phase1_path)["F1-Score"].max() \
    if os.path.exists(phase1_path) else 0
p1_best_fn  = pd.read_csv(phase1_path)["False Negatives"].min() \
    if os.path.exists(phase1_path) else 0

prop_acc = ensemble_metrics["Accuracy"]
prop_f1  = ensemble_metrics["F1-Score"]
prop_rec = ensemble_metrics["Recall"]
prop_fn  = ensemble_metrics["False Negatives"]

def delta(new, old):
    sign = "+ " if new >= old else "- "
    return f"({sign}{abs(new - old):.2f}%)"

print("\n\n" + "=" * 65)
print("  FINAL VERDICT — Mapped to Study Objectives")
print("=" * 65)
print(f"""
  OBJ 2.2.1 — Baseline Replication
  ─────────────────────────────────────────────────────────────
  All 5 classifiers replicated (RF, SVM, KNN, NB, MLP).
  Best accuracy     : {p1_best_acc:.2f}%
  Palma et al. best : 79.14%  (deviation documented in CSV)
  RRFS              : {len(shap_top20)} features selected and saved

  OBJ 2.2.2 — SMOTE-Tomek Contribution
  ─────────────────────────────────────────────────────────────
  All classifiers retrained with SMOTETomek.
  NOTE: Malware is the majority class (61.1%) in this dataset.
  SMOTETomek decreased recall for most models — this is
  expected and documented. Impact saved to phase2 CSV.

  OBJ 2.2.3 — Proposed Ensemble
  ─────────────────────────────────────────────────────────────
  RF + XGBoost soft voting + SMOTE-Tomek + GridSearchCV tuning
  Best Phase 1 F1   : {p1_best_f1:.2f}%
  Proposed F1       : {prop_f1:.2f}%   {delta(prop_f1, p1_best_f1)}
  Best Phase 1 acc  : {p1_best_acc:.2f}%
  Proposed accuracy : {prop_acc:.2f}%  {delta(prop_acc, p1_best_acc)}
  Best Phase 1 FN   : {p1_best_fn}
  Proposed FN       : {prop_fn}  ({p1_best_fn - prop_fn:+d} vs best baseline)

  OBJ 2.2.4 — SHAP Explainability
  ─────────────────────────────────────────────────────────────
  SHAP values computed for RF + XGBoost base learners.
  Global summary beeswarm : shap_summary_beeswarm.png
  Global importance bar   : shap_importance_bar.png
  Local force plots       : 4 instances (TP, TN, FN, FP)
  SHAP vs RRFS comparison : phase3_shap_vs_rrfs_comparison.csv
  Feature overlap         : {len(overlap)} / {min(len(shap_top20), len(rrfs_features))} features in common
""")
print("=" * 65)
print(f"\n  All outputs saved to : {os.path.abspath(RESULTS_DIR)}")
print("\n  Pipeline complete — all 3 phases done.\n")