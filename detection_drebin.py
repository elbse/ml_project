# =============================================================================
#  Android Malware Detection — Full Pipeline (Drebin Dataset)
#  Aligned to study objectives:
#    OBJ 1  : Replicate 5 baselines + document deviations from Palma et al.
#    OBJ 2  : XGBoost standalone — without SMOTE and with SMOTE
#    OBJ 3  : RF+XGBoost soft voting ensemble with SMOTE
#             + false negative analysis + full metric comparison
#
#  Dataset  : Drebin
#  Target   : 'class'  →  S = Malware (1), B = Benign (0)
#  Features : 215 binary API/permission features (all numeric after encoding)
#  No metadata columns to drop
# =============================================================================

import matplotlib
matplotlib.use("Agg")   # must be before pyplot — prevents tkinter thread crash

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
#  1. CONFIGURATION
# =============================================================================

DATASET_PATH = "drebin.csv"
TARGET_COL   = "class"          # S = Malware, B = Benign
DROP_COLS    = []               # Drebin has no metadata columns to drop
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5
RESULTS_DIR  = "Updated_results_drebin"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Published results from Palma et al. (2024) on CICAndMal2017
# Used for deviation analysis in Objective 1
PALMA_RESULTS = {
    "Random Forest (RF)":           {"Accuracy": 79.14},
    "Support Vector Machine (SVM)": {"Accuracy": 70.81},
    "k-Nearest Neighbour (KNN)":    {"Accuracy": 62.01},
    "Naive Bayes (NB)":             {"Accuracy": 74.53},
    "Multilayer Perceptron (MLP)":  {"Accuracy": 78.12},
}

# =============================================================================
#  2. DATA LOADING & PREPROCESSING
# =============================================================================

print("=" * 65)
print("  Android Malware Detection — Drebin Dataset")
print("=" * 65)

print("\n[SETUP] Loading and preparing dataset ...")
df = pd.read_csv(DATASET_PATH, low_memory=False)

# Drop any metadata columns if present
cols_to_drop = [c for c in DROP_COLS if c in df.columns]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)

df.dropna(inplace=True)
print(f"  Shape after cleaning : {df.shape}")

# Encode target: S (Malware) → 1, B (Benign) → 0
le = LabelEncoder()
y = pd.Series(le.fit_transform(df[TARGET_COL]), name=TARGET_COL)
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"  Label encoding       : {label_map}")
print(f"  Malware class        : '{[k for k,v in label_map.items() if v==1][0]}' → 1")
print(f"  Benign  class        : '{[k for k,v in label_map.items() if v==0][0]}' → 0")

# Features — all columns except target, keep only numeric
X = df.drop(columns=[TARGET_COL])

# Column 92 has mixed types — coerce to numeric, fill errors with 0
for col in X.columns:
    if X[col].dtype == object:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

X = X.select_dtypes(include=[np.number])
print(f"  Feature count        : {X.shape[1]}")

print(f"\n  Class distribution:")
vc = y.value_counts().sort_index()
for cls, cnt in vc.items():
    label = "Malware (S)" if cls == 1 else "Benign (B)"
    print(f"    {label} : {cnt}  ({cnt/len(y)*100:.1f}%)")

n_malware = int(vc.get(1, 0))
n_benign  = int(vc.get(0, 0))

# Determine majority class and set scale_pos_weight accordingly
if n_malware > n_benign:
    print(f"\n  NOTE: Malware is MAJORITY — scale_pos_weight boosts benign")
    spw = n_malware / n_benign
else:
    print(f"\n  NOTE: Benign is MAJORITY — scale_pos_weight boosts malware")
    spw = n_benign / n_malware

print(f"  scale_pos_weight     : {spw:.2f}")

# Stratified train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n  Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")

# Recalculate spw from training set only (more accurate)
spw = (y_train == 1).sum() / (y_train == 0).sum()

# =============================================================================
#  3. HELPERS
# =============================================================================

def evaluate(name, y_true, y_pred, y_prob=None, print_report=False):
    """Compute and return all metrics including false negative count."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  {'─'*55}")
    print(f"  {name}")
    print(f"  {'─'*55}")
    print(f"    Accuracy       : {acc*100:.2f}%")
    print(f"    Precision      : {prec*100:.2f}%")
    print(f"    Recall         : {rec*100:.2f}%")
    print(f"    F1-Score       : {f1*100:.2f}%")
    if auc is not None:
        print(f"    AUC-ROC        : {auc*100:.2f}%")
    print(f"    False Negatives: {fn}  (malware missed)")
    print(f"    False Positives: {fp}  (benign flagged as malware)")

    if print_report:
        print(f"\n    Classification Report:")
        report = classification_report(
            y_true, y_pred, target_names=["Benign (B)", "Malware (S)"]
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
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def save_cm(name, y_true, y_pred, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=["Benign (B)", "Malware (S)"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()


# =============================================================================
#  OBJECTIVE 1 — Replicate 5 baseline classifiers from Palma et al. (2024)
#                and document deviations from originally reported results
# =============================================================================

print("\n\n" + "=" * 65)
print("  OBJECTIVE 1: Baseline Replication (Palma et al., 2024)")
print("=" * 65)

baseline_defs = {
    "Random Forest (RF)": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Support Vector Machine (SVM)": SVC(
        kernel="rbf", probability=True, random_state=RANDOM_STATE
    ),
    "k-Nearest Neighbour (KNN)": KNeighborsClassifier(
        n_neighbors=5, n_jobs=-1
    ),
    "Naive Bayes (NB)": GaussianNB(),
    "Multilayer Perceptron (MLP)": MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=300, random_state=RANDOM_STATE
    ),
}

baseline_results = []

for name, clf in baseline_defs.items():
    print(f"\n  Training {name} ...")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1] \
             if hasattr(clf, "predict_proba") else None
    metrics = evaluate(name, y_test, y_pred, y_prob)
    baseline_results.append(metrics)
    safe = name.split("(")[1].rstrip(")").strip().lower()
    save_cm(name, y_test, y_pred, f"cm_obj1_{safe}.png")

baseline_df = pd.DataFrame(baseline_results)

# --- Deviation table vs Palma et al. ----------------------------------------
print("\n\n  --- Deviation Analysis vs Palma et al. (2024) ---")
print(f"\n  {'Model':<35} {'Palma Acc':>10} {'Our Acc':>10} {'Deviation':>10}")
print(f"  {'─'*68}")

deviation_rows = []
for row in baseline_results:
    name      = row["Model"]
    our_acc   = row["Accuracy"]
    palma_acc = PALMA_RESULTS.get(name, {}).get("Accuracy", None)
    if palma_acc:
        dev  = our_acc - palma_acc
        sign = "+" if dev >= 0 else ""
        print(f"  {name:<35} {palma_acc:>10.2f} {our_acc:>10.2f} "
              f"{sign+str(round(dev,2)):>10}")
        deviation_rows.append({
            "Model":                  name,
            "Palma et al. Accuracy":  palma_acc,
            "Our Accuracy (Drebin)":  our_acc,
            "Deviation":              round(dev, 2),
        })

dev_df = pd.DataFrame(deviation_rows)
dev_df.to_csv(os.path.join(RESULTS_DIR, "obj1_deviation_analysis.csv"), index=False)
print(f"\n  NOTE: Deviations expected — Drebin vs CICAndMal2017 datasets differ")
print(f"  Saved -> {RESULTS_DIR}/obj1_deviation_analysis.csv")

# =============================================================================
#  OBJECTIVE 2 — XGBoost standalone: without SMOTE and with SMOTE
# =============================================================================

print("\n\n" + "=" * 65)
print("  OBJECTIVE 2: XGBoost Standalone (without SMOTE vs with SMOTE)")
print("=" * 65)

xgb_results = []

# --- 2a. XGBoost WITHOUT SMOTE -----------------------------------------------
print("\n  2a. XGBoost — WITHOUT SMOTE ...")

xgb_no_smote = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(
        n_estimators=100,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])
xgb_no_smote.fit(X_train, y_train)
y_pred_xgb = xgb_no_smote.predict(X_test)
y_prob_xgb = xgb_no_smote.predict_proba(X_test)[:, 1]
m_xgb = evaluate("XGBoost (no SMOTE)", y_test, y_pred_xgb, y_prob_xgb)
xgb_results.append(m_xgb)
save_cm("XGBoost (no SMOTE)", y_test, y_pred_xgb, "cm_obj2_xgb_nosmote.png")

# --- 2b. XGBoost WITH SMOTE --------------------------------------------------
print("\n  2b. XGBoost — WITH SMOTE ...")

xgb_smote = ImbPipeline([
    ("smote",  SMOTE(random_state=RANDOM_STATE)),
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(
        n_estimators=100,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])
xgb_smote.fit(X_train, y_train)
y_pred_xgb_s = xgb_smote.predict(X_test)
y_prob_xgb_s = xgb_smote.predict_proba(X_test)[:, 1]
m_xgb_s = evaluate("XGBoost (with SMOTE)", y_test, y_pred_xgb_s, y_prob_xgb_s)
xgb_results.append(m_xgb_s)
save_cm("XGBoost (with SMOTE)", y_test, y_pred_xgb_s, "cm_obj2_xgb_smote.png")

xgb_df = pd.DataFrame(xgb_results)

print("\n  --- XGBoost SMOTE Impact Summary ---")
print(f"  {'Metric':<16} {'No SMOTE':>12} {'With SMOTE':>12} {'Change':>10}")
print(f"  {'─'*53}")
for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    v1   = m_xgb[col]
    v2   = m_xgb_s[col]
    diff = v2 - v1
    sign = "+" if diff >= 0 else ""
    print(f"  {col:<16} {v1:>11.2f}% {v2:>11.2f}% "
          f"{sign+str(round(diff,2))+'%':>10}")

xgb_df.to_csv(os.path.join(RESULTS_DIR, "obj2_xgboost_comparison.csv"), index=False)
print(f"\n  Saved -> {RESULTS_DIR}/obj2_xgboost_comparison.csv")

# =============================================================================
#  OBJECTIVE 3 — RF + XGBoost Soft Voting Ensemble WITH SMOTE
#                + false negative reduction analysis
# =============================================================================

print("\n\n" + "=" * 65)
print("  OBJECTIVE 3: RF+XGBoost Soft Voting Ensemble with SMOTE")
print("=" * 65)

rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
)

xgb_clf = XGBClassifier(
    n_estimators=100,
    scale_pos_weight=spw,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

soft_voting = VotingClassifier(
    estimators=[("rf", rf_clf), ("xgb", xgb_clf)],
    voting="soft",
)

proposed_pipe = ImbPipeline([
    ("smote",    SMOTE(random_state=RANDOM_STATE)),
    ("scaler",   StandardScaler()),
    ("ensemble", soft_voting),
])

print("\n  Fitting RF+XGBoost soft voting ensemble with SMOTE ...")
proposed_pipe.fit(X_train, y_train)

y_pred_prop = proposed_pipe.predict(X_test)
y_prob_prop = proposed_pipe.predict_proba(X_test)[:, 1]

proposed_metrics = evaluate(
    "RF+XGBoost Soft Voting + SMOTE",
    y_test, y_pred_prop, y_prob_prop,
    print_report=True
)
save_cm("RF+XGBoost Soft Voting + SMOTE",
        y_test, y_pred_prop, "cm_obj3_proposed.png")

# --- Cross-validation --------------------------------------------------------
print("\n  Running 5-fold stratified cross-validation ...")
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_results = cross_validate(
    proposed_pipe, X, y,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    return_train_score=False,
)

print(f"\n  {'Metric':<12}  {'Mean':>8}  {'Std':>8}")
print(f"  {'─'*32}")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    scores = cv_results[f"test_{metric}"]
    print(f"  {metric:<12}  {scores.mean()*100:>7.2f}%  "
          f"+/-{scores.std()*100:>5.2f}%")

# =============================================================================
#  4. FULL COMPARISON TABLE (all models)
# =============================================================================

print("\n\n" + "=" * 65)
print("  FULL COMPARISON — All Models")
print("=" * 65)

all_results = pd.concat([
    baseline_df,
    xgb_df,
    pd.DataFrame([proposed_metrics]),
], ignore_index=True)

display_cols = ["Model", "Accuracy", "Precision", "Recall",
                "F1-Score", "AUC-ROC", "False Negatives", "False Positives"]
print("\n" + all_results[display_cols].to_string(index=False))
all_results[display_cols].to_csv(
    os.path.join(RESULTS_DIR, "full_results_summary.csv"), index=False
)
print(f"\n  Saved -> {RESULTS_DIR}/full_results_summary.csv")

# =============================================================================
#  5. FALSE NEGATIVE ANALYSIS
# =============================================================================

print("\n\n" + "=" * 65)
print("  FALSE NEGATIVE ANALYSIS")
print("  (lower = fewer malware samples missed)")
print("=" * 65)

fn_data = all_results[["Model", "False Negatives", "Recall"]].sort_values(
    "False Negatives"
)
print("\n" + fn_data.to_string(index=False))

best_baseline_fn = baseline_df["False Negatives"].min()
proposed_fn      = proposed_metrics["False Negatives"]
fn_improvement   = best_baseline_fn - proposed_fn

print(f"\n  Best baseline FN  : {best_baseline_fn}")
print(f"  Proposed model FN : {proposed_fn}")
if fn_improvement > 0:
    print(f"  Improvement       : {fn_improvement} fewer malware samples missed ✓")
elif fn_improvement == 0:
    print(f"  No change in false negatives")
else:
    print(f"  Increase of {abs(fn_improvement)} more missed — "
          f"see F1 and AUC for context")

# =============================================================================
#  6. VISUALISATIONS
# =============================================================================

print("\n\n[PLOTS] Generating all visualisations ...")

n_base = len(baseline_defs)
BLUE   = "#B5D4F4"
PURPLE = "#534AB7"
DPURP  = "#3C3489"
AMBER  = "#EF9F27"
RED    = "#A32D2D"

bar_colors = [BLUE] * n_base + [AMBER, AMBER] + [PURPLE]


def hbar_plot(title, xlabel, values, models, colors, filename,
              vline=None, vline_label=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(models, values, color=colors, edgecolor="white", height=0.6)
    if vline:
        ax.axvline(vline, color=RED, linewidth=1.2, linestyle="--",
                   label=vline_label)
        ax.legend(fontsize=9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%" if isinstance(val, float) else str(val),
                va="center", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.18)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()


# 6a. Accuracy comparison
hbar_plot(
    "Accuracy comparison — all models (Drebin)",
    "Accuracy (%)",
    all_results["Accuracy"].tolist(),
    all_results["Model"].tolist(),
    bar_colors,
    "plot_accuracy_comparison.png",
    vline=79.14,
    vline_label="Palma et al. best (79.14%)"
)

# 6b. F1-Score comparison
hbar_plot(
    "F1-Score comparison — all models  [PRIMARY METRIC]",
    "F1-Score (%)",
    all_results["F1-Score"].tolist(),
    all_results["Model"].tolist(),
    bar_colors,
    "plot_f1_comparison.png"
)

# 6c. Recall comparison
hbar_plot(
    "Recall (malware detection rate) — all models",
    "Recall (%)",
    all_results["Recall"].tolist(),
    all_results["Model"].tolist(),
    bar_colors,
    "plot_recall_comparison.png"
)

# 6d. False Negatives
fig, ax = plt.subplots(figsize=(11, 6))
fn_vals   = all_results["False Negatives"].tolist()
fn_models = all_results["Model"].tolist()
fn_colors = [BLUE] * n_base + [AMBER, AMBER] + [PURPLE]
bars = ax.barh(fn_models, fn_vals, color=fn_colors, edgecolor="white", height=0.6)
for bar, val in zip(bars, fn_vals):
    ax.text(bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9)
ax.set_xlabel("False Negatives (malware missed — lower is better)", fontsize=10)
ax.set_title("False Negative analysis — all models", fontsize=11, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plot_false_negatives.png"), dpi=150)
plt.close()

# 6e. Metrics heatmap
metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
heatmap_data = all_results.set_index("Model")[metrics_cols].astype(float)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Score (%)"})
ax.set_title("All models — metrics heatmap (Drebin)",
             fontsize=11, fontweight="bold")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plot_metrics_heatmap.png"), dpi=150)
plt.close()

# 6f. XGBoost SMOTE impact
fig, ax = plt.subplots(figsize=(7, 4))
x     = np.arange(4)
width = 0.35
cols  = ["Accuracy", "Precision", "Recall", "F1-Score"]
v1    = [m_xgb[c] for c in cols]
v2    = [m_xgb_s[c] for c in cols]
ax.bar(x - width/2, v1, width, label="XGBoost (no SMOTE)",
       color=AMBER, edgecolor="white")
ax.bar(x + width/2, v2, width, label="XGBoost (with SMOTE)",
       color=DPURP, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(cols, fontsize=10)
ax.set_ylim(0, 115)
ax.set_ylabel("Score (%)", fontsize=10)
ax.set_title("Objective 2: XGBoost — impact of SMOTE",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plot_obj2_xgb_smote_impact.png"), dpi=150)
plt.close()

# 6g. Proposed model per-metric
fig, ax = plt.subplots(figsize=(7, 4))
bar_colors_prop = [PURPLE, "#0F6E56", RED, "#185FA5"]
vals = [proposed_metrics[c] for c in cols]
bars = ax.bar(cols, vals, color=bar_colors_prop, width=0.5, edgecolor="white")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 115)
ax.set_ylabel("Score (%)", fontsize=10)
ax.set_title("Objective 3: Proposed model — per-metric performance",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plot_obj3_proposed_metrics.png"), dpi=150)
plt.close()

print(f"  7 plots saved to /{RESULTS_DIR}/")

# =============================================================================
#  7. FINAL VERDICT — mapped to each objective
# =============================================================================

best_bl_acc = baseline_df["Accuracy"].max()
best_bl_f1  = baseline_df["F1-Score"].max()
best_bl_fn  = baseline_df["False Negatives"].min()
prop_acc    = proposed_metrics["Accuracy"]
prop_f1     = proposed_metrics["F1-Score"]
prop_fn     = proposed_metrics["False Negatives"]


def delta(new, old):
    sign = "+ " if new >= old else "- "
    return f"({sign}{abs(new - old):.2f}%)"


print("\n\n" + "=" * 65)
print("  FINAL VERDICT — Mapped to Study Objectives (Drebin)")
print("=" * 65)

print(f"""
  OBJECTIVE 1 — Baseline Replication
  ─────────────────────────────────────────────────────────────
  All 5 classifiers (RF, SVM, KNN, NB, MLP) replicated.
  Deviation analysis saved to : obj1_deviation_analysis.csv
  Best replicated accuracy    : {best_bl_acc:.2f}%
  (Palma et al. reported      : 79.14% on CICAndMal2017)

  OBJECTIVE 2 — XGBoost Standalone
  ─────────────────────────────────────────────────────────────
  XGBoost without SMOTE : Acc={m_xgb['Accuracy']:.2f}%  F1={m_xgb['F1-Score']:.2f}%  FN={m_xgb['False Negatives']}
  XGBoost with SMOTE    : Acc={m_xgb_s['Accuracy']:.2f}%  F1={m_xgb_s['F1-Score']:.2f}%  FN={m_xgb_s['False Negatives']}
  SMOTE impact saved to : obj2_xgboost_comparison.csv

  OBJECTIVE 3 — Proposed RF+XGBoost Soft Voting Ensemble + SMOTE
  ─────────────────────────────────────────────────────────────
  Best baseline F1       : {best_bl_f1:.2f}%
  Proposed F1            : {prop_f1:.2f}%  {delta(prop_f1, best_bl_f1)}
  Best baseline accuracy : {best_bl_acc:.2f}%
  Proposed accuracy      : {prop_acc:.2f}%  {delta(prop_acc, best_bl_acc)}
  Best baseline FN       : {best_bl_fn}
  Proposed FN            : {prop_fn}  ({best_bl_fn - prop_fn:+d} vs best baseline)
  Full results saved to  : full_results_summary.csv
""")
print("=" * 65)
print(f"  All outputs saved to : {os.path.abspath(RESULTS_DIR)}")
print("  Done.\n")