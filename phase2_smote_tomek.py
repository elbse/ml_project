# =============================================================================
#  PHASE 2 — Controlled SMOTE-Tomek Experiment
#  Retrains all 5 baseline classifiers WITH SMOTETomek
#  Adds standalone XGBoost (without and with SMOTETomek)
#  Side-by-side comparison isolating resampling contribution
# =============================================================================

import matplotlib
matplotlib.use("Agg")

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

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
RESULTS_DIR  = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
#  1. DATA LOADING (same preprocessing as Phase 1)
# =============================================================================

print("=" * 65)
print("  PHASE 2 — SMOTE-Tomek Controlled Experiment")
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

print(f"    Shape            : {X.shape}")
print(f"\n    Class distribution:")
vc = y.value_counts().sort_index()
for cls, cnt in vc.items():
    label = "Malware" if cls == 1 else "Benign"
    print(f"      {label} ({cls}) : {cnt}  ({cnt/len(y)*100:.1f}%)")

# scale_pos_weight for XGBoost = majority / minority
n_malware = int(vc.get(1, 0))
n_benign  = int(vc.get(0, 0))
spw = n_malware / n_benign if n_benign > 0 else 1.0
print(f"\n    XGBoost scale_pos_weight : {spw:.2f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"    Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")

# Verify SMOTE-Tomek will be applied to training set only
print(f"\n    Train class counts before resampling:")
tr_vc = pd.Series(y_train).value_counts().sort_index()
for cls, cnt in tr_vc.items():
    label = "Malware" if cls == 1 else "Benign"
    print(f"      {label} : {cnt}")

# =============================================================================
#  2. HELPERS
# =============================================================================

def evaluate(name, y_true, y_pred, y_prob=None):
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
#  3. CLASSIFIERS WITHOUT SMOTE-TOMEK (load from Phase 1 results)
# =============================================================================

print("\n\n[2] Loading Phase 1 baseline results (no resampling) ...")

phase1_path = os.path.join(RESULTS_DIR, "phase1_baseline_results.csv")
if os.path.exists(phase1_path):
    phase1_df = pd.read_csv(phase1_path)
    print(f"    Loaded {len(phase1_df)} baseline results from Phase 1.")
else:
    print("    WARNING: phase1_baseline_results.csv not found.")
    print("    Run phase1_baseline.py first, then re-run this script.")
    exit(1)

# =============================================================================
#  4. CLASSIFIERS WITH SMOTE-TOMEK
# =============================================================================

print("\n\n[3] Training classifiers WITH SMOTE-Tomek ...")
print("    (Resampling applied to training set only — test set unchanged)")

smote_tomek = SMOTETomek(random_state=RANDOM_STATE)

classifier_defs = {
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

smote_results = []

for name, clf in classifier_defs.items():
    print(f"\n  Training {name} + SMOTE-Tomek ...")
    pipe = ImbPipeline([
        ("smote_tomek", SMOTETomek(random_state=RANDOM_STATE)),
        ("scaler",      StandardScaler()),
        ("clf",         clf),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1] \
             if hasattr(clf, "predict_proba") else None
    metrics = evaluate(f"{name} + SMOTE-Tomek", y_test, y_pred, y_prob)
    metrics["Model"] = name           # keep clean name for comparison table
    metrics["Resampling"] = "SMOTE-Tomek"
    smote_results.append(metrics)
    safe = name.split("(")[1].rstrip(")").strip().lower()
    save_cm(f"{name} + SMOTE-Tomek", y_test, y_pred,
            f"cm_phase2_{safe}_smote.png")

smote_df = pd.DataFrame(smote_results)

# =============================================================================
#  5. XGBOOST STANDALONE — without and with SMOTE-Tomek
# =============================================================================

print("\n\n[4] XGBoost standalone — without and with SMOTE-Tomek ...")
xgb_results = []

# 5a. XGBoost WITHOUT SMOTE-Tomek
print("\n  5a. XGBoost — WITHOUT SMOTE-Tomek ...")
xgb_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(
        n_estimators=100,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])
xgb_pipe.fit(X_train, y_train)
y_pred_xgb = xgb_pipe.predict(X_test)
y_prob_xgb = xgb_pipe.predict_proba(X_test)[:, 1]
m_xgb = evaluate("XGBoost (no SMOTE-Tomek)", y_test, y_pred_xgb, y_prob_xgb)
m_xgb["Model"]      = "XGBoost"
m_xgb["Resampling"] = "None"
xgb_results.append(m_xgb)
save_cm("XGBoost (no SMOTE-Tomek)", y_test, y_pred_xgb,
        "cm_phase2_xgb_nosmote.png")

# 5b. XGBoost WITH SMOTE-Tomek
print("\n  5b. XGBoost — WITH SMOTE-Tomek ...")
xgb_smote_pipe = ImbPipeline([
    ("smote_tomek", SMOTETomek(random_state=RANDOM_STATE)),
    ("scaler",      StandardScaler()),
    ("clf", XGBClassifier(
        n_estimators=100,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])
xgb_smote_pipe.fit(X_train, y_train)
y_pred_xgb_s = xgb_smote_pipe.predict(X_test)
y_prob_xgb_s = xgb_smote_pipe.predict_proba(X_test)[:, 1]
m_xgb_s = evaluate("XGBoost + SMOTE-Tomek", y_test, y_pred_xgb_s, y_prob_xgb_s)
m_xgb_s["Model"]      = "XGBoost"
m_xgb_s["Resampling"] = "SMOTE-Tomek"
xgb_results.append(m_xgb_s)
save_cm("XGBoost + SMOTE-Tomek", y_test, y_pred_xgb_s,
        "cm_phase2_xgb_smote.png")

# =============================================================================
#  6. SIDE-BY-SIDE COMPARISON TABLE
# =============================================================================

print("\n\n[5] Building side-by-side comparison table ...")

# Add resampling tag to phase1 results
phase1_tagged = phase1_df.copy()
phase1_tagged["Resampling"] = "None"

# Combine all Phase 2 results
all_phase2 = pd.concat([
    phase1_tagged,
    smote_df,
    pd.DataFrame(xgb_results),
], ignore_index=True)

# Build clean comparison: each model has two rows (None vs SMOTE-Tomek)
display_cols = ["Model", "Resampling", "Accuracy", "Precision",
                "Recall", "F1-Score", "AUC-ROC",
                "False Negatives", "False Positives"]

# Keep only columns that exist
display_cols = [c for c in display_cols if c in all_phase2.columns]

print("\n" + all_phase2[display_cols].to_string(index=False))
all_phase2[display_cols].to_csv(
    os.path.join(RESULTS_DIR, "phase2_smote_tomek_comparison.csv"), index=False
)
print(f"\n  Saved -> results/phase2_smote_tomek_comparison.csv")

# =============================================================================
#  7. SMOTE-TOMEK IMPACT SUMMARY (per model)
# =============================================================================

print("\n\n[6] SMOTE-Tomek impact per model ...")
print(f"\n  {'Model':<35} {'Recall (None)':>14} {'Recall (SMOTE)':>15} "
      f"{'Change':>8}  {'F1 (None)':>10} {'F1 (SMOTE)':>11} {'Change':>8}")
print(f"  {'─'*105}")

models_to_compare = ["Random Forest (RF)", "Support Vector Machine (SVM)",
                     "k-Nearest Neighbour (KNN)", "Naive Bayes (NB)",
                     "Multilayer Perceptron (MLP)", "XGBoost"]

impact_rows = []
for model in models_to_compare:
    row_none  = all_phase2[
        (all_phase2["Model"] == model) & (all_phase2["Resampling"] == "None")
    ]
    row_smote = all_phase2[
        (all_phase2["Model"] == model) & (all_phase2["Resampling"] == "SMOTE-Tomek")
    ]
    if row_none.empty or row_smote.empty:
        continue

    r_none  = row_none["Recall"].values[0]
    r_smote = row_smote["Recall"].values[0]
    f_none  = row_none["F1-Score"].values[0]
    f_smote = row_smote["F1-Score"].values[0]
    r_diff  = r_smote - r_none
    f_diff  = f_smote - f_none

    r_sign = "+" if r_diff >= 0 else ""
    f_sign = "+" if f_diff >= 0 else ""

    print(f"  {model:<35} {r_none:>13.2f}% {r_smote:>14.2f}% "
          f"{r_sign+str(round(r_diff,2))+'%':>8}  "
          f"{f_none:>9.2f}% {f_smote:>10.2f}% "
          f"{f_sign+str(round(f_diff,2))+'%':>8}")

    impact_rows.append({
        "Model":           model,
        "Recall (None)":   r_none,
        "Recall (SMOTE-Tomek)": r_smote,
        "Recall Change":   round(r_diff, 2),
        "F1 (None)":       f_none,
        "F1 (SMOTE-Tomek)": f_smote,
        "F1 Change":       round(f_diff, 2),
    })

impact_df = pd.DataFrame(impact_rows)
impact_df.to_csv(
    os.path.join(RESULTS_DIR, "phase2_smote_impact_summary.csv"), index=False
)
print(f"\n  Saved -> results/phase2_smote_impact_summary.csv")

# =============================================================================
#  8. VISUALISATIONS
# =============================================================================

print("\n[7] Generating Phase 2 plots ...")

BLUE   = "#B5D4F4"
PURPLE = "#534AB7"
AMBER  = "#EF9F27"
RED    = "#A32D2D"

# 8a. Grouped bar — Recall before vs after SMOTE-Tomek
fig, ax = plt.subplots(figsize=(11, 6))
x     = np.arange(len(models_to_compare))
width = 0.35

vals_none  = []
vals_smote = []
for m in models_to_compare:
    rn = all_phase2[(all_phase2["Model"]==m) & (all_phase2["Resampling"]=="None")]["Recall"]
    rs = all_phase2[(all_phase2["Model"]==m) & (all_phase2["Resampling"]=="SMOTE-Tomek")]["Recall"]
    vals_none.append(rn.values[0] if not rn.empty else 0)
    vals_smote.append(rs.values[0] if not rs.empty else 0)

bars1 = ax.bar(x - width/2, vals_none,  width, label="Without SMOTE-Tomek",
               color=BLUE, edgecolor="white")
bars2 = ax.bar(x + width/2, vals_smote, width, label="With SMOTE-Tomek",
               color=PURPLE, edgecolor="white")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", fontsize=8)

short_names = ["RF", "SVM", "KNN", "NB", "MLP", "XGBoost"]
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylim(0, 115)
ax.set_ylabel("Recall (%)", fontsize=10)
ax.set_title("Phase 2 — Recall: without vs with SMOTE-Tomek",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase2_recall_comparison.png"), dpi=150)
plt.close()

# 8b. Grouped bar — F1-Score before vs after SMOTE-Tomek
fig, ax = plt.subplots(figsize=(11, 6))
vals_f1_none  = []
vals_f1_smote = []
for m in models_to_compare:
    fn = all_phase2[(all_phase2["Model"]==m) & (all_phase2["Resampling"]=="None")]["F1-Score"]
    fs = all_phase2[(all_phase2["Model"]==m) & (all_phase2["Resampling"]=="SMOTE-Tomek")]["F1-Score"]
    vals_f1_none.append(fn.values[0] if not fn.empty else 0)
    vals_f1_smote.append(fs.values[0] if not fs.empty else 0)

bars1 = ax.bar(x - width/2, vals_f1_none,  width, label="Without SMOTE-Tomek",
               color=BLUE, edgecolor="white")
bars2 = ax.bar(x + width/2, vals_f1_smote, width, label="With SMOTE-Tomek",
               color=PURPLE, edgecolor="white")
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylim(0, 115)
ax.set_ylabel("F1-Score (%)", fontsize=10)
ax.set_title("Phase 2 — F1-Score: without vs with SMOTE-Tomek",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase2_f1_comparison.png"), dpi=150)
plt.close()

# 8c. False Negatives comparison
fig, ax = plt.subplots(figsize=(11, 6))
vals_fn_none  = []
vals_fn_smote = []
for m in models_to_compare:
    fn_n = all_phase2[(all_phase2["Model"]==m) & (all_phase2["Resampling"]=="None")]["False Negatives"]
    fn_s = all_phase2[(all_phase2["Model"]==m) & (all_phase2["Resampling"]=="SMOTE-Tomek")]["False Negatives"]
    vals_fn_none.append(fn_n.values[0] if not fn_n.empty else 0)
    vals_fn_smote.append(fn_s.values[0] if not fn_s.empty else 0)

bars1 = ax.bar(x - width/2, vals_fn_none,  width, label="Without SMOTE-Tomek",
               color=BLUE, edgecolor="white")
bars2 = ax.bar(x + width/2, vals_fn_smote, width, label="With SMOTE-Tomek",
               color=PURPLE, edgecolor="white")
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(int(bar.get_height())), ha="center", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(int(bar.get_height())), ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylabel("False Negatives (lower is better)", fontsize=10)
ax.set_title("Phase 2 — False Negatives: without vs with SMOTE-Tomek",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase2_false_negatives.png"), dpi=150)
plt.close()

# 8d. Metrics heatmap — all Phase 2 models with SMOTE-Tomek
smote_only = all_phase2[all_phase2["Resampling"] == "SMOTE-Tomek"].copy()
smote_only["Label"] = smote_only["Model"] + " + ST"
hmap_data = smote_only.set_index("Label")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].astype(float)

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(hmap_data, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Score (%)"})
ax.set_title("Phase 2 — Metrics heatmap (SMOTE-Tomek models)",
             fontsize=11, fontweight="bold")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase2_metrics_heatmap.png"), dpi=150)
plt.close()

# 8e. XGBoost SMOTE-Tomek impact
fig, ax = plt.subplots(figsize=(7, 4))
cols  = ["Accuracy", "Precision", "Recall", "F1-Score"]
xgb_none_vals  = [m_xgb[c] for c in cols]
xgb_smote_vals = [m_xgb_s[c] for c in cols]
xi    = np.arange(4)
wid   = 0.35
ax.bar(xi - wid/2, xgb_none_vals,  wid, label="XGBoost (no SMOTE-Tomek)",
       color=AMBER, edgecolor="white")
ax.bar(xi + wid/2, xgb_smote_vals, wid, label="XGBoost + SMOTE-Tomek",
       color=PURPLE, edgecolor="white")
ax.set_xticks(xi)
ax.set_xticklabels(cols, fontsize=10)
ax.set_ylim(0, 115)
ax.set_ylabel("Score (%)", fontsize=10)
ax.set_title("Phase 2 — XGBoost: impact of SMOTE-Tomek",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase2_xgb_smote_impact.png"), dpi=150)
plt.close()

print("  5 plots saved to /results/")

# =============================================================================
#  9. PHASE 2 SUMMARY
# =============================================================================

best_smote_rec = smote_df["Recall"].max()
best_smote_f1  = smote_df["F1-Score"].max()
best_smote_fn  = smote_df["False Negatives"].min()
best_xgb_f1    = max(m_xgb["F1-Score"], m_xgb_s["F1-Score"])

print("\n\n" + "=" * 65)
print("  PHASE 2 COMPLETE — Summary")
print("=" * 65)
print(f"""
  Models evaluated with SMOTE-Tomek : RF, SVM, KNN, NB, MLP
  XGBoost standalone                : without + with SMOTE-Tomek

  Best recall  (SMOTE-Tomek models) : {best_smote_rec:.2f}%
  Best F1-Score (SMOTE-Tomek models): {best_smote_f1:.2f}%
  Lowest FN    (SMOTE-Tomek models) : {best_smote_fn}
  Best XGBoost F1                   : {best_xgb_f1:.2f}%

  Files saved:
    results/phase2_smote_tomek_comparison.csv
    results/phase2_smote_impact_summary.csv
    results/cm_phase2_*.png  (7 confusion matrices)
    results/phase2_*.png     (5 comparison plots)
""")
print("=" * 65)
print("\n  Next step: Run phase3_ensemble_shap.py")
print("  Done.\n")