# =============================================================================
#  PHASE 1 — Baseline Replication
#  Replicates Palma, Ferreira & Figueiredo (2024)
#  5 classifiers: RF, SVM, KNN, NB, MLP
#  No resampling · StandardScaler only · RRFS explainability
# =============================================================================

import matplotlib
matplotlib.use("Agg")  # prevents tkinter crash on Windows

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import r_regression
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

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
#  CONFIGURATION
# =============================================================================

DATASET_PATH = "dataset.csv"
TARGET_COL   = "Class"          # 1 = Malware, 0 = Benign
DROP_COLS    = ["App", "Package", "Category", "Description", "Related apps"]
TEST_SIZE    = 0.2
RANDOM_STATE = 42
RESULTS_DIR  = "results"
RRFS_TOP_N   = 20               # number of top features to keep via RRFS

os.makedirs(RESULTS_DIR, exist_ok=True)

# Published results from Palma et al. (2024) on CICAndMal2017
PALMA_RESULTS = {
    "Random Forest (RF)":           {"Accuracy": 79.14, "F1-Score": None},
    "Support Vector Machine (SVM)": {"Accuracy": 70.81, "F1-Score": None},
    "k-Nearest Neighbour (KNN)":    {"Accuracy": 62.01, "F1-Score": None},
    "Naive Bayes (NB)":             {"Accuracy": 74.53, "F1-Score": None},
    "Multilayer Perceptron (MLP)":  {"Accuracy": 78.12, "F1-Score": None},
}

# =============================================================================
#  1. DATA LOADING & PREPROCESSING
# =============================================================================

print("=" * 65)
print("  PHASE 1 — Baseline Replication (Palma et al., 2024)")
print("=" * 65)

print("\n[1] Loading dataset ...")
df = pd.read_csv(DATASET_PATH, low_memory=False)
print(f"    Raw shape         : {df.shape}")

# Drop specified non-feature columns
if DROP_COLS:
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Detect target column if default not found
if TARGET_COL not in df.columns:
    print(f"\n  WARNING: '{TARGET_COL}' not found. Trying last column ...")
    TARGET_COL = df.columns[-1]
    print(f"  Using '{TARGET_COL}' as target column.")

# Replace missing values with column medians (numeric only)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print(f"    Missing values    : filled with column medians")

# Remove duplicate rows
before = len(df)
df.drop_duplicates(inplace=True)
print(f"    Duplicates removed: {before - len(df)}")

# Encode target if categorical
if df[TARGET_COL].dtype == object:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
    print(f"    Label encoding    : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Features and target
X = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number])
y = df[TARGET_COL]

# Remove zero-variance features
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=0.0)
X = pd.DataFrame(vt.fit_transform(X),
                 columns=X.columns[vt.get_support()])
print(f"    After zero-var    : {X.shape[1]} features remain")

print(f"\n    Class distribution:")
vc = y.value_counts().sort_index()
for cls, cnt in vc.items():
    label = "Malware" if cls == 1 else "Benign"
    print(f"      {label} ({cls}) : {cnt}  ({cnt/len(y)*100:.1f}%)")

n_benign  = int(vc.get(0, 0))
n_malware = int(vc.get(1, 0))
ratio     = n_benign / n_malware if n_malware > 0 else 0
print(f"    Imbalance ratio   : {ratio:.1f}:1 (benign:malware)")

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n    Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")

# =============================================================================
#  2. RRFS — Relevance-Redundancy Feature Selection (baseline explainability)
# =============================================================================

print("\n\n[2] RRFS — Relevance-Redundancy Feature Selection ...")

# Step 1: Relevance — Pearson correlation with target
correlations = np.abs(r_regression(X_train, y_train))
relevance    = pd.Series(correlations, index=X_train.columns).sort_values(ascending=False)

# Step 2: Redundancy removal — iteratively remove features correlated >0.9
# with a higher-relevance feature already selected
selected_features = []
corr_matrix = X_train.corr().abs()

for feat in relevance.index:
    if len(selected_features) == 0:
        selected_features.append(feat)
    else:
        redundant = False
        for selected in selected_features:
            if corr_matrix.loc[feat, selected] > 0.9:
                redundant = True
                break
        if not redundant:
            selected_features.append(feat)
    if len(selected_features) >= RRFS_TOP_N:
        break

print(f"    Features selected : {len(selected_features)} (from {X_train.shape[1]} total)")
print(f"    Top 10 features   : {selected_features[:10]}")

# Save RRFS feature list
rrfs_df = pd.DataFrame({
    "Rank":      range(1, len(selected_features) + 1),
    "Feature":   selected_features,
    "Relevance": [relevance[f] for f in selected_features],
})
rrfs_df.to_csv(os.path.join(RESULTS_DIR, "phase1_rrfs_top_features.csv"), index=False)
print(f"    Saved -> results/phase1_rrfs_top_features.csv")

# RRFS feature importance bar plot
fig, ax = plt.subplots(figsize=(9, 6))
rrfs_df.sort_values("Relevance").plot(
    kind="barh", x="Feature", y="Relevance",
    ax=ax, color="#B5D4F4", edgecolor="white", legend=False
)
ax.set_title("RRFS — Top selected features by relevance (Phase 1)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Pearson correlation with target", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase1_rrfs_feature_importance.png"), dpi=150)
plt.close()

# =============================================================================
#  3. HELPERS
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
        print(f"\n    Classification Report:")
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
#  4. TRAIN 5 BASELINE CLASSIFIERS (no resampling)
# =============================================================================

print("\n\n[3] Training 5 baseline classifiers (no resampling) ...")

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
    metrics = evaluate(name, y_test, y_pred, y_prob, print_report=True)
    baseline_results.append(metrics)
    safe = name.split("(")[1].rstrip(")").strip().lower()
    save_cm(name, y_test, y_pred, f"cm_phase1_{safe}.png")

baseline_df = pd.DataFrame(baseline_results)

# =============================================================================
#  5. DEVIATION ANALYSIS vs PALMA et al.
# =============================================================================

print("\n\n[4] Deviation analysis vs Palma et al. (2024) ...")
print(f"\n  {'Model':<35} {'Palma Acc':>10} {'Our Acc':>10} {'Deviation':>10}")
print(f"  {'─'*68}")

deviation_rows = []
for row in baseline_results:
    name      = row["Model"]
    our_acc   = row["Accuracy"]
    palma_acc = PALMA_RESULTS.get(name, {}).get("Accuracy")
    if palma_acc:
        dev  = our_acc - palma_acc
        sign = "+" if dev >= 0 else ""
        print(f"  {name:<35} {palma_acc:>10.2f} {our_acc:>10.2f} "
              f"{sign+str(round(dev,2)):>10}")
        deviation_rows.append({
            "Model":                 name,
            "Palma et al. Accuracy": palma_acc,
            "Our Accuracy":          our_acc,
            "Deviation":             round(dev, 2),
            "Notes":                 "Different dataset version / feature set"
        })

dev_df = pd.DataFrame(deviation_rows)
dev_df.to_csv(
    os.path.join(RESULTS_DIR, "phase1_deviation_analysis.csv"), index=False
)
print(f"\n  Saved -> results/phase1_deviation_analysis.csv")

# =============================================================================
#  6. SAVE BASELINE RESULTS
# =============================================================================

display_cols = ["Model", "Accuracy", "Precision", "Recall",
                "F1-Score", "AUC-ROC", "False Negatives", "False Positives"]
baseline_df[display_cols].to_csv(
    os.path.join(RESULTS_DIR, "phase1_baseline_results.csv"), index=False
)
print(f"  Saved -> results/phase1_baseline_results.csv")

# =============================================================================
#  7. VISUALISATIONS
# =============================================================================

print("\n[5] Generating Phase 1 plots ...")

BLUE   = "#B5D4F4"
RED    = "#A32D2D"
models = baseline_df["Model"].tolist()

def hbar(title, xlabel, values, colors, filename, vline=None, vline_label=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, values, color=colors,
                   edgecolor="white", height=0.6)
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

hbar("Phase 1 — Accuracy comparison",
     "Accuracy (%)",
     baseline_df["Accuracy"].tolist(),
     [BLUE] * 5,
     "phase1_accuracy.png",
     vline=79.14,
     vline_label="Palma et al. best (79.14%)")

hbar("Phase 1 — F1-Score comparison",
     "F1-Score (%)",
     baseline_df["F1-Score"].tolist(),
     [BLUE] * 5,
     "phase1_f1.png")

hbar("Phase 1 — Recall comparison",
     "Recall (%)",
     baseline_df["Recall"].tolist(),
     [BLUE] * 5,
     "phase1_recall.png")

# False negatives
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(models, baseline_df["False Negatives"].tolist(),
               color=BLUE, edgecolor="white", height=0.6)
for bar, val in zip(bars, baseline_df["False Negatives"].tolist()):
    ax.text(bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            str(val), va="center", fontsize=9)
ax.set_xlabel("False Negatives (malware missed — lower is better)", fontsize=10)
ax.set_title("Phase 1 — False Negative analysis", fontsize=11, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase1_false_negatives.png"), dpi=150)
plt.close()

# Metrics heatmap
fig, ax = plt.subplots(figsize=(8, 5))
hmap = baseline_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].astype(float)
sns.heatmap(hmap, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Score (%)"})
ax.set_title("Phase 1 — Metrics heatmap", fontsize=11, fontweight="bold")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "phase1_metrics_heatmap.png"), dpi=150)
plt.close()

print("  6 plots saved to /results/")

# =============================================================================
#  8. PHASE 1 SUMMARY
# =============================================================================

best_acc = baseline_df["Accuracy"].max()
best_rec = baseline_df["Recall"].max()
best_f1  = baseline_df["F1-Score"].max()
best_fn  = baseline_df["False Negatives"].min()

print("\n\n" + "=" * 65)
print("  PHASE 1 COMPLETE — Summary")
print("=" * 65)
print(f"""
  Classifiers evaluated : RF, SVM, KNN, NB, MLP (no resampling)
  RRFS features selected: {len(selected_features)} features saved

  Best accuracy  : {best_acc:.2f}%  (Palma et al. reported: 79.14%)
  Best recall    : {best_rec:.2f}%
  Best F1-Score  : {best_f1:.2f}%
  Lowest FN      : {best_fn}

  Files saved:
    results/phase1_baseline_results.csv
    results/phase1_deviation_analysis.csv
    results/phase1_rrfs_top_features.csv
    results/phase1_rrfs_feature_importance.png
    results/cm_phase1_*.png  (5 confusion matrices)
    results/phase1_*.png     (4 comparison plots + heatmap)
""")
print("=" * 65)
print(f"\n  Next step: Run phase2_smote_tomek.py")
print("  Done.\n")