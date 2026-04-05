# =============================================================================
#  Android Malware Detection — Full Pipeline
#  Dataset  : CICAndMal2017 (permission-based features)
#  Baseline : RF, SVM, KNN, NB, MLP  (replicates Palma et al., 2024)
#  Proposed : SMOTE + Soft Voting Ensemble (RF + XGBoost)
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

# Baseline classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Proposed
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
#  1. CONFIGURATION
# =============================================================================

DATASET_PATH = "CICAndMal2017.csv"      # <-- update to your actual filename
TARGET_COL   = "Class"                  # 1 = malware, 0 = benign

# Non-feature text columns to drop before training
DROP_COLS = ["App", "Package", "Category", "Description", "Related apps"]

TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5
RESULTS_DIR  = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
#  2. DATA LOADING & PREPROCESSING
# =============================================================================

print("=" * 65)
print("  Android Malware Detection Pipeline")
print("=" * 65)

print("\n[1] Loading dataset ...")
df = pd.read_csv(DATASET_PATH)
print(f"    Raw shape       : {df.shape}")

# Drop non-feature text columns
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)
print(f"    After cleaning  : {df.shape}")

# Features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Keep only numeric features (safety net)
X = X.select_dtypes(include=[np.number])
print(f"    Feature count   : {X.shape[1]}")

print(f"\n    Class distribution:")
vc = y.value_counts().sort_index()
for cls, cnt in vc.items():
    label = "Malware" if cls == 1 else "Benign"
    print(f"      {label} ({cls}) : {cnt}  ({cnt/len(y)*100:.1f}%)")

n_benign  = int(vc.get(0, 0))
n_malware = int(vc.get(1, 0))
imbalance_ratio = n_benign / n_malware if n_malware > 0 else float("inf")
print(f"    Imbalance ratio : {imbalance_ratio:.1f}:1  (benign:malware)")

# Stratified train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"\n    Train : {X_train.shape[0]} samples  |  Test : {X_test.shape[0]} samples")

# =============================================================================
#  3. HELPER: EVALUATION
# =============================================================================

def evaluate(name, y_true, y_pred, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    print(f"\n  {'─'*55}")
    print(f"  {name}")
    print(f"  {'─'*55}")
    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Precision : {prec*100:.2f}%")
    print(f"    Recall    : {rec*100:.2f}%   <- malware detection rate")
    print(f"    F1-Score  : {f1*100:.2f}%")
    if auc is not None:
        print(f"    AUC-ROC   : {auc*100:.2f}%")

    return {
        "Model":     name,
        "Accuracy":  round(acc  * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall":    round(rec  * 100, 2),
        "F1-Score":  round(f1   * 100, 2),
        "AUC-ROC":   round(auc  * 100, 2) if auc is not None else None,
    }


def save_confusion_matrix(name, y_true, y_pred, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malware"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()

# =============================================================================
#  4. BASELINE CLASSIFIERS  (no SMOTE — replicates Palma et al., 2024)
# =============================================================================

print("\n\n[2] Baseline classifiers (no SMOTE) ...")

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
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=RANDOM_STATE
    ),
}

baseline_results = []

for name, clf in baseline_defs.items():
    print(f"\n    Training {name} ...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1] \
             if hasattr(clf, "predict_proba") else None

    metrics = evaluate(name, y_test, y_pred, y_prob)
    baseline_results.append(metrics)
    safe_name = name.split("(")[1].rstrip(")").strip().lower()
    save_confusion_matrix(name, y_test, y_pred,
                          f"cm_baseline_{safe_name}.png")

baseline_df = pd.DataFrame(baseline_results)

# =============================================================================
#  5. PROPOSED: SMOTE + RF + XGBOOST SOFT VOTING ENSEMBLE
# =============================================================================

print("\n\n[3] Proposed: SMOTE + Soft Voting Ensemble (RF + XGBoost) ...")

# scale_pos_weight = benign_train / malware_train
n_ben_tr = int((y_train == 0).sum())
n_mal_tr = int((y_train == 1).sum())
spw = n_ben_tr / n_mal_tr
print(f"\n    scale_pos_weight = {spw:.2f}  ({n_ben_tr} benign / {n_mal_tr} malware in train)")

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

ensemble = VotingClassifier(
    estimators=[("rf", rf_clf), ("xgb", xgb_clf)],
    voting="soft",
    n_jobs=-1,
)

proposed_pipe = ImbPipeline([
    ("smote",    SMOTE(random_state=RANDOM_STATE)),
    ("scaler",   StandardScaler()),
    ("ensemble", ensemble),
])

print("\n    Fitting proposed pipeline ...")
proposed_pipe.fit(X_train, y_train)

y_pred_prop = proposed_pipe.predict(X_test)
y_prob_prop = proposed_pipe.predict_proba(X_test)[:, 1]

proposed_metrics = evaluate(
    "SMOTE + RF-XGBoost Ensemble",
    y_test, y_pred_prop, y_prob_prop
)

save_confusion_matrix(
    "SMOTE + RF-XGBoost Ensemble",
    y_test, y_pred_prop,
    "cm_proposed_ensemble.png"
)

print("\n    Full classification report:")
print(classification_report(y_test, y_pred_prop,
                             target_names=["Benign", "Malware"]))

# =============================================================================
#  6. CROSS-VALIDATION ON PROPOSED MODEL
# =============================================================================

print("\n[4] Stratified 5-fold cross-validation (proposed model) ...")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cv_results = cross_validate(
    proposed_pipe, X, y,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    return_train_score=False,
    n_jobs=-1,
)

print(f"\n    {'Metric':<12}  {'Mean':>8}  {'Std':>8}")
print(f"    {'─'*32}")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    scores = cv_results[f"test_{metric}"]
    print(f"    {metric:<12}  {scores.mean()*100:>7.2f}%  +/-{scores.std()*100:>5.2f}%")

# =============================================================================
#  7. RESULTS SUMMARY TABLE
# =============================================================================

print("\n\n[5] Summary comparison ...")

all_results = pd.concat(
    [baseline_df, pd.DataFrame([proposed_metrics])],
    ignore_index=True
)
print("\n" + all_results.to_string(index=False))
all_results.to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)
print(f"\n    Saved -> {RESULTS_DIR}/results_summary.csv")

# =============================================================================
#  8. VISUALISATIONS
# =============================================================================

print("\n[6] Generating plots ...")

# 8a. Accuracy comparison
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#B5D4F4"] * len(baseline_defs) + ["#534AB7"]
bars = ax.barh(all_results["Model"], all_results["Accuracy"],
               color=colors, edgecolor="white", height=0.6)
ax.axvline(79.14, color="#A32D2D", linewidth=1.2,
           linestyle="--", label="Palma et al. benchmark (79.14%)")
for bar, val in zip(bars, all_results["Accuracy"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}%", va="center", fontsize=9)
ax.set_xlabel("Accuracy (%)", fontsize=10)
ax.set_title("Accuracy comparison — baseline vs proposed", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 112)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"), dpi=150)
plt.close()

# 8b. Per-metric chart (proposed model)
metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score"]
values_list  = [proposed_metrics[m] for m in metrics_list]

fig, ax = plt.subplots(figsize=(7, 4))
bar_colors = ["#534AB7", "#0F6E56", "#A32D2D", "#185FA5"]
bars = ax.bar(metrics_list, values_list, color=bar_colors,
              width=0.5, edgecolor="white")
for bar, val in zip(bars, values_list):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 115)
ax.set_ylabel("Score (%)", fontsize=10)
ax.set_title("Proposed model — per-metric performance", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "proposed_metrics.png"), dpi=150)
plt.close()

# 8c. Recall comparison
fig, ax = plt.subplots(figsize=(10, 5))
recall_colors = ["#B5D4F4"] * len(baseline_defs) + ["#A32D2D"]
bars = ax.barh(all_results["Model"], all_results["Recall"],
               color=recall_colors, edgecolor="white", height=0.6)
for bar, val in zip(bars, all_results["Recall"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}%", va="center", fontsize=9)
ax.set_xlabel("Recall / Malware Detection Rate (%)", fontsize=10)
ax.set_title("Malware recall — baseline vs proposed", fontsize=11, fontweight="bold")
ax.set_xlim(0, 115)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "recall_comparison.png"), dpi=150)
plt.close()

# 8d. Metrics heatmap
metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
heatmap_data = all_results.set_index("Model")[metrics_cols].astype(float)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Score (%)"})
ax.set_title("All models — metrics heatmap", fontsize=11, fontweight="bold")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "metrics_heatmap.png"), dpi=150)
plt.close()

print(f"    4 plots saved to /{RESULTS_DIR}/")

# =============================================================================
#  9. FINAL VERDICT
# =============================================================================

best_acc = baseline_df["Accuracy"].max()
best_rec = baseline_df["Recall"].max()
prop_acc = proposed_metrics["Accuracy"]
prop_rec = proposed_metrics["Recall"]

def delta(new, old):
    sign = "+ " if new > old else "- "
    return f"{sign}{abs(new - old):.2f}%"

print("\n" + "=" * 65)
print("  FINAL RESULTS")
print("=" * 65)
print(f"  Best baseline accuracy  : {best_acc:.2f}%  (Palma et al. RF: 79.14%)")
print(f"  Proposed accuracy       : {prop_acc:.2f}%  ({delta(prop_acc, best_acc)})")
print(f"\n  Best baseline recall    : {best_rec:.2f}%")
print(f"  Proposed recall         : {prop_rec:.2f}%  ({delta(prop_rec, best_rec)})")
print("=" * 65)
print(f"\n  All outputs saved to : {os.path.abspath(RESULTS_DIR)}")
print("  Done.\n")