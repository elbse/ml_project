# =============================================================================
#  CICMalAnal2017 — Dataset Merger & Preprocessor (Memory-Efficient Version)
#  Reads CSVs category by category, samples to match paper distribution,
#  aligns columns, and saves final dataset_cicandmal2017.csv
# =============================================================================

import matplotlib
matplotlib.use("Agg")

import os
import pandas as pd
import numpy as np

# =============================================================================
#  CONFIGURATION
# =============================================================================

BASE_DIR    = "CICMalAnal2017"
OUTPUT_FILE = "dataset_cicandmal2017.csv"

# Target sample counts matching Palma et al. (2024)
# Benign: 5065, Malware total: 426 split across 4 categories
TARGET_BENIGN  = 5065
TARGET_MALWARE = 426   # total across all malware categories

CATEGORY_MAP = {
    os.path.join(BASE_DIR, "Benign-CSVs",     "Benign"):     0,
    os.path.join(BASE_DIR, "Adware-CSVs",     "Adware"):     1,
    os.path.join(BASE_DIR, "Ransomware-CSVs", "Ransomware"): 1,
    os.path.join(BASE_DIR, "Scareware-CSVs",  "Scareware"):  1,
    os.path.join(BASE_DIR, "SMSmalware-CSVs", "SMSmalware"): 1,
}

# Metadata columns to drop
DROP_META = ["Flow ID", "Source IP", "Destination IP", "Timestamp",
             "src_ip", "dst_ip", "Flow.ID", "Source.IP",
             "Destination.IP", "Category"]

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def find_all_csvs(folder):
    csv_paths = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".csv"):
                csv_paths.append(os.path.join(root, f))
    return csv_paths

def load_category(folder_path, label, max_rows=None):
    """Load all CSVs from a category folder, optionally sampling max_rows."""
    csv_files = find_all_csvs(folder_path)
    if not csv_files:
        print(f"    WARNING: No CSVs found in {folder_path}")
        return None

    dfs = []
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, low_memory=False)
            # Drop metadata columns immediately to save memory
            drop_cols = [c for c in DROP_META if c in df.columns]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
            # Coerce to numeric immediately
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            dfs.append(df)
        except Exception as e:
            print(f"    Skipping {os.path.basename(fpath)}: {e}")

    if not dfs:
        return None

    cat_df = pd.concat(dfs, ignore_index=True)

    # Sample if needed
    if max_rows and len(cat_df) > max_rows:
        cat_df = cat_df.sample(n=max_rows, random_state=RANDOM_STATE)
        print(f"    Sampled to {max_rows} rows")

    cat_df["Label"] = label
    return cat_df

# =============================================================================
#  1. LOAD BENIGN
# =============================================================================

print("=" * 65)
print("  CICMalAnal2017 — Dataset Merger (Memory-Efficient)")
print("=" * 65)

print(f"\n[1] Loading Benign (target: {TARGET_BENIGN} rows) ...")
benign_path = os.path.join(BASE_DIR, "Benign-CSVs", "Benign")
benign_df   = load_category(benign_path, label=0, max_rows=TARGET_BENIGN)

if benign_df is None:
    print("ERROR: Could not load Benign data. Check folder path.")
    exit(1)

print(f"    Benign rows loaded : {len(benign_df)}")
print(f"    Benign columns     : {len(benign_df.columns)}")
benign_cols = set(benign_df.columns) - {"Label"}

# =============================================================================
#  2. LOAD MALWARE CATEGORIES
# =============================================================================

print(f"\n[2] Loading Malware categories (target total: {TARGET_MALWARE} rows) ...")

malware_cats = {
    "Adware":     os.path.join(BASE_DIR, "Adware-CSVs",     "Adware"),
    "Ransomware": os.path.join(BASE_DIR, "Ransomware-CSVs", "Ransomware"),
    "Scareware":  os.path.join(BASE_DIR, "Scareware-CSVs",  "Scareware"),
    "SMSmalware": os.path.join(BASE_DIR, "SMSmalware-CSVs", "SMSmalware"),
}

# Split target evenly across 4 categories
per_cat = TARGET_MALWARE // len(malware_cats)
remainder = TARGET_MALWARE % len(malware_cats)

malware_dfs = []
for i, (cat_name, cat_path) in enumerate(malware_cats.items()):
    rows = per_cat + (1 if i < remainder else 0)
    print(f"\n  Loading {cat_name} (target: {rows} rows) ...")
    df = load_category(cat_path, label=1, max_rows=rows)
    if df is not None:
        print(f"    {cat_name} rows loaded : {len(df)}")
        malware_dfs.append(df)

if not malware_dfs:
    print("ERROR: Could not load any malware data.")
    exit(1)

malware_df = pd.concat(malware_dfs, ignore_index=True)
print(f"\n  Total malware rows : {len(malware_df)}")

# =============================================================================
#  3. ALIGN COLUMNS — keep only columns common to benign AND malware
# =============================================================================

print("\n[3] Aligning columns ...")

malware_cols = set(malware_df.columns) - {"Label"}
common_cols  = list(benign_cols & malware_cols)
common_cols.sort()

print(f"    Benign columns   : {len(benign_cols)}")
print(f"    Malware columns  : {len(malware_cols)}")
print(f"    Common columns   : {len(common_cols)}")

benign_df  = benign_df[common_cols + ["Label"]]
malware_df = malware_df[common_cols + ["Label"]]

# =============================================================================
#  4. COMBINE
# =============================================================================

print("\n[4] Combining benign and malware ...")
combined = pd.concat([benign_df, malware_df], ignore_index=True)
print(f"    Combined shape : {combined.shape}")

# Free memory
del benign_df, malware_df, malware_dfs
import gc; gc.collect()

# =============================================================================
#  5. CLEAN
# =============================================================================

print("\n[5] Cleaning ...")
combined.replace([np.inf, -np.inf], np.nan, inplace=True)

missing_pct = combined.isnull().mean().mean() * 100
print(f"    Missing value % : {missing_pct:.2f}%")

numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
combined[numeric_cols] = combined[numeric_cols].fillna(
    combined[numeric_cols].median()
)

before = len(combined)
combined.drop_duplicates(inplace=True)
print(f"    Duplicates removed : {before - len(combined)}")

# Remove zero-variance columns
from sklearn.feature_selection import VarianceThreshold
label_col = combined["Label"].copy()
X_tmp = combined.drop(columns=["Label"])
vt = VarianceThreshold(threshold=0.0)
X_tmp = pd.DataFrame(vt.fit_transform(X_tmp),
                     columns=X_tmp.columns[vt.get_support()])
combined = X_tmp.copy()
combined["Label"] = label_col.values
print(f"    After zero-var removal : {combined.shape}")

# =============================================================================
#  6. CLASS DISTRIBUTION
# =============================================================================

print("\n[6] Class distribution ...")
vc = combined["Label"].value_counts().sort_index()
for cls, cnt in vc.items():
    label_str = "Benign" if cls == 0 else "Malware"
    print(f"    {label_str} ({cls}) : {cnt}  ({cnt/len(combined)*100:.1f}%)")

n_benign  = int(vc.get(0, 0))
n_malware = int(vc.get(1, 0))
ratio     = n_benign / n_malware if n_malware > 0 else 0
print(f"    Imbalance ratio : {ratio:.1f}:1  (benign:malware)")

if ratio >= 5:
    print("    ✓ Imbalance confirmed — SMOTE-Tomek justified!")
else:
    print("    ⚠ Lower than expected — check if all files loaded.")

# =============================================================================
#  7. CLEAN COLUMN NAMES
# =============================================================================

print("\n[7] Cleaning column names ...")
combined.columns = (
    combined.columns
    .str.strip()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.replace(r"\s+", "_", regex=True)
)
print(f"    Sample columns : {list(combined.columns[:5])}")
print(f"    Target column  : Label")

# =============================================================================
#  8. SAVE
# =============================================================================

print(f"\n[8] Saving to {OUTPUT_FILE} ...")
combined.to_csv(OUTPUT_FILE, index=False)
print(f"    Saved {len(combined)} rows x {len(combined.columns)} columns")
print(f"    File : {os.path.abspath(OUTPUT_FILE)}")

# =============================================================================
#  SUMMARY
# =============================================================================

print("\n\n" + "=" * 65)
print("  MERGE COMPLETE")
print("=" * 65)
print(f"""
  Output file    : {OUTPUT_FILE}
  Total samples  : {len(combined)}
  Total features : {len(combined.columns) - 1}
  Benign         : {n_benign}
  Malware        : {n_malware}
  Imbalance      : {ratio:.1f}:1

  Next steps:
    Update DATASET_PATH in all phase scripts:
      DATASET_PATH = "{OUTPUT_FILE}"
      TARGET_COL   = "Label"
      DROP_COLS    = []
""")
print("=" * 65)
print("  Done.\n")