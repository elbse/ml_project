# =============================================================================
#  CICMalAnal2017 — Dataset Merger & Preprocessor
#  Merges all category CSVs into one clean dataset_cicandmal2017.csv
#  Structure expected:
#    CICMalAnal2017/
#      Benign-CSVs/Benign/       -> Label 0
#      Adware-CSVs/Adware/       -> Label 1
#      Ransomware-CSVs/Ransomware/ -> Label 1
#      Scareware-CSVs/Scareware/ -> Label 1
#      SMSmalware-CSVs/SMSmalware/ -> Label 1
# =============================================================================

import os
import pandas as pd
import numpy as np

# =============================================================================
#  CONFIGURATION — update BASE_DIR if your folder is in a different location
# =============================================================================

BASE_DIR    = "CICMalAnal2017"       # folder in your project root
OUTPUT_FILE = "dataset_cicandmal2017.csv"

# Map each subfolder path to its label (0 = Benign, 1 = Malware)
CATEGORY_MAP = {
    os.path.join(BASE_DIR, "Benign-CSVs",     "Benign"):     0,
    os.path.join(BASE_DIR, "Adware-CSVs",     "Adware"):     1,
    os.path.join(BASE_DIR, "Ransomware-CSVs", "Ransomware"): 1,
    os.path.join(BASE_DIR, "Scareware-CSVs",  "Scareware"):  1,
    os.path.join(BASE_DIR, "SMSmalware-CSVs", "SMSmalware"): 1,
}

def find_all_csvs(folder):
    """Recursively find all CSV files under a folder."""
    csv_paths = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".csv"):
                csv_paths.append(os.path.join(root, f))
    return csv_paths

# =============================================================================
#  1. LOAD AND MERGE
# =============================================================================

print("=" * 65)
print("  CICMalAnal2017 — Dataset Merger")
print("=" * 65)

all_dfs = []

for folder_path, label in CATEGORY_MAP.items():
    category = os.path.basename(folder_path)
    label_str = "Benign" if label == 0 else "Malware"

    if not os.path.exists(folder_path):
        print(f"\n  WARNING: Folder not found — {folder_path}")
        print(f"           Check that BASE_DIR is correct.")
        continue

    csv_files = [f for f in os.listdir(folder_path)
                 if f.endswith(".csv")]

    if not csv_files:
        print(f"\n  WARNING: No CSV files found in {folder_path}")
        continue

    print(f"\n  Loading {category} ({label_str}) — {len(csv_files)} file(s) ...")

    cat_dfs = []
    for fname in csv_files:
        fpath = os.path.join(folder_path, fname)
        try:
            df = pd.read_csv(fpath, low_memory=False)
            cat_dfs.append(df)
        except Exception as e:
            print(f"    Skipping {fname}: {e}")

    if not cat_dfs:
        continue

    cat_df = pd.concat(cat_dfs, ignore_index=True)
    cat_df["Label"] = label
    cat_df["Category"] = category
    all_dfs.append(cat_df)

    print(f"    Rows loaded : {len(cat_df)}")
    print(f"    Columns     : {len(cat_df.columns)}")

# =============================================================================
#  2. COMBINE ALL CATEGORIES
# =============================================================================

print("\n\n[2] Combining all categories ...")
combined = pd.concat(all_dfs, ignore_index=True)
print(f"    Combined shape : {combined.shape}")

# =============================================================================
#  3. ALIGN COLUMNS
#  Different category CSVs may have slightly different columns
#  Keep only columns common to ALL categories
# =============================================================================

print("\n[3] Aligning columns across categories ...")

# Identify columns to drop (non-feature metadata)
drop_candidates = ["Category", "Flow ID", "Source IP", "Destination IP",
                   "Timestamp", "src_ip", "dst_ip", "Flow.ID",
                   "Source.IP", "Destination.IP"]

# Drop metadata columns if present
meta_dropped = [c for c in drop_candidates if c in combined.columns]
if meta_dropped:
    combined.drop(columns=meta_dropped, inplace=True)
    print(f"    Dropped metadata columns: {meta_dropped}")

# Keep Label column separate
label_col = combined["Label"].copy()
category_col = combined.get("Category", None)

# Select only numeric feature columns (exclude Label)
feature_cols = [c for c in combined.columns
                if c not in ["Label", "Category"]]
X = combined[feature_cols].copy()

# Coerce all feature columns to numeric
print("    Coercing all features to numeric ...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Reassemble
combined = X.copy()
combined["Label"] = label_col.values

print(f"    Final shape    : {combined.shape}")

# =============================================================================
#  4. BASIC CLEANING
# =============================================================================

print("\n[4] Cleaning dataset ...")

# Replace infinite values with NaN
combined.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"    Infinite values replaced with NaN")

# Report missing values before filling
missing_pct = combined.isnull().mean().mean() * 100
print(f"    Missing value % (before fill): {missing_pct:.2f}%")

# Fill missing values with column medians
numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
combined[numeric_cols] = combined[numeric_cols].fillna(
    combined[numeric_cols].median()
)

# Remove duplicate rows
before = len(combined)
combined.drop_duplicates(inplace=True)
print(f"    Duplicates removed : {before - len(combined)}")

print(f"    Final shape        : {combined.shape}")

# =============================================================================
#  5. CLASS DISTRIBUTION
# =============================================================================

print("\n[5] Class distribution ...")
vc = combined["Label"].value_counts().sort_index()
for cls, cnt in vc.items():
    label_str = "Benign" if cls == 0 else "Malware"
    print(f"    {label_str} ({cls}) : {cnt}  ({cnt/len(combined)*100:.1f}%)")

n_benign  = int(vc.get(0, 0))
n_malware = int(vc.get(1, 0))
ratio     = n_benign / n_malware if n_malware > 0 else 0
print(f"    Imbalance ratio : {ratio:.1f}:1  (benign:malware)")
print(f"\n    Expected from Palma et al.: ~12:1 (5,065 benign : 426 malware)")

if ratio >= 8:
    print("    ✓ Imbalance ratio matches expected range — correct dataset!")
else:
    print("    ⚠ Ratio differs from expected. Check if all folders loaded.")

# =============================================================================
#  6. COLUMN NAME CLEANUP
# =============================================================================

print("\n[6] Cleaning column names ...")

# Strip whitespace and special characters from column names
combined.columns = (
    combined.columns
    .str.strip()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.replace(r"\s+", "_", regex=True)
)

print(f"    Sample columns: {list(combined.columns[:5])}")
print(f"    Target column : {combined.columns[-1]}")

# =============================================================================
#  7. SAVE
# =============================================================================

print(f"\n[7] Saving to {OUTPUT_FILE} ...")
combined.to_csv(OUTPUT_FILE, index=False)
print(f"    Saved {len(combined)} rows × {len(combined.columns)} columns")
print(f"    File: {os.path.abspath(OUTPUT_FILE)}")

# =============================================================================
#  8. SUMMARY
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

  Next step:
    Update DATASET_PATH in phase1_baseline.py:
      DATASET_PATH = "{OUTPUT_FILE}"
      TARGET_COL   = "Label"
      DROP_COLS    = []
""")
print("=" * 65)
print("  Done.\n")