import pandas as pd
import numpy as np

# ── Configuration 
DATA_PATH = "data/diabetic_data.csv"

# ── Load 
print("=" * 65)
print("STEP 2 — DATA LOADING & EXPLORATION")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"\n✅  Dataset loaded successfully from '{DATA_PATH}'")

# ── Basic Shape 
print("\n" + "─" * 65)
print("DATASET SHAPE")
print("─" * 65)
print(f"  Rows (encounters)  : {df.shape[0]:,}")
print(f"  Columns (features) : {df.shape[1]}")

# ── Column Names
print("\n" + "─" * 65)
print("COLUMN NAMES")
print("─" * 65)
for i, col in enumerate(df.columns, 1):
    print(f"  {i:>2}. {col}")

# ── Data Types 
print("\n" + "─" * 65)
print("DATA TYPES")
print("─" * 65)
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ["Column", "Dtype"]
print(dtype_df.to_string(index=False))

# ── Missing Values 
print("\n" + "─" * 65)
print("MISSING / UNKNOWN VALUES  ('?' sentinel)")
print("─" * 65)
# Replace '?' with NaN temporarily for counting
df_temp = df.replace("?", np.nan)
missing = df_temp.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct.round(2)
}).query("`Missing Count` > 0").sort_values("Missing %", ascending=False)

if missing_df.empty:
    print("  No NaN missing values detected (? sentinels may still exist).")
else:
    print(missing_df.to_string())

# ── Target Variable Distribution
print("\n" + "─" * 65)
print("TARGET VARIABLE  →  'readmitted'")
print("─" * 65)
vc = df["readmitted"].value_counts()
for label, count in vc.items():
    pct = count / len(df) * 100
    print(f"  {label:<6} : {count:>7,}  ({pct:.1f}%)")

# ── Numeric Summary 
print("\n" + "─" * 65)
print("SUMMARY STATISTICS — NUMERIC COLUMNS")
print("─" * 65)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[num_cols].describe().round(2).to_string())

# ── Categorical Columns Preview 
print("\n" + "─" * 65)
print("CATEGORICAL COLUMNS — UNIQUE VALUE COUNTS")
print("─" * 65)
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    n_unique = df[col].nunique()
    sample_vals = df[col].unique()[:5].tolist()
    print(f"  {col:<35} | unique={n_unique:>4} | sample: {sample_vals}")

# ── Duplicate Check 
print("\n" + "─" * 65)
print("DUPLICATE ROWS")
print("─" * 65)
n_dups = df.duplicated().sum()
print(f"  Exact duplicate rows: {n_dups:,}")

print("\n✅  Step 2 Complete — Data loaded and explored.\n")