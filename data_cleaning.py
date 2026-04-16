import os
import numpy as np
import pandas as pd

# ─── path configuration
BASE_DIR   = r"C:\project\Patient Readmission Risk Prediction + Cost Optimization\data"
INPUT_CSV  = os.path.join(BASE_DIR, "diabetic_data.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "cleaned_data.csv")

DIVIDER = "\n" + "=" * 70 + "\n"

def section(title):
    print(f"{DIVIDER}STEP: {title}{DIVIDER}")


# STEP 1 — Load the dataset and display basic information

section("1 · Load dataset and display basic information")

df = pd.read_csv(INPUT_CSV, low_memory=False, dtype=str)  # FIX: load all as str to avoid mixed-type warnings

print(f"Dataset shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumn names ({len(df.columns)} total):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:>2}. {col}")

print("\nData types per column:")
print(df.dtypes.to_string())

print("\nFirst 3 rows (transposed for readability):")
print(df.head(3).T.to_string())

# STEP 2 — Identify and replace invalid entries ("?") with NaN

section("2 · Replace '?' with NaN and report missing values")

df = df.replace("?", np.nan)

missing_counts = df.isnull().sum()
missing_pct    = (missing_counts / len(df) * 100).round(2)
missing_report = (
    pd.DataFrame({"missing_count": missing_counts, "missing_%": missing_pct})
      .query("missing_count > 0")
      .sort_values("missing_%", ascending=False)
)
print("Columns with missing values after '?' → NaN replacement:")
print(missing_report.to_string())

# STEP 3 — Handle missing values

section("3 · Handle missing values")

MISSING_THRESHOLD = 40.0

# 3a — drop high-missingness columns
cols_to_drop = missing_report[missing_report["missing_%"] > MISSING_THRESHOLD].index.tolist()
print(f"Columns with > {MISSING_THRESHOLD}% missing → dropping {len(cols_to_drop)} column(s):")
for col in cols_to_drop:
    print(f"  • {col}  ({missing_pct[col]:.1f}% missing)")
df = df.drop(columns=cols_to_drop)

# 3b — convert known numeric columns explicitly (since we loaded all as str)
numeric_col_names = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
for col in numeric_col_names:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 3c — identify column types after conversion
numerical_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"\nNumerical columns   ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# 3d — fill numerical NaN with median
num_filled = []
for col in numerical_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        num_filled.append((col, median_val))
if num_filled:
    print("\nNumerical columns filled with median:")
    for col, val in num_filled:
        print(f"  • {col} → median = {val}")
else:
    print("\nNo numerical columns required median imputation.")

# 3e — fill categorical NaN with mode
cat_filled = []
for col in categorical_cols:
    if df[col].isnull().any():
        mode_series = df[col].mode(dropna=True)
        if not mode_series.empty:
            df[col] = df[col].fillna(mode_series.iloc[0])
            cat_filled.append((col, mode_series.iloc[0]))
if cat_filled:
    print("\nCategorical columns filled with mode:")
    for col, val in cat_filled:
        print(f"  • {col} → mode = '{val}'")
else:
    print("\nNo categorical columns required mode imputation.")


# STEP 4 — Remove duplicate rows

section("4 · Remove duplicate rows")

n_before = len(df)
df = df.drop_duplicates()
n_removed = n_before - len(df)
print(f"Rows before deduplication : {n_before:,}")
print(f"Duplicate rows removed     : {n_removed:,}")
print(f"Rows after deduplication  : {len(df):,}")

# STEP 5 — Drop irrelevant identifier columns

section("5 · Drop irrelevant columns (identifiers)")

irrelevant_cols    = ["encounter_id", "patient_nbr"]
irrelevant_present = [c for c in irrelevant_cols if c in df.columns]
df = df.drop(columns=irrelevant_present)
print(f"Dropped columns: {irrelevant_present}")
print(f"Dataset shape after dropping identifiers: {df.shape}")

# STEP 6 — Clean and standardise categorical columns

section("6 · Standardise categorical features")

# 6a — gender
if "gender" in df.columns:
    print("gender — unique values before:", df["gender"].unique().tolist())
    df["gender"] = df["gender"].replace("Unknown/Invalid", np.nan)
    df["gender"] = df["gender"].fillna(df["gender"].mode(dropna=True).iloc[0])
    print("gender — unique values after :", df["gender"].unique().tolist())

# 6b — race
if "race" in df.columns:
    print("\nrace — value counts before cleaning:")
    print(df["race"].value_counts(dropna=False).to_string())
    df["race"] = df["race"].fillna("Unknown")
    print("race — unique values after  :", df["race"].unique().tolist())

# 6c — age: bracket → numeric midpoint
if "age" in df.columns:
    age_midpoint_map = {
        "[0-10)": 5,  "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    df["age"] = df["age"].map(age_midpoint_map)
    print("\nage — midpoint value counts:")
    print(df["age"].value_counts().sort_index().to_string())

# 6d — readmitted: encode as binary target
if "readmitted" in df.columns:
    print("\nreadmitted — original value counts:")
    print(df["readmitted"].value_counts().to_string())
    readmitted_map = {"NO": 0, "<30": 1, ">30": 0}
    df["readmitted"] = df["readmitted"].map(readmitted_map)
    # FIX: map() produces NaN for unrecognised values — fill with 0 as safe default
    df["readmitted"] = df["readmitted"].fillna(0).astype(int)
    print("readmitted — after binary encoding:")
    print(df["readmitted"].value_counts().to_string())

# 6e — diag_1 / diag_2 / diag_3: sanitise ICD-9 codes
for diag_col in ["diag_1", "diag_2", "diag_3"]:
    if diag_col in df.columns:
        df[diag_col] = df[diag_col].astype(str).str.strip()
        df[diag_col] = df[diag_col].replace("nan", "Unknown")
        print(f"\n{diag_col} — sample values: {df[diag_col].unique()[:6].tolist()}")

# 6f — medication columns: strip whitespace
med_cols = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
    "change", "diabetesMed",
]
med_cols_present = [c for c in med_cols if c in df.columns]
for col in med_cols_present:
    df[col] = df[col].astype(str).str.strip()
if "insulin" in df.columns:
    print(f"\nMedication columns ({len(med_cols_present)}) stripped. "
          f"Sample 'insulin' values: {df['insulin'].unique().tolist()}")


# STEP 7 — Handle outliers in numerical columns (IQR capping)

section("7 · Handle outliers using IQR capping (Winsorisation)")

cols_for_outlier = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in ("readmitted", "age")
]

outlier_summary = []
for col in cols_for_outlier:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
    if n_outliers > 0:
        df[col] = df[col].clip(lower=lower, upper=upper)
        outlier_summary.append({
            "column"   : col,
            "outliers" : n_outliers,
            "lower_cap": round(lower, 2),
            "upper_cap": round(upper, 2),
        })

if outlier_summary:
    print("Columns where outliers were capped:")
    print(pd.DataFrame(outlier_summary).to_string(index=False))
else:
    print("No outliers found beyond IQR fences.")


# STEP 8 — Convert data types

section("8 · Convert data types")

# 8a — cast numerical columns to int64
int_cols = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
int_cols_present = [c for c in int_cols if c in df.columns]
for col in int_cols_present:
    df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("int64")
print(f"Integer columns cast to int64: {int_cols_present}")

# Ensure age and readmitted are int64
for col in ("age", "readmitted"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype("int64")  # FIX: fillna before cast

# 8b — convert string columns to category dtype (skip high-cardinality diag cols)
skip_category = {"diag_1", "diag_2", "diag_3"}
cat_dtype_cols = [
    c for c in df.select_dtypes(include=["object"]).columns  # FIX: removed "str" — not a valid pandas dtype alias
    if c not in skip_category
]
for col in cat_dtype_cols:
    df[col] = df[col].astype("category")
print(f"\nCategory dtype applied to {len(cat_dtype_cols)} columns: {cat_dtype_cols}")

print("\nUpdated dtypes:")
print(df.dtypes.to_string())

# STEP 9 — Standardise column names

section("9 · Standardise column names")

original_names = df.columns.tolist()
df.columns = (
    df.columns
      .str.lower()
      .str.strip()
      .str.replace(r"[\s\-]+", "_", regex=True)
      .str.replace(r"[^\w]", "", regex=True)
)
renamed = {old: new for old, new in zip(original_names, df.columns) if old != new}
if renamed:
    print(f"Renamed {len(renamed)} column(s):")
    for old, new in renamed.items():
        print(f"  '{old}'  →  '{new}'")
else:
    print("All column names already conform to the standard; no changes needed.")
print(f"\nFinal column names:\n{df.columns.tolist()}")

# STEP 10 — Verify the cleaned dataset

section("10 · Verify the cleaned dataset")

remaining_missing = df.isnull().sum()
total_missing = int(remaining_missing.sum())
print(f"Total remaining missing values: {total_missing}")
if total_missing > 0:
    print("Breakdown:")
    print(remaining_missing[remaining_missing > 0].to_string())
else:
    print("✓  No missing values remain — dataset is complete.")

print(f"\nFinal dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

print("\nDescriptive statistics (numerical columns):")
print(df.describe(include=[np.number]).T.to_string())

print("\nTarget column 'readmitted' distribution:")
if "readmitted" in df.columns:
    vc = df["readmitted"].value_counts()
    for label, count in vc.items():
        pct = count / len(df) * 100
        tag = "readmitted <30 days" if label == 1 else "not readmitted / >30 days"
        print(f"  {label} ({tag}): {count:,}  ({pct:.1f}%)")

mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
print(f"\nMemory usage of cleaned dataframe: {mem_mb:.1f} MB")

# STEP 11 — Save the cleaned dataset

section("11 · Save cleaned dataset to CSV")

df_to_save = df.copy()
for col in df_to_save.select_dtypes(include="category").columns:
    df_to_save[col] = df_to_save[col].astype(str)

df_to_save.to_csv(OUTPUT_CSV, index=False)
saved_size_mb = os.path.getsize(OUTPUT_CSV) / (1024 ** 2)

print(f"Cleaned dataset saved to:\n  {OUTPUT_CSV}")
print(f"File size : {saved_size_mb:.2f} MB")
print(f"Rows      : {len(df_to_save):,}")
print(f"Columns   : {df_to_save.shape[1]}")
print(DIVIDER + "Data cleaning complete!\n")