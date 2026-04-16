import pandas as pd
import numpy as np
import os

DATA_PATH = r"C:\project\Patient Readmission Risk Prediction + Cost Optimization\data\cleaned_data.csv"
OUT_PATH  = r"C:\project\Patient Readmission Risk Prediction + Cost Optimization\data\diabetic_features.csv"

print("=" * 65)
print("STEP 5 — FEATURE ENGINEERING")
print("=" * 65)

# Load data

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Input file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Input shape: {df.shape}")

# 1. Age group → ordinal numeric

age_map = {
    "[0-10)": 1, "[10-20)": 2, "[20-30)": 3, "[30-40)": 4,
    "[40-50)": 5, "[50-60)": 6, "[60-70)": 7, "[70-80)": 8,
    "[80-90)": 9, "[90-100)": 10
}
if "age" in df.columns:
    df["age_num"] = df["age"].map(age_map)
    print("  ✔  age_num — ordinal age encoding")
else:
    print("  ⚠  'age' column not found, skipping age_num")


# 2. Total prior healthcare utilisation

visit_cols = ["number_outpatient", "number_emergency", "number_inpatient"]
present_visit_cols = [c for c in visit_cols if c in df.columns]
if present_visit_cols:
    df["total_prior_visits"] = df[present_visit_cols].sum(axis=1)
    print(f"  ✔  total_prior_visits = {' + '.join(present_visit_cols)}")
else:
    df["total_prior_visits"] = 0
    print("  ⚠  No visit columns found, total_prior_visits set to 0")


# 3. Chronic condition flags from primary diagnosis (ICD-9)

def safe_float(code):
    try:
        return float(str(code).strip().split(".")[0])
    except (ValueError, TypeError):
        return None

def is_circulatory(code):
    n = safe_float(code)
    if n is None:
        return 0
    return int((390 <= n <= 459) or n == 785)

def is_respiratory(code):
    n = safe_float(code)
    if n is None:
        return 0
    return int((460 <= n <= 519) or n == 786)

def is_diabetes_diag(code):
    n = safe_float(code)
    if n is None:
        return 0
    return int(250 <= n <= 250.99)

if "diag_1" in df.columns:
    df["flag_circulatory"] = df["diag_1"].apply(is_circulatory)
    df["flag_respiratory"] = df["diag_1"].apply(is_respiratory)
    df["flag_diabetes"]    = df["diag_1"].apply(is_diabetes_diag)
    df["flag_chronic"]     = (
        df["flag_circulatory"] | df["flag_respiratory"] | df["flag_diabetes"]
    ).astype(int)
    print("  ✔  Chronic condition flags: circulatory, respiratory, diabetes, chronic")
else:
    print("  ⚠  'diag_1' column not found, skipping chronic flags")


# 4. Medication change indicators

if "change" in df.columns:
    df["med_changed"] = (df["change"] == "Ch").astype(int)
    print("  ✔  med_changed from 'change' column")
else:
    df["med_changed"] = 0
    print("  ⚠  'change' column not found, med_changed set to 0")

med_cols_check = [
    "metformin", "glipizide", "glyburide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "insulin"
]
med_cols_present = [c for c in med_cols_check if c in df.columns]

for col in med_cols_present:
    df[col + "_active"] = (df[col].isin(["Up", "Down", "Steady"])).astype(int)

if med_cols_present:
    df["num_active_meds"] = df[[c + "_active" for c in med_cols_present]].sum(axis=1)
    print(f"  ✔  num_active_meds from {len(med_cols_present)} medication columns")
else:
    df["num_active_meds"] = 0
    print("  ⚠  No medication columns found, num_active_meds set to 0")


# 5. Diagnosis category (broad ICD-9 bucketing)

def map_diag_category(code):
    try:
        c = str(code).strip()
        if c.upper().startswith("V") or c.upper().startswith("E"):
            return 9   # External/Other
        n = float(c.split(".")[0])
    except (ValueError, TypeError):
        return 0
    if (390 <= n <= 459) or n == 785: return 1   # Circulatory
    if (460 <= n <= 519) or n == 786: return 2   # Respiratory
    if (520 <= n <= 579) or n == 787: return 3   # Digestive
    if 250 <= n <= 250.99:            return 4   # Diabetes
    if 800 <= n <= 999:               return 5   # Injury
    if 710 <= n <= 739:               return 6   # Musculoskeletal
    if (580 <= n <= 629) or n == 788: return 7   # Genitourinary
    if 140 <= n <= 239:               return 8   # Neoplasms
    return 0                                     # Other

for col in ["diag_1", "diag_2", "diag_3"]:
    if col in df.columns:
        df[col + "_cat"] = df[col].apply(map_diag_category)
print("  ✔  diag_1_cat, diag_2_cat, diag_3_cat — ICD-9 category codes")

# 6. High-frequency specialties (top-10 + 'Other')

if "medical_specialty" in df.columns:
    top_spec = df["medical_specialty"].value_counts().head(10).index
    df["specialty_grouped"] = df["medical_specialty"].apply(
        lambda x: x if x in top_spec else "Other"
    )
    spec_dummies = pd.get_dummies(df["specialty_grouped"], prefix="spec", drop_first=True)
    # Convert bool columns to int for consistency
    spec_dummies = spec_dummies.astype(int)
    df = pd.concat([df, spec_dummies], axis=1)
    df.drop(columns=["specialty_grouped", "medical_specialty"], inplace=True)
    print(f"  ✔  medical_specialty → {spec_dummies.shape[1]} one-hot columns")
else:
    print("  ⚠  'medical_specialty' column not found, skipping")


# 7. Encode remaining binary/low-cardinality categoricals

if "gender" in df.columns:
    df["gender_enc"] = (df["gender"] == "Male").astype(int)
    print("  ✔  gender_enc encoded")

if "diabetesMed" in df.columns:
    df["diabetesMed_enc"] = (df["diabetesMed"] == "Yes").astype(int)
    print("  ✔  diabetesMed_enc encoded")

if "A1Cresult" in df.columns:
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
    df["A1Cresult_enc"] = df["A1Cresult"].map(a1c_map).fillna(0).astype(int)
    print("  ✔  A1Cresult_enc encoded")

if "max_glu_serum" in df.columns:
    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    df["max_glu_serum_enc"] = df["max_glu_serum"].map(glu_map).fillna(0).astype(int)
    print("  ✔  max_glu_serum_enc encoded")

# 8. Encode race (one-hot)

if "race" in df.columns:
    race_dummies = pd.get_dummies(df["race"], prefix="race", drop_first=True)
    race_dummies = race_dummies.astype(int)
    df = pd.concat([df, race_dummies], axis=1)
    df.drop(columns=["race"], inplace=True)
    print(f"  ✔  race → {race_dummies.shape[1]} one-hot columns")
else:
    print("  ⚠  'race' column not found, skipping")



# 9. Drop original raw columns that have been encoded/replaced

drop_cols = [
    "age", "gender", "diabetesMed", "A1Cresult", "max_glu_serum",
    "change", "diag_1", "diag_2", "diag_3",
    "payer_code",
    "metformin", "glipizide", "glyburide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "insulin",
]
# Add _active intermediate columns to drop list
for c in med_cols_present:
    drop_cols.append(c + "_active")

# Only drop columns that actually exist
drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True)
print(f"  ✔  Dropped {len(drop_cols)} raw/redundant columns")


# 10. Final type cleanup — ensure no object columns remain

obj_cols = df.select_dtypes(include="object").columns.tolist()
if obj_cols:
    print(f"\n  ⚠  Remaining object columns (will be label-encoded): {obj_cols}")
    for col in obj_cols:
        df[col] = pd.Categorical(df[col]).codes
    print("  ✔  Label-encoded remaining object columns")
else:
    print("  ✔  No remaining object columns — all features are numeric")


# 11. Summary

print(f"\n  Output shape : {df.shape}")
print(f"  Columns      : {df.columns.tolist()}")
print(f"  Null counts  :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\n  Sample:\n{df.head(3).to_string()}")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"\n✅  Step 5 Complete — Feature-engineered dataset saved → '{OUT_PATH}'\n")