import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths 
BASE_DIR  = r"C:\project\Patient Readmission Risk Prediction + Cost Optimization"
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_PATH  = os.path.join(DATA_DIR, "diabetic_dashboard.csv")

print("=" * 65)
print("GENERATE DASHBOARD DATA")
print("=" * 65)

# ── Guard: check required files 
required = {
    "X_train"       : os.path.join(DATA_DIR,  "X_train.csv"),
    "X_test"        : os.path.join(DATA_DIR,  "X_test.csv"),
    "y_train"       : os.path.join(DATA_DIR,  "y_train.csv"),
    "y_test"        : os.path.join(DATA_DIR,  "y_test.csv"),
    "Random Forest" : os.path.join(MODEL_DIR, "random_forest.pkl"),
    "Feature names" : os.path.join(MODEL_DIR, "feature_names.pkl"),
    "Raw data"      : os.path.join(DATA_DIR,  "diabetic_data.csv"),
}
missing = [(lbl, p) for lbl, p in required.items() if not os.path.exists(p)]
if missing:
    print("\n  ERROR — Missing files:")
    for lbl, p in missing:
        print(f"    [{lbl}]  {p}")
    raise SystemExit(1)

# ── Load train/test splits 
print("\n  Loading train/test splits ...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"), low_memory=False)
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"),  low_memory=False)
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))

def extract_y(raw):
    if isinstance(raw, pd.DataFrame):
        if "readmitted_30d" in raw.columns:
            return raw["readmitted_30d"]
        return raw.iloc[:, -1]
    return raw

y_train = extract_y(y_train).astype(int).reset_index(drop=True)
y_test  = extract_y(y_test ).astype(int).reset_index(drop=True)

X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train, y_test],  ignore_index=True)

print(f"  X_all shape : {X_all.shape}")
print(f"  y_all dist  : {y_all.value_counts().sort_index().to_dict()}")

# ── Load model and feature names 
with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
    rf = pickle.load(f)
with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

print(f"  ✔  Model loaded  — expects {rf.n_features_in_} features")
print(f"  ✔  Feature names — {len(feature_names)} columns")

# ── Align X_all to model features 
X_all = X_all.fillna(X_all.median(numeric_only=True))
X_all = X_all.astype(float)

for col in feature_names:
    if col not in X_all.columns:
        X_all[col] = 0.0
extra = [c for c in X_all.columns if c not in feature_names]
if extra:
    X_all = X_all.drop(columns=extra)
X_all = X_all[feature_names]

print(f"  ✔  Features aligned: {X_all.shape[1]} columns")

# ── Generate risk scores 
print("\n  Generating risk scores ...")
risk_prob = rf.predict_proba(X_all.values)[:, 1]

P50 = float(np.percentile(risk_prob, 33))
P85 = float(np.percentile(risk_prob, 66))

print(f"  ✔  Scores generated for {len(risk_prob):,} patients")
print(f"  P33 (medium threshold) : {P50:.4f}")
print(f"  P66 (high threshold)   : {P85:.4f}")
print(f"  Score min/max/mean     : {risk_prob.min():.4f} / {risk_prob.max():.4f} / {risk_prob.mean():.4f}")

def assign_tier(p):
    if p >= P85:   return "High"
    elif p >= P50: return "Medium"
    else:          return "Low"

risk_tier = [assign_tier(p) for p in risk_prob]

# ── Load raw data for display columns 
print("\n  Loading raw data for display columns ...")
df_raw = pd.read_csv(os.path.join(DATA_DIR, "diabetic_data.csv"), low_memory=False)
df_raw = df_raw.reset_index(drop=True)
print(f"  Raw data shape: {df_raw.shape}")

# ── Extract display columns safely 
def safe_col(df, col, default, n):
    if col in df.columns:
        vals = df[col].values
        return vals[:n] if len(vals) >= n else list(vals) + [default] * (n - len(vals))
    return [default] * n

n = len(y_all)

age_group  = safe_col(df_raw, "age",              "[60-70)", n)
gender_col = safe_col(df_raw, "gender",            "Unknown", n)
los_col    = safe_col(df_raw, "time_in_hospital",  5,         n)
n_diag_col = safe_col(df_raw, "number_diagnoses",  7,         n)

# ── Diagnosis category from engineered features 
diag_cat_map = {
    0: "Other",        1: "Circulatory",   2: "Respiratory",
    3: "Digestive",    4: "Diabetes",      5: "Injury",
    6: "Musculoskeletal", 7: "Genitourinary",
    8: "Neoplasms",    9: "External/Other"
}
if "diag_1_cat" in X_all.columns:
    diag_cat_col = X_all["diag_1_cat"].map(diag_cat_map).fillna("Other").values
else:
    diag_cat_col = ["Other"] * n

# ── Build dashboard dataframe 
COST_READMISSION_INR = 1_250_000  # ~$15,000 USD equivalent in INR

df_dash = pd.DataFrame({
    "readmitted_30d"      : y_all.values,
    "risk_score"          : np.round(risk_prob, 4),
    "risk_tier"           : risk_tier,
    "age_group"           : age_group,
    "gender"              : gender_col,
    "diag_category"       : diag_cat_col,
    "time_in_hospital"    : los_col,
    "number_diagnoses"    : n_diag_col,
    "readmission_cost_inr": (y_all.values * COST_READMISSION_INR),
})

# ── Clean gender 
gender_clean_map = {
    "Male": "Male", "Female": "Female",
    "male": "Male", "female": "Female",
    1: "Male",  0: "Female",
    "1": "Male","0": "Female",
}
df_dash["gender"] = df_dash["gender"].map(
    lambda x: gender_clean_map.get(x, "Unknown")
)

# ── Clean age group 
valid_ages = [
    "[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
    "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"
]
df_dash["age_group"] = df_dash["age_group"].apply(
    lambda x: x if x in valid_ages else "[60-70)"
)

# ── Save thresholds alongside CSV as a small config 
config_path = os.path.join(DATA_DIR, "dashboard_config.csv")
pd.DataFrame({
    "P50": [P50],
    "P85": [P85],
    "cost_inr_per_readmission": [COST_READMISSION_INR],
}).to_csv(config_path, index=False)
print(f"  ✔  Config saved → {config_path}")

# ── Summary 
print(f"\n  Dashboard dataset summary:")
print(f"    Shape              : {df_dash.shape}")
print(f"    Readmission rate   : {df_dash['readmitted_30d'].mean()*100:.1f}%")
print(f"    Risk tier counts   :")
for tier in ["High","Medium","Low"]:
    cnt = (df_dash["risk_tier"] == tier).sum()
    pct = cnt / len(df_dash) * 100
    print(f"      {tier:<8} : {cnt:>7,}  ({pct:.1f}%)")
print(f"    Gender values      : {sorted(df_dash['gender'].unique())}")
print(f"    Age groups         : {sorted(df_dash['age_group'].unique())}")
print(f"    Diagnosis cats     : {sorted(df_dash['diag_category'].unique())}")

# ── Save 
df_dash.to_csv(OUT_PATH, index=False)
print(f"\n  ✔  diabetic_dashboard.csv saved → {OUT_PATH}")
print(f"     Rows    : {len(df_dash):,}")
print(f"     Columns : {df_dash.shape[1]}")

print(f"""
  Use these threshold values in streamlit_app.py:
    P50 = {P50:.4f}   (medium risk threshold)
    P85 = {P85:.4f}   (high risk threshold)
  These are auto-loaded from dashboard_config.csv by the app.
""")
print("✅  generate_dashboard.py Complete.\n")