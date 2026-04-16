import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Paths 
BASE_DIR    = r"C:\project\Patient Readmission Risk Prediction + Cost Optimization"
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_READY = os.path.join(DATA_DIR, "diabetic_model_ready.csv")

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 65)
print("STEP 7 — TRAIN / TEST SPLIT")
print("=" * 65)

# ── Load model-ready dataset 
if not os.path.exists(MODEL_READY):
    raise FileNotFoundError(f"Not found: {MODEL_READY}\nRe-run Steps 5 and 6 first.")

df = pd.read_csv(MODEL_READY, low_memory=False)
print(f"\n  Input shape : {df.shape}")

TARGET = "readmitted_30d"
if TARGET not in df.columns:
    raise KeyError(f"'{TARGET}' not found — re-run Step 6.")

# ── Separate features / target 
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

print(f"\n  Target distribution:")
vc = y.value_counts().sort_index()
for val, cnt in vc.items():
    print(f"    Class {val} → {cnt:,}  ({cnt/len(y)*100:.1f}%)")

if 1 not in y.values:
    raise ValueError("Class 1 missing — re-run Step 6.")

# ── Drop non-numeric columns 
obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
if obj_cols:
    print(f"\n  ⚠  Dropping {len(obj_cols)} non-numeric columns: {obj_cols}")
    X = X.drop(columns=obj_cols)
else:
    print("\n  ✔  All columns numeric")

# ── Fill NaN 
nan_total = X.isnull().sum().sum()
if nan_total > 0:
    X = X.fillna(X.median(numeric_only=True))
    print(f"  ✔  Filled {nan_total} NaN values with column median")

# ── CORE FIX: Drop zero-variance columns 
# StandardScaler divides by std — if std=0 the result is NaN
col_std      = X.std(numeric_only=True)
zero_var_cols = col_std[col_std == 0].index.tolist()

if zero_var_cols:
    print(f"\n  ⚠  Dropping {len(zero_var_cols)} zero-variance columns")
    print(f"     (all values identical — useless for model & breaks scaler):")
    for col in zero_var_cols:
        print(f"       - {col}  (unique value: {X[col].iloc[0]})")
    X = X.drop(columns=zero_var_cols)
else:
    print("  ✔  No zero-variance columns found")

# ── Drop near-zero variance columns (std < 1e-6) 
col_std2       = X.std(numeric_only=True)
near_zero_cols = col_std2[col_std2 < 1e-6].index.tolist()
if near_zero_cols:
    print(f"\n  ⚠  Dropping {len(near_zero_cols)} near-zero-variance columns:")
    for col in near_zero_cols:
        print(f"       - {col}  (std={col_std2[col]:.2e})")
    X = X.drop(columns=near_zero_cols)

# ── Final NaN check after drops 
remaining_nan = X.isnull().sum().sum()
if remaining_nan > 0:
    X = X.fillna(0)
    print(f"  ✔  Filled {remaining_nan} remaining NaN after variance drops")

print(f"\n  Features after cleanup : {X.shape[1]}")
print(f"  Samples                : {len(X):,}")
print(f"  Dtypes : {X.dtypes.value_counts().to_dict()}")

# ── Stratified split 
min_class    = y.value_counts().min()
use_stratify = min_class >= 10

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y if use_stratify else None
)

X_train = X_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

print(f"\n  Train size : {len(X_train):,} rows")
print(f"  Test  size : {len(X_test):,}  rows")
print(f"  Stratified : {use_stratify}")

print(f"\n  Class balance — train:")
for val, cnt in y_train.value_counts().sort_index().items():
    print(f"    Class {val} → {cnt:,}  ({cnt/len(y_train)*100:.1f}%)")

print(f"\n  Class balance — test:")
for val, cnt in y_test.value_counts().sort_index().items():
    print(f"    Class {val} → {cnt:,}  ({cnt/len(y_test)*100:.1f}%)")

# ── Verify train split has no zero-variance columns 
train_std       = X_train.std()
train_zero_var  = train_std[train_std < 1e-9].index.tolist()
if train_zero_var:
    print(f"\n  ⚠  Zero-variance in train split after split: {train_zero_var}")
    print(f"     Dropping these too ...")
    X_train = X_train.drop(columns=train_zero_var)
    X_test  = X_test.drop(columns=[c for c in train_zero_var
                                    if c in X_test.columns])

# ── Feature scaling 
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values.astype(float))
X_test_scaled  = scaler.transform(X_test.values.astype(float))

X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_df  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

# ── Fix any NaN produced by scaler (safety net) 
nan_train = X_train_df.isnull().sum().sum()
nan_test  = X_test_df.isnull().sum().sum()

if nan_train > 0 or nan_test > 0:
    print(f"\n  ⚠  NaN after scaling: train={nan_train}, test={nan_test}")
    print(f"     Identifying problem columns ...")

    # Find which columns still produce NaN
    bad_cols = X_train_df.columns[X_train_df.isnull().any()].tolist()
    print(f"     Bad columns: {bad_cols}")
    print(f"     Dropping and refitting scaler ...")

    X_train = X_train.drop(columns=bad_cols)
    X_test  = X_test.drop(columns=[c for c in bad_cols
                                    if c in X_test.columns])

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(float))
    X_test_scaled  = scaler.transform(X_test.values.astype(float))

    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

    nan_train_final = X_train_df.isnull().sum().sum()
    nan_test_final  = X_test_df.isnull().sum().sum()
    print(f"     NaN after refit — train:{nan_train_final}  test:{nan_test_final}")
else:
    print(f"\n  ✔  Scaler applied cleanly — zero NaN")

# ── Final assertion 
final_nan_train = X_train_df.isnull().sum().sum()
final_nan_test  = X_test_df.isnull().sum().sum()

assert final_nan_train == 0, \
    f"X_train still has {final_nan_train} NaN — check input data"
assert final_nan_test == 0, \
    f"X_test still has {final_nan_test} NaN — check input data"
assert X_train_df.select_dtypes(include=["object"]).empty, \
    "X_train still has string columns"

print(f"  ✔  Final verification passed — clean numeric, zero NaN")

# ── Save splits 
X_train_df.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
X_test_df.to_csv( os.path.join(DATA_DIR, "X_test.csv"),  index=False)
y_train.to_csv(   os.path.join(DATA_DIR, "y_train.csv"), index=False)
y_test.to_csv(    os.path.join(DATA_DIR, "y_test.csv"),  index=False)

print(f"\n  ✔  Saved splits → {DATA_DIR}")
print(f"       X_train.csv : {X_train_df.shape}")
print(f"       X_test.csv  : {X_test_df.shape}")
print(f"       y_train.csv : {y_train.shape}")
print(f"       y_test.csv  : {y_test.shape}")

# ── Save scaler and feature names 
feature_names = X_train_df.columns.tolist()

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)

print(f"\n  ✔  Saved → {MODELS_DIR}")
print(f"       scaler.pkl        : StandardScaler ({len(feature_names)} features)")
print(f"       feature_names.pkl : {feature_names}")

# ── Final summary 
print(f"""
  Summary:
    Source         : diabetic_model_ready.csv
    Total samples  : {len(df):,}
    Train samples  : {len(X_train_df):,}
    Test  samples  : {len(X_test_df):,}
    Features kept  : {X_train_df.shape[1]}
    Zero-var drops : {len(zero_var_cols)}
    All numeric    : YES
    NaN values     : NONE
""")

print("✅  Step 7 Complete — Clean numeric splits saved.\n")