import pandas as pd
import numpy as np
import pickle
import time
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# ── Paths 
BASE_DIR   = r"C:\project\Patient Readmission Risk Prediction + Cost Optimization"
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAIN_X_PATH = os.path.join(DATA_DIR,   "X_train.csv")
TRAIN_Y_PATH = os.path.join(DATA_DIR,   "y_train.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 65)
print("STEP 8 — MODEL BUILDING")
print("=" * 65)

# ── Load data 
if not os.path.exists(TRAIN_X_PATH):
    raise FileNotFoundError(f"X_train not found: {TRAIN_X_PATH}")
if not os.path.exists(TRAIN_Y_PATH):
    raise FileNotFoundError(f"y_train not found: {TRAIN_Y_PATH}")

X_train = pd.read_csv(TRAIN_X_PATH, low_memory=False)
y_raw   = pd.read_csv(TRAIN_Y_PATH)

# ── Robust y_train extraction 
def extract_y(raw):
    if isinstance(raw, pd.DataFrame):
        if "readmitted_30d" in raw.columns:
            return raw["readmitted_30d"]
        non_idx = [c for c in raw.columns
                   if c.lower() not in ("index", "unnamed: 0")]
        return raw[non_idx[-1]] if non_idx else raw.iloc[:, -1]
    return raw

y_train = extract_y(y_raw).astype(int).reset_index(drop=True)
X_train = X_train.reset_index(drop=True)

print(f"\n  X_train shape : {X_train.shape}")
print(f"  y_train shape : {y_train.shape}")
print(f"  Classes       : {sorted(y_train.unique().tolist())}")
print(f"  Class counts  : {y_train.value_counts().sort_index().to_dict()}")

# ── Validate 
if 1 not in y_train.values:
    raise ValueError("Class 1 missing from y_train. Re-run Steps 6 and 7.")

# ── Drop non-numeric columns 
non_numeric = X_train.select_dtypes(include=["object"]).columns.tolist()
if non_numeric:
    print(f"  ⚠  Dropping non-numeric columns: {non_numeric}")
    X_train = X_train.drop(columns=non_numeric)

# ── Fill NaN 
nan_total = X_train.isnull().sum().sum()
if nan_total > 0:
    X_train = X_train.fillna(X_train.median(numeric_only=True))
    print(f"  ✔  Filled {nan_total} NaN values")

# ── Drop zero-variance columns 
col_std       = X_train.std()
zero_var_cols = col_std[col_std < 1e-9].index.tolist()
if zero_var_cols:
    print(f"  ⚠  Dropping {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")
    X_train = X_train.drop(columns=zero_var_cols)

X_train = X_train.astype(float)

# ── Save updated feature names 
feature_names = X_train.columns.tolist()
with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)

print(f"\n  ✔  Feature names saved  : {len(feature_names)} features")
print(f"  ✔  Features             : {feature_names}")

# ── Class imbalance 
neg   = (y_train == 0).sum()
pos   = (y_train == 1).sum()
ratio = neg / pos
sample_weight = np.where(y_train == 1, neg / pos, 1.0)

print(f"\n  Class imbalance : {ratio:.1f}:1  (neg:pos)")
print(f"  Training set    : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")

models = {}

# MODEL 1 — Logistic Regression

print("\n" + "─" * 65)
print("1/3  Logistic Regression")
print("─" * 65)

t0 = time.time()
lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=0.1,
    solver="lbfgs",
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train, y_train)
elapsed   = time.time() - t0
train_auc = roc_auc_score(y_train, lr.predict_proba(X_train)[:, 1])
print(f"  ✔  Trained in {elapsed:.1f}s  |  Train AUC-ROC: {train_auc:.4f}")

lr_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
with open(lr_path, "wb") as f:
    pickle.dump(lr, f)
models["Logistic Regression"] = lr
print(f"  Saved → {lr_path}")

# MODEL 2 — Random Forest

print("\n" + "─" * 65)
print("2/3  Random Forest")
print("─" * 65)

t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
elapsed   = time.time() - t0
train_auc = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
print(f"  ✔  Trained in {elapsed:.1f}s  |  Train AUC-ROC: {train_auc:.4f}")

rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
with open(rf_path, "wb") as f:
    pickle.dump(rf, f)
models["Random Forest"] = rf
print(f"  Saved → {rf_path}")

# MODEL 3 — Gradient Boosting

print("\n" + "─" * 65)
print("3/3  Gradient Boosting Classifier")
print("─" * 65)

t0 = time.time()
gbm = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gbm.fit(X_train, y_train, sample_weight=sample_weight)
elapsed   = time.time() - t0
train_auc = roc_auc_score(y_train, gbm.predict_proba(X_train)[:, 1])
print(f"  ✔  Trained in {elapsed:.1f}s  |  Train AUC-ROC: {train_auc:.4f}")

gbm_path = os.path.join(MODELS_DIR, "gradient_boosting.pkl")
with open(gbm_path, "wb") as f:
    pickle.dump(gbm, f)
models["Gradient Boosting"] = gbm
print(f"  Saved → {gbm_path}")

# Summary

print("\n" + "─" * 65)
print("MODEL SUMMARY")
print("─" * 65)
print(f"  {'Model':<30} {'Type':<35} {'Features':>8}")
print(f"  {'─'*29} {'─'*34} {'─'*8}")
for name, model in models.items():
    print(f"  {name:<30} {type(model).__name__:<35} {X_train.shape[1]:>8}")

print(f"\n  All models saved     → {MODELS_DIR}")
print(f"  feature_names.pkl    → {len(feature_names)} features")
print(f"\n✅  Step 8 Complete — All 3 models trained on {X_train.shape[1]} features.\n")