import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

os.makedirs("models", exist_ok=True)

THRESHOLD = 0.30   # lower threshold → better recall for minority class

print("=" * 65)
print("STEP 8 — MODEL TRAINING (FIXED)")
print("=" * 65)

# ── Load training data 
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()

X_val_path = "data/X_val.csv"
y_val_path  = "data/y_val.csv"
has_val = os.path.exists(X_val_path) and os.path.exists(y_val_path)

if has_val:
    X_val = pd.read_csv(X_val_path)
    y_val = pd.read_csv(y_val_path).squeeze()
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    print("  i  No val set found — using 15% of train as validation.")

# ── Reset indices to prevent alignment bugs 
X_train = X_train.reset_index(drop=True)
X_val   = X_val.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val   = y_val.reset_index(drop=True)

print(f"\n  Train : {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
print(f"  Val   : {X_val.shape[0]:,} rows x {X_val.shape[1]} features")
print(f"  Train positives: {y_train.sum():,}  ({y_train.mean()*100:.1f}%)")
print(f"  Val   positives: {y_val.sum():,}  ({y_val.mean()*100:.1f}%)")

# ── Validate labels 
if y_train.nunique() < 2:
    print("\n  FATAL: y_train has only one class. Run fix_target_encoding.py first.")
    raise SystemExit(1)

# ── Encode categorical columns 
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

if cat_cols:
    print(f"\n  Found {len(cat_cols)} categorical column(s) — label encoding:")
    for col in cat_cols:
        print(f"       * {col}")

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on combined train + val to avoid unseen-label errors
        combined = pd.concat(
            [X_train[col].astype(str), X_val[col].astype(str)], axis=0
        )
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col]   = le.transform(X_val[col].astype(str))
        le_dict[col] = le

    with open("models/label_encoders.pkl", "wb") as f:
        pickle.dump(le_dict, f)
    print(f"  [OK] Label encoders saved -> models/label_encoders.pkl")
else:
    print("\n  [OK] No categorical columns found — all numeric.")

# ── Force all columns to numeric (safety net) 
X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
X_val   = X_val.apply(pd.to_numeric, errors="coerce").fillna(0)

# ── Save feature names 
feature_names = X_train.columns.tolist()
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print(f"\n  [OK] Saved feature names ({len(feature_names)} features)")

# ── Scale features for Logistic Regression 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("  [OK] Scaler fitted and saved")

# ── Class weights (for LR and RF) 
classes = np.array([0, 1])
cw      = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = {0: cw[0], 1: cw[1]}
print(f"\n  Class weights -> 0: {cw[0]:.3f}  |  1: {cw[1]:.3f}")

# ── Sample weights for Gradient Boosting 
sample_weights_train = compute_sample_weight("balanced", y=y_train)

# ── Save threshold for downstream inference 
with open("models/threshold.pkl", "wb") as f:
    pickle.dump(THRESHOLD, f)
print(f"  Decision threshold : {THRESHOLD}  (saved -> models/threshold.pkl)")

# Helper functions

def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy"  : accuracy_score(y_true, y_pred),
        "Precision" : precision_score(y_true, y_pred, zero_division=0),
        "Recall"    : recall_score(y_true, y_pred, zero_division=0),
        "F1"        : f1_score(y_true, y_pred, zero_division=0),
        "AUC"       : roc_auc_score(y_true, y_prob),
    }


def train_and_save(name, model, X_tr, X_v, y_tr, y_v,
                   save_path, sample_weight=None):
    print(f"\n  {'-'*55}")
    print(f"  Training: {name}")
    print(f"  {'-'*55}")
    t0 = time.time()

    if sample_weight is not None:
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
    else:
        model.fit(X_tr, y_tr)

    elapsed = time.time() - t0
    y_prob  = model.predict_proba(X_v)[:, 1]

    m_default   = evaluate_at_threshold(y_v, y_prob, 0.50)
    m_threshold = evaluate_at_threshold(y_v, y_prob, THRESHOLD)

    print(f"  Training time      : {elapsed:.1f}s")

    print(f"\n  -- @ threshold = 0.50 (default) --------------------------")
    print(f"  Accuracy    : {m_default['Accuracy']:.4f}")
    print(f"  Precision   : {m_default['Precision']:.4f}")
    print(f"  Recall      : {m_default['Recall']:.4f}")
    print(f"  F1-Score    : {m_default['F1']:.4f}")
    print(f"  AUC-ROC     : {m_default['AUC']:.4f}")

    print(f"\n  -- @ threshold = {THRESHOLD} (tuned for recall) ------------------")
    print(f"  Accuracy    : {m_threshold['Accuracy']:.4f}")
    print(f"  Precision   : {m_threshold['Precision']:.4f}")
    print(f"  Recall      : {m_threshold['Recall']:.4f}  <- key metric")
    print(f"  F1-Score    : {m_threshold['F1']:.4f}")
    print(f"  AUC-ROC     : {m_threshold['AUC']:.4f}")

    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  [OK] Saved -> {save_path}")

    return m_threshold   # summary uses tuned-threshold metrics

# 1. Logistic Regression

lr = LogisticRegression(
    C=0.1,
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
lr_metrics = train_and_save(
    "Logistic Regression", lr,
    X_train_scaled, X_val_scaled, y_train, y_val,
    "models/logistic_regression.pkl"
)

# 2. Random Forest

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_metrics = train_and_save(
    "Random Forest", rf,
    X_train, X_val, y_train, y_val,
    "models/random_forest.pkl"
)


# 3. Gradient Boosting  (sample_weight fixes class imbalance)

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)
gb_metrics = train_and_save(
    "Gradient Boosting", gb,
    X_train, X_val, y_train, y_val,
    "models/gradient_boosting.pkl",
    sample_weight=sample_weights_train      
)

# ── Summary table 
print("\n" + "=" * 65)
print(f"TRAINING SUMMARY  --  VAL PERFORMANCE @ threshold = {THRESHOLD}")
print("=" * 65)
all_metrics = {
    "Logistic Regression": lr_metrics,
    "Random Forest":       rf_metrics,
    "Gradient Boosting":   gb_metrics,
}
print(f"  {'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("  " + "-" * 62)
best_auc = max(v["AUC"] for v in all_metrics.values())
for name, m in all_metrics.items():
    tag = " <-- BEST" if m["AUC"] == best_auc else ""
    print(f"  {name:<25} {m['Accuracy']:>9.4f} {m['Precision']:>10.4f}"
          f" {m['Recall']:>8.4f} {m['F1']:>8.4f} {m['AUC']:>8.4f}{tag}")

# ── Pick & save best model by AUC 
best_name = max(all_metrics, key=lambda k: all_metrics[k]["AUC"])
print(f"\n  Best model by AUC: {best_name}")

model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Random Forest":       "models/random_forest.pkl",
    "Gradient Boosting":   "models/gradient_boosting.pkl",
}
with open(model_files[best_name], "rb") as f:
    best_model = pickle.load(f)
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print(f"  [OK] Best model saved -> models/best_model.pkl")

# ── Save summary CSV 
summary_df = pd.DataFrame(all_metrics).T
summary_df.to_csv("models/training_summary.csv")
print(f"  [OK] Summary saved  -> models/training_summary.csv")

print("""
  -- Saved model files 
    models/logistic_regression.pkl
    models/random_forest.pkl
    models/gradient_boosting.pkl
    models/best_model.pkl
    models/scaler.pkl
    models/feature_names.pkl
    models/label_encoders.pkl
    models/threshold.pkl
    models/training_summary.csv
""")
print("STEP 8 Complete -- All models trained and saved.\n")
print("  Next: python model_evaluation.py\n")