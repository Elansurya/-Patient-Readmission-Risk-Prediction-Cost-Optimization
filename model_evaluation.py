import pandas as pd
import numpy as np
import pickle
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)

# ── Paths 
BASE_DIR  = r"C:\project\Patient Readmission Risk Prediction + Cost Optimization"
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("STEP 9 — MODEL EVALUATION")
print("=" * 65)

# ── Guard: check all required files 
required = {
    "X_test data"       : os.path.join(DATA_DIR,  "X_test.csv"),
    "y_test data"       : os.path.join(DATA_DIR,  "y_test.csv"),
    "Feature names"     : os.path.join(MODEL_DIR, "feature_names.pkl"),
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression.pkl"),
    "Random Forest"     : os.path.join(MODEL_DIR, "random_forest.pkl"),
    "Gradient Boosting" : os.path.join(MODEL_DIR, "gradient_boosting.pkl"),
}
missing = [(label, path) for label, path in required.items()
           if not os.path.exists(path)]
if missing:
    print("\n  ERROR — Missing files (run Steps 7 & 8 first):")
    for label, path in missing:
        print(f"    [{label}]  {path}")
    sys.exit(1)

# ── Load feature names 
with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)
print(f"\n  Feature names loaded : {len(feature_names)} columns")

# ── Load test data 
X_test_raw = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"), low_memory=False)
y_raw      = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))

# Robust y_test extraction
def extract_y(raw):
    if isinstance(raw, pd.DataFrame):
        if "readmitted_30d" in raw.columns:
            return raw["readmitted_30d"]
        non_idx = [c for c in raw.columns
                   if c.lower() not in ("index", "unnamed: 0")]
        return raw[non_idx[-1]] if non_idx else raw.iloc[:, -1]
    return raw

y_test = extract_y(y_raw).astype(int).reset_index(drop=True)

print(f"  Test set  : {X_test_raw.shape[0]:,} rows x {X_test_raw.shape[1]} features")
print(f"  Positives : {y_test.sum():,}  ({y_test.mean()*100:.1f}%)")
print(f"  Negatives : {(y_test==0).sum():,}  ({(y_test==0).mean()*100:.1f}%)")

if y_test.nunique() < 2:
    print("\n  FATAL: y_test has only one class — cannot evaluate.")
    sys.exit(1)

# ── Encode any string columns 
str_cols = X_test_raw.select_dtypes(include=["object"]).columns.tolist()
if str_cols:
    print(f"\n  ⚠  Encoding {len(str_cols)} string columns: {str_cols}")
    le = LabelEncoder()
    for col in str_cols:
        X_test_raw[col] = le.fit_transform(X_test_raw[col].astype(str))
else:
    print("  ✔  All features already numeric")

# ── Fill NaN 
nan_total = X_test_raw.isnull().sum().sum()
if nan_total > 0:
    X_test_raw = X_test_raw.fillna(X_test_raw.median(numeric_only=True))
    print(f"  ✔  Filled {nan_total} NaN values")

# ── Align columns to training feature order 
for col in feature_names:
    if col not in X_test_raw.columns:
        X_test_raw[col] = 0.0
        print(f"  ⚠  Added missing column: {col}")

extra_cols = [c for c in X_test_raw.columns if c not in feature_names]
if extra_cols:
    X_test_raw = X_test_raw.drop(columns=extra_cols)

X_test = X_test_raw[feature_names].astype(float)
print(f"  ✔  Features aligned: {X_test.shape[1]} columns")

# ── Load scaler 
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("  ✔  Loaded saved StandardScaler")
else:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_test)
    print("  ⚠  No saved scaler — fitted fresh on X_test")

# ── Detect if X_test is already scaled 
val_std = float(X_test.values.std())
if val_std < 10:
    X_test_scaled = X_test.values
    print(f"  ✔  Data pre-scaled (std={val_std:.3f}) — skipping re-scale")
else:
    X_test_scaled = scaler.transform(X_test.values)
    print(f"  ✔  Applied StandardScaler (std={val_std:.3f})")

# ── Load models 
model_files = {
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression.pkl"),
    "Random Forest"      : os.path.join(MODEL_DIR, "random_forest.pkl"),
    "Gradient Boosting"  : os.path.join(MODEL_DIR, "gradient_boosting.pkl"),
}
models = {}
for name, path in model_files.items():
    with open(path, "rb") as f:
        models[name] = pickle.load(f)
    print(f"  ✔  Loaded: {name}")

# ── Verify feature count matches each model 
print(f"\n  Feature count check:")
for name, model in models.items():
    expected = model.n_features_in_
    actual   = X_test_scaled.shape[1]
    status   = "✔" if expected == actual else "✖ MISMATCH"
    print(f"    {name:<25} expects {expected}, got {actual}  {status}")
    if expected != actual:
        print(f"\n  FATAL: {name} expects {expected} features but X_test has {actual}.")
        print("  Re-run Steps 7 and 8 to sync features and models.")
        sys.exit(1)

# ── Evaluate all models 
results = {}
for name, model in models.items():
    X_eval = X_test_scaled if name == "Logistic Regression" else X_test.values
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    results[name] = {
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score(y_test, y_pred, zero_division=0),
        "f1"       : f1_score(y_test, y_pred, zero_division=0),
        "auc"      : roc_auc_score(y_test, y_prob),
    }

# ── Metrics table 
print("\n" + "=" * 65)
print("PERFORMANCE METRICS — TEST SET")
print("=" * 65)
print(f"  {'Model':<25} {'Accuracy':>9} {'Precision':>10} "
      f"{'Recall':>8} {'F1':>8} {'AUC-ROC':>9}")
print("  " + "-" * 63)

best_auc   = max(m["auc"] for m in results.values())
best_recall = max(results, key=lambda n: results[n]["recall"])

for name, m in results.items():
    flag = "  <- BEST AUC" if m["auc"] == best_auc else ""
    print(f"  {name:<25} {m['accuracy']:>9.4f} {m['precision']:>10.4f}"
          f" {m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc']:>9.4f}{flag}")

print(f"\n  Best Recall : {best_recall}  "
      f"({results[best_recall]['recall']:.4f})")
print("  High Recall = fewer high-risk patients missed.\n")

# ── Detailed classification reports 
print("-" * 65)
print("DETAILED CLASSIFICATION REPORTS")
print("-" * 65)
for name, m in results.items():
    print(f"\n  -- {name} --")
    print(classification_report(
        y_test, m["y_pred"],
        target_names=["Not Readmitted (0)", "Readmitted (1)"],
        digits=4, zero_division=0
    ))

# FIGURE A — Confusion Matrices

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Fig 9a — Confusion Matrices (Test Set)",
             fontsize=13, fontweight="bold")

for ax, (name, m) in zip(axes, results.items()):
    cm  = confusion_matrix(y_test, m["y_pred"])
    im  = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (Not)", "1 (Readmit)"])
    ax.set_yticklabels(["0 (Not)", "1 (Readmit)"])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
path_a = os.path.join(OUT_DIR, "fig9a_confusion_matrices.png")
plt.savefig(path_a, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔  Saved: fig9a_confusion_matrices.png")

# FIGURE B — ROC Curves

line_colors = ["#2196F3", "#4CAF50", "#FF5722"]
fig, ax = plt.subplots(figsize=(8, 6))

for (name, m), color in zip(results.items(), line_colors):
    fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
    ax.plot(fpr, tpr,
            label=f"{name}  (AUC={m['auc']:.3f})",
            color=color, linewidth=2)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("Fig 9b — ROC Curves Comparison",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
path_b = os.path.join(OUT_DIR, "fig9b_roc_curves.png")
plt.savefig(path_b, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔  Saved: fig9b_roc_curves.png")

# FIGURE C — Precision-Recall Curves

fig, ax = plt.subplots(figsize=(8, 6))

for (name, m), color in zip(results.items(), line_colors):
    prec, rec, _ = precision_recall_curve(y_test, m["y_prob"])
    ax.plot(rec, prec, label=name, color=color, linewidth=2)

baseline = float(y_test.mean())
ax.axhline(baseline, color="gray", linestyle="--",
           label=f"No-skill baseline ({baseline:.2f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Fig 9c — Precision-Recall Curves",
             fontsize=13, fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
path_c = os.path.join(OUT_DIR, "fig9c_precision_recall.png")
plt.savefig(path_c, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔  Saved: fig9c_precision_recall.png")

# FIGURE D — Feature Importance (Random Forest)
rf_model = models["Random Forest"]

if hasattr(rf_model, "feature_importances_"):
    fi     = pd.Series(rf_model.feature_importances_, index=feature_names)
    fi_top = fi.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(fi_top)))
    fi_top.sort_values().plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Fig 9d — Top 20 Feature Importances (Random Forest)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature Importance Score")
    plt.tight_layout()
    path_d = os.path.join(OUT_DIR, "fig9d_feature_importance_rf.png")
    plt.savefig(path_d, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Saved: fig9d_feature_importance_rf.png")
else:
    print("  ⚠  Random Forest has no feature_importances_ — skipping Fig 9d")
    fi_top = pd.Series(dtype=float)

# FIGURE E — Feature Importance (Gradient Boosting)
gb_model = models["Gradient Boosting"]

if hasattr(gb_model, "feature_importances_"):
    fi_gb     = pd.Series(gb_model.feature_importances_, index=feature_names)
    fi_gb_top = fi_gb.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(fi_gb_top)))
    fi_gb_top.sort_values().plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Fig 9e — Top 20 Feature Importances (Gradient Boosting)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature Importance Score")
    plt.tight_layout()
    path_e = os.path.join(OUT_DIR, "fig9e_feature_importance_gb.png")
    plt.savefig(path_e, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Saved: fig9e_feature_importance_gb.png")
else:
    print("  ⚠  Gradient Boosting has no feature_importances_ — skipping Fig 9e")


# FIGURE F — Model Comparison Bar Chart

metric_keys   = ["accuracy", "precision", "recall", "f1", "auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
x     = np.arange(len(metric_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, color) in enumerate(zip(models.keys(), line_colors)):
    vals = [results[name][k] for k in metric_keys]
    bars = ax.bar(x + i * width, vals, width,
                  label=name, color=color,
                  edgecolor="white", alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.005, f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=7, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Fig 9f — Model Performance Comparison",
             fontsize=13, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path_f = os.path.join(OUT_DIR, "fig9f_model_comparison.png")
plt.savefig(path_f, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✔  Saved: fig9f_model_comparison.png")

# ── Console: top features 
if len(fi_top) > 0:
    print(f"\n  Top 10 Predictive Features (Random Forest):")
    for feat, score in fi_top.head(10).items():
        bar = "X" * int(score * 300)
        print(f"    {feat:<35} {score:.4f}  {bar}")

# ── Save metrics to CSV 
metrics_df = pd.DataFrame({
    name: {
        "Accuracy" : m["accuracy"],
        "Precision": m["precision"],
        "Recall"   : m["recall"],
        "F1-Score" : m["f1"],
        "AUC-ROC"  : m["auc"],
    }
    for name, m in results.items()
}).T.round(4)

csv_path = os.path.join(OUT_DIR, "model_metrics.csv")
metrics_df.to_csv(csv_path)
print(f"\n  ✔  Metrics saved → {csv_path}")
print(f"\n{metrics_df.to_string()}")

# ── Output summary 
print(f"""
  Output files saved → {OUT_DIR}
    fig9a_confusion_matrices.png
    fig9b_roc_curves.png
    fig9c_precision_recall.png
    fig9d_feature_importance_rf.png
    fig9e_feature_importance_gb.png
    fig9f_model_comparison.png
    model_metrics.csv
""")
print("=" * 65)
print("✅  Step 9 Complete — Model evaluation done.")
print("=" * 65)