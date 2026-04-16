import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ── Paths 
BASE_DIR  = r"C:\project\Patient Readmission Risk Prediction + Cost Optimization"
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")

OUT_PATH = os.path.join(DATA_DIR, "diabetic_scored.csv")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 65)
print("STEP 10 — RISK SCORING")
print("=" * 65)

# ── Guard: check required files 
required = {
    "X_train"      : os.path.join(DATA_DIR,  "X_train.csv"),
    "X_test"       : os.path.join(DATA_DIR,  "X_test.csv"),
    "y_train"      : os.path.join(DATA_DIR,  "y_train.csv"),
    "y_test"       : os.path.join(DATA_DIR,  "y_test.csv"),
    "Random Forest": os.path.join(MODEL_DIR, "random_forest.pkl"),
    "Scaler"       : os.path.join(MODEL_DIR, "scaler.pkl"),
}
missing = [(lbl, p) for lbl, p in required.items() if not os.path.exists(p)]
if missing:
    print("\n  ERROR — Missing files (run Steps 7 & 8 first):")
    for lbl, p in missing:
        print(f"    [{lbl}]  {p}")
    raise SystemExit(1)

# CORE FIX: Always build X from X_train + X_test (already
# engineered, scaled, and model-compatible). Never rely on
# diabetic_model_ready.csv which has raw/unencoded columns.


# ── Load train/test splits 
print("\n  Loading engineered train/test splits ...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"), low_memory=False)
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"),  low_memory=False)
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv") ).squeeze()

# Robust y extraction
def extract_y(raw):
    if isinstance(raw, pd.DataFrame):
        if "readmitted_30d" in raw.columns:
            return raw["readmitted_30d"]
        return raw.iloc[:, -1]
    return raw

y_train = extract_y(y_train).astype(int).reset_index(drop=True)
y_test  = extract_y(y_test ).astype(int).reset_index(drop=True)

# ── Concatenate to full dataset 
X_all = pd.concat([X_train, X_test], ignore_index=True)
y_all = pd.concat([y_train, y_test],  ignore_index=True)

print(f"  X_train : {X_train.shape[0]:,} rows x {X_train.shape[1]} cols")
print(f"  X_test  : {X_test.shape[0]:,}  rows x {X_test.shape[1]} cols")
print(f"  X_all   : {X_all.shape[0]:,} rows x {X_all.shape[1]} cols")
print(f"  y_all class distribution: {y_all.value_counts().sort_index().to_dict()}")

# ── Update feature_names.pkl to match actual training cols ─
feature_names = X_train.columns.tolist()
with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(feature_names, f)
print(f"  ✔  feature_names.pkl updated: {len(feature_names)} columns")

# ── Encode any remaining string columns 
str_cols = X_all.select_dtypes(include=["object"]).columns.tolist()
if str_cols:
    print(f"  ⚠  Encoding {len(str_cols)} string columns: {str_cols}")
    le = LabelEncoder()
    for col in str_cols:
        X_all[col] = le.fit_transform(X_all[col].astype(str))
else:
    print("  ✔  All columns already numeric")

# ── Fill NaN 
nan_total = X_all.isnull().sum().sum()
if nan_total > 0:
    X_all = X_all.fillna(X_all.median(numeric_only=True))
    print(f"  ✔  Filled {nan_total} NaN values with column median")

# ── Ensure all values are float 
X_all = X_all.astype(float)
print(f"  ✔  All features cast to float")

# ── Load scaler and model 
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
    rf = pickle.load(f)
print("  ✔  Loaded scaler and Random Forest model")

# ── Align columns exactly to model's training order 
model_features = X_train.columns.tolist()

# Add missing cols as 0, drop extra cols, reorder
for col in model_features:
    if col not in X_all.columns:
        X_all[col] = 0.0
extra = [c for c in X_all.columns if c not in model_features]
if extra:
    X_all = X_all.drop(columns=extra)
X_all = X_all[model_features]

print(f"  ✔  Features aligned: {X_all.shape[1]} columns")

# ── Scale 
# X_train was already scaled when saved — use as-is
# Only re-scale if scaler expects unscaled input
# Check: if values are roughly in [-5, 5] range, already scaled
val_range = float(X_all.values.std())
if val_range < 10:
    print(f"  ✔  Data appears pre-scaled (std={val_range:.3f}) — skipping re-scale")
    X_final = X_all.values
else:
    print(f"  ✔  Applying StandardScaler (std={val_range:.3f})")
    X_final = scaler.transform(X_all.values)

# ── Generate risk probabilities 
risk_prob = rf.predict_proba(X_final)[:, 1]
risk_pred = rf.predict(X_final)

print(f"\n  ✔  Generated risk scores for {len(risk_prob):,} patients")
print(f"  Score distribution:")
print(f"    Min    : {risk_prob.min():.4f}")
print(f"    Max    : {risk_prob.max():.4f}")
print(f"    Mean   : {risk_prob.mean():.4f}")
print(f"    Median : {np.median(risk_prob):.4f}")
print(f"    Std    : {risk_prob.std():.4f}")

# ── Dynamic percentile-based thresholds 
# Guarantees meaningful 3-way split regardless of score range
LOW_THRESHOLD  = float(np.percentile(risk_prob, 33))
HIGH_THRESHOLD = float(np.percentile(risk_prob, 66))

print(f"\n  Risk Tier Thresholds (33rd / 66th percentile):")
print(f"    Low    : prob < {LOW_THRESHOLD:.4f}")
print(f"    Medium : {LOW_THRESHOLD:.4f} <= prob < {HIGH_THRESHOLD:.4f}")
print(f"    High   : prob >= {HIGH_THRESHOLD:.4f}")

def assign_risk(prob):
    if prob >= HIGH_THRESHOLD:
        return "High"
    elif prob >= LOW_THRESHOLD:
        return "Medium"
    else:
        return "Low"

risk_tier = pd.Series(risk_prob).apply(assign_risk)

print(f"\n  Risk Tier Distribution:")
tier_vc = risk_tier.value_counts()
for tier in ["High", "Medium", "Low"]:
    cnt = tier_vc.get(tier, 0)
    pct = cnt / len(risk_prob) * 100
    bar = "█" * int(pct / 2)
    print(f"    {tier:<8} : {cnt:>7,}  ({pct:.1f}%)  {bar}")

# ── Build scored dataset 
df_scored = pd.DataFrame({
    "readmitted_30d" : y_all.values,
    "risk_score"     : np.round(risk_prob, 4),
    "risk_tier"      : risk_tier.values,
    "predicted_label": risk_pred.astype(int),
    "split"          : (["train"] * len(y_train)) + (["test"] * len(y_test)),
})

# ── Capture rate 
total_pos = int(y_all.sum())
if total_pos > 0:
    actual_pos_high = int(
        ((df_scored["readmitted_30d"] == 1) &
         (df_scored["risk_tier"] == "High")).sum()
    )
    capture_rate = actual_pos_high / total_pos * 100
    print(f"\n  Capture Rate — actual positives in 'High' tier:")
    print(f"    {actual_pos_high:,} / {total_pos:,}  ({capture_rate:.1f}%)")
else:
    print("\n  ⚠  No positive samples — capture rate not computed.")

# ── Tier summary table 
print(f"\n  Tier Summary:")
print(f"  {'Tier':<10} {'Count':>8} {'%':>7}  {'Avg Score':>10}  {'True Positives':>15}")
print("  " + "-" * 58)
for tier in ["High", "Medium", "Low"]:
    subset   = df_scored[df_scored["risk_tier"] == tier]
    cnt      = len(subset)
    pct      = cnt / len(df_scored) * 100 if len(df_scored) > 0 else 0
    avg_sc   = float(subset["risk_score"].mean()) if cnt > 0 else 0.0
    true_pos = int((subset["readmitted_30d"] == 1).sum())
    print(f"  {tier:<10} {cnt:>8,} {pct:>6.1f}%  {avg_sc:>10.4f}  {true_pos:>15,}")

# ── Chart 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Fig 10 — Patient Risk Score Distribution",
             fontsize=13, fontweight="bold")

neg_scores = df_scored[df_scored["readmitted_30d"] == 0]["risk_score"]
pos_scores = df_scored[df_scored["readmitted_30d"] == 1]["risk_score"]

axes[0].hist(neg_scores, bins=50, alpha=0.6,
             color="#4CAF50", label="Not readmitted <30d")
if len(pos_scores) > 0:
    axes[0].hist(pos_scores, bins=50, alpha=0.6,
                 color="#F44336", label="Readmitted <30d")
axes[0].axvline(LOW_THRESHOLD,  color="orange", linestyle="--",
                linewidth=1.5, label=f"Low  ({LOW_THRESHOLD:.3f})")
axes[0].axvline(HIGH_THRESHOLD, color="red",    linestyle="--",
                linewidth=1.5, label=f"High ({HIGH_THRESHOLD:.3f})")
axes[0].set_xlabel("Risk Probability Score")
axes[0].set_ylabel("Patient Count")
axes[0].set_title("Risk Score Distribution by Actual Class")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

tier_order  = ["Low", "Medium", "High"]
tier_counts = [tier_vc.get(t, 0) for t in tier_order]
tier_colors = ["#4CAF50", "#FFC107", "#F44336"]
tier_labels = ["Low Risk", "Medium Risk", "High Risk"]
valid = [(l, c, col) for l, c, col
         in zip(tier_labels, tier_counts, tier_colors) if c > 0]
if valid:
    v_labels, v_counts, v_colors = zip(*valid)
    axes[1].pie(v_counts, labels=v_labels, colors=v_colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(width=0.5, edgecolor="white"))
axes[1].set_title("Patient Risk Tier Segmentation")

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, "fig10_risk_score_distribution.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ✔  Fig 10 saved → {chart_path}")

# ── Save scored dataset 
df_scored.to_csv(OUT_PATH, index=False)
print(f"\n  ✔  Scored dataset saved → {OUT_PATH}")
print(f"     Rows    : {len(df_scored):,}")
print(f"     Columns : {df_scored.shape[1]}")

print("\n✅  Step 10 Complete — Risk scores generated.\n")