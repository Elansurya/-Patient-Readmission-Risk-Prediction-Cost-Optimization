import pandas as pd
import numpy as np
import os

DATA_PATH = r"C:\project\Patient Readmission Risk Prediction + Cost Optimization\data\diabetic_features.csv"
OUT_PATH  = r"C:\project\Patient Readmission Risk Prediction + Cost Optimization\data\diabetic_model_ready.csv"

print("=" * 65)
print("STEP 6 — TARGET VARIABLE TRANSFORMATION")
print("=" * 65)

# ── Load data 
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Input file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"\n  Input shape: {df.shape}")

# ── Check target column exists 
if "readmitted" not in df.columns:
    raise KeyError("'readmitted' column not found in dataset. Check Step 5 output.")

# ── Show original distribution 
print("\n  Original 'readmitted' distribution:")
for val, cnt in df["readmitted"].value_counts().items():
    pct = cnt / len(df) * 100
    print(f"    {str(val):<6}  →  {cnt:>7,}  ({pct:.1f}%)")

# ── Binary transformation 
df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
df.drop(columns=["readmitted"], inplace=True)

# ── New distribution 
print("\n  Binary target 'readmitted_30d' distribution:")
vc = df["readmitted_30d"].value_counts().sort_index()

for val, cnt in vc.items():
    label = "READMITTED <30d (HIGH RISK)" if val == 1 else "Not readmitted <30d (LOW RISK)"
    pct = cnt / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"    {val}  →  {cnt:>7,}  ({pct:.1f}%)  {bar}  {label}")

# ── Class imbalance ratio 
count_0 = vc.get(0, 0)
count_1 = vc.get(1, 0)

if count_1 == 0:
    print("\n  ⚠  WARNING: No positive samples (class 1) found in target column!")
    print("     Check that 'readmitted' column contains '<30' values.")
else:
    ratio = count_0 / count_1
    print(f"\n  Class imbalance ratio (0:1) : {ratio:.1f} : 1")
    if ratio >= 5:
        print("  → This is a heavily imbalanced dataset.")
    elif ratio >= 2:
        print("  → This is a moderately imbalanced dataset.")
    else:
        print("  → Classes are relatively balanced.")
    print("  → We will use class_weight='balanced' and focus on Recall.")

# ── Verify no NaNs in target 
nan_count = df["readmitted_30d"].isnull().sum()
assert nan_count == 0, f"NaN values found in target column: {nan_count}"
print(f"\n  ✔  Target column is clean — zero NaN values")

# ── Final shape & preview 
print(f"  Final dataset shape : {df.shape}")
print(f"  Columns             : {df.columns.tolist()}")
print(f"\n  Sample (first 3 rows):\n{df.head(3).to_string()}")

# ── Save 
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"\n✅  Step 6 Complete — Model-ready dataset saved → '{OUT_PATH}'\n")