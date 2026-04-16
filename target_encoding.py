import pandas as pd
import numpy as np
import os, glob

print("=" * 60)
print("TARGET ENCODING DIAGNOSIS & FIX")
print("=" * 60)

# ── Step 1: Inspect current y files 
for fname in ["data/y_test.csv", "data/y_train.csv"]:
    if os.path.exists(fname):
        s = pd.read_csv(fname).squeeze()
        print(f"\n  [{fname}]")
        print(f"    dtype        : {s.dtype}")
        print(f"    unique values: {s.unique().tolist()}")
        print(f"    value counts :\n{s.value_counts().to_string()}")
    else:
        print(f"\n  [{fname}] — NOT FOUND")

# ── Step 2: Try to find the original raw data 
print("\n" + "─" * 60)
print("  Searching for original raw data file...")

search_patterns = [
    "*.csv", "data/*.csv", "raw/*.csv",
    "dataset/*.csv", "../*.csv"
]
found_raws = []
for pat in search_patterns:
    found_raws.extend(glob.glob(pat))

# Filter out already-split files
exclude = {"X_train", "X_test", "y_train", "y_test",
           "X_val", "y_val", "model_metrics"}
raw_candidates = [
    f for f in found_raws
    if not any(e in f for e in exclude)
]

if not raw_candidates:
    print("  ⚠  No raw CSV found. Checking if 'readmitted' column exists in X files...")
else:
    print(f"  Found candidates: {raw_candidates}")

# ── Step 3: Check if 'readmitted' was accidentally put in X ─
x_files = {"data/X_test.csv": "data/y_test.csv",
           "data/X_train.csv": "data/y_train.csv"}

for x_path, y_path in x_files.items():
    if not os.path.exists(x_path):
        continue
    X = pd.read_csv(x_path)
    print(f"\n  [{x_path}] columns sample: {X.columns.tolist()[:10]}")

    if "readmitted" in X.columns:
        print(f"  ⚠  'readmitted' column FOUND inside {x_path}!")
        print(f"     Values: {X['readmitted'].unique().tolist()}")
        print(f"     → This means the target was NOT separated correctly.")

# ── Step 4: Fix from raw data if available 
fixed = False
for raw_file in raw_candidates:
    try:
        df = pd.read_csv(raw_file, low_memory=False)
        print(f"\n  Opened: {raw_file}  ({df.shape})")

        if "readmitted" not in df.columns:
            print(f"  'readmitted' column not in this file, skipping.")
            continue

        print(f"  'readmitted' raw values: {df['readmitted'].unique().tolist()}")

        # ── THE CORRECT MAPPING 
        # '<30'  → 1  (readmitted within 30 days = HIGH RISK)
        # '>30'  → 0  (readmitted after 30 days  = lower risk)
        # 'NO'   → 0  (not readmitted             = lower risk)
        df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

        pos = df["readmitted_binary"].sum()
        total = len(df)
        print(f"\n  ✔  Encoded: '<30'→1, rest→0")
        print(f"     Positives : {pos:,}  ({pos/total*100:.1f}%)")
        print(f"     Negatives : {total-pos:,}  ({(total-pos)/total*100:.1f}%)")

        if pos == 0:
            print("  ❌  Still 0 positives after encoding.")
            print(f"      Unique values again: {df['readmitted'].unique().tolist()}")
            print("      Try manual mapping below ↓")
            # Show all unique and let user map
            vals = df["readmitted"].unique().tolist()
            print(f"      All unique target values: {vals}")
            continue

        # ── Rebuild train/test split with correct labels ─
        from sklearn.model_selection import train_test_split

        if os.path.exists("data/X_test.csv") and os.path.exists("data/X_train.csv"):
            X_test  = pd.read_csv("data/X_test.csv")
            X_train = pd.read_csv("data/X_train.csv")
            n_test  = len(X_test)
            n_train = len(X_train)

            # Align indices: use the same rows as X split
            # Best approach: re-split from full data to match sizes
            feature_cols = [c for c in df.columns
                            if c not in ("readmitted", "readmitted_binary",
                                         "encounter_id", "patient_nbr")]
            X_full = df[feature_cols]
            y_full = df["readmitted_binary"]

            # Remove rows with NaN in target
            mask = y_full.notna()
            X_full = X_full[mask].reset_index(drop=True)
            y_full = y_full[mask].reset_index(drop=True)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_full, y_full,
                test_size=n_test / len(X_full),
                random_state=42,
                stratify=y_full          # keeps class ratio in both splits
            )

            y_te.to_csv("data/y_test.csv",  index=False)
            y_tr.to_csv("data/y_train.csv", index=False)
            X_te.to_csv("data/X_test.csv",  index=False)
            X_tr.to_csv("data/X_train.csv", index=False)

            print(f"\n  ✔  Rebuilt and saved all 4 data files with correct labels.")
            print(f"     y_test  positives: {y_te.sum():,} / {len(y_te):,}")
            print(f"     y_train positives: {y_tr.sum():,} / {len(y_tr):,}")
            fixed = True
            break
        else:
            # Just fix y files using the full dataset split proportionally
            y_full = df["readmitted_binary"]
            n = len(y_full)
            test_size = 0.2
            split_idx = int(n * (1 - test_size))
            y_tr = y_full.iloc[:split_idx]
            y_te = y_full.iloc[split_idx:]
            y_te.to_csv("data/y_test.csv",  index=False)
            y_tr.to_csv("data/y_train.csv", index=False)
            print(f"\n  ✔  y files rebuilt. y_test positives: {y_te.sum():,}")
            fixed = True
            break

    except Exception as e:
        print(f"  Error reading {raw_file}: {e}")

# ── Step 5: Manual fix fallback 
if not fixed:
    print("\n" + "=" * 60)
    print("  MANUAL FIX INSTRUCTIONS")
    print("=" * 60)
    print("""
  Option A — If you have the original CSV:
  import pandas as pd
  from sklearn.model_selection import train_test_split

  df = pd.read_csv("your_original_file.csv")
  print(df["readmitted"].unique())   # see what values exist

  # Correct encoding:
  df["label"] = (df["readmitted"] == "<30").astype(int)

  # Re-split
  X = df.drop(columns=["readmitted", "label"])
  y = df["label"]
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )
  X_train.to_csv("data/X_train.csv", index=False)
  X_test.to_csv("data/X_test.csv",   index=False)
  y_train.to_csv("data/y_train.csv", index=False)
  y_test.to_csv("data/y_test.csv",   index=False)

  Option B — Check your preprocessing script for this mistake:
  # WRONG — this maps <30 to 0 if you used pd.get_dummies or LabelEncoder
  df["readmitted"] = LabelEncoder().fit_transform(df["readmitted"])
  # LabelEncoder sorts alphabetically: '<30'=0, '>30'=1, 'NO'=2

  # RIGHT — explicit binary mapping
  df["readmitted"] = (df["readmitted"] == "<30").astype(int)
    """)

print("\n✅  Diagnosis complete. Re-run model_evaluation.py after fixing.\n")