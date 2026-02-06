# ============================================================
# STEP 5: TRAIN-TEST SPLIT
# Project: Team Chemistry in Football
# Goal: Create train/test sets for modeling the Chemistry Index
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split

# ─── 1. Load the data with chemistry index ──────────────────────────
df = pd.read_csv("teams_with_chemistry_index.csv")

print("Loaded shape:", df.shape)

# ─── 2. Define target (what we want to predict) ─────────────────────
target = 'Chemistry_Index_100'

# ─── 3. Choose features (X) ─────────────────────────────────────────
# Use the 4 normalized components as main predictors
# (they are already scaled 0–1 and independent-ish from raw quality)
features = [
    'NationalityDiversity_norm',
    'WorkRate_Balance_norm',
    'International Reputation_norm',
    'Age_closeness_norm'
]

# Optional: add a few controls / quality proxies if you want
# (uncomment if you want to include them in modeling)
# features += ['Overall', 'SquadSize', 'Value', 'Wage']

X = df[features]
y = df[target]

# Quick check for missing values (should be none after previous steps)
print("\nMissing values in features:", X.isna().sum().sum())
print("Missing values in target:", y.isna().sum())

# ─── 4. Perform the split ───────────────────────────────────────────
# 80/20 split is standard; random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,          # 20% for testing
    random_state=42,         # same split every time you run
    shuffle=True             # randomize (good default)
)

# ─── 5. Print summary ───────────────────────────────────────────────
print("\n" + "="*60)
print("TRAIN-TEST SPLIT SUMMARY")
print("="*60)
print(f"Total samples:     {len(df):>5}")
print(f"Train samples:     {len(X_train):>5}  ({len(X_train)/len(df):.1%})")
print(f"Test samples:      {len(X_test):>5}  ({len(X_test)/len(df):.1%})")
print(f"Features used:     {len(features)}")
print("Features:", features)

# Optional: quick stats comparison (to check no big leakage/bias)
print("\nTarget mean in train vs test:")
print(f"  Train: {y_train.mean():.2f}")
print(f"  Test:  {y_test.mean():.2f}")

# ─── 6. Save the splits (optional but useful for later steps) ───────
# You can save them as separate CSV files or just keep in memory
train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
test_df  = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

train_df.to_csv("train_set.csv", index=False)
test_df.to_csv("test_set.csv", index=False)

print("\nSaved → train_set.csv and test_set.csv")
print("You can now move to Step 6: modeling (Linear / Logistic Regression, etc.)")