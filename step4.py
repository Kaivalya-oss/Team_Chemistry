# ============================================================
# STEP 4: TARGET VARIABLE CONSTRUCTION (SIMPLIFIED)
# Project: Team Chemistry in Football
# Goal: Create ONE equal-weight Chemistry Index (0–100)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ─── 1. Load the processed team-level data ──────────────────────────
df = pd.read_csv("team_level_processed.csv")

print("Loaded data shape:", df.shape)

# ─── 2. Define the 4 shortlisted features ───────────────────────────
features = [
    'NationalityDiversity',
    'Attack_Work_Intensity',
    'Defense_Work_Intensity',
    'Age',
    'International Reputation'
]

# ─── 3. Create work-rate balance component (higher = better) ────────
df['WorkRate_Balance'] = 1 / (1 + abs(df['Attack_Work_Intensity'] - df['Defense_Work_Intensity']))

# ─── 4. Prepare components for the index ────────────────────────────
# We will use these four:
# 1. NationalityDiversity          → higher better
# 2. WorkRate_Balance              → higher better
# 3. International Reputation      → higher better
# 4. Age closeness to ~26.5        → closer better

optimal_age = 26.5
df['Age_closeness'] = 1 - abs(df['Age'] - optimal_age) / df['Age'].std()   # simple normalization around optimal

# Select final 4 components (all should be "higher = better")
components = [
    'NationalityDiversity',
    'WorkRate_Balance',
    'International Reputation',
    'Age_closeness'
]

# ─── 5. Normalize each to 0–1 range ────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_components = []
for col in components:
    scaled_name = f"{col}_norm"
    df[scaled_name] = scaler.fit_transform(df[[col]])
    scaled_components.append(scaled_name)

# ─── 6. Equal-weight Chemistry Index (0–100) ───────────────────────
df['Chemistry_Index_100'] = df[scaled_components].mean(axis=1) * 100

# ─── 7. Show results ────────────────────────────────────────────────
print("\n" + "="*70)
print("Teams sorted by Chemistry Index (0–100) — Top 12")
print("="*70)

show_cols = ['Club', 'Overall', 'NationalityDiversity', 'WorkRate_Balance',
             'International Reputation', 'Age', 'Chemistry_Index_100']

print(df[show_cols].sort_values('Chemistry_Index_100', ascending=False).head(12).round(2))

print("\nBottom 8 teams by Chemistry Index:")
print(df[show_cols].sort_values('Chemistry_Index_100').head(8).round(2))

# ─── 8. Quick validation ────────────────────────────────────────────
print("\nAverage Chemistry Index:")
print(f"  Overall dataset:      {df['Chemistry_Index_100'].mean():.1f}")
print(f"  Top 25 teams:         {df.nlargest(25, 'Overall')['Chemistry_Index_100'].mean():.1f}")
print(f"  Bottom 25 teams:      {df.nsmallest(25, 'Overall')['Chemistry_Index_100'].mean():.1f}")

print("\nCorrelation with Overall rating:")
print(df[['Chemistry_Index_100', 'Overall']].corr().round(3))

# ─── 9. Save result ─────────────────────────────────────────────────
output_file = "teams_with_chemistry_index.csv"
df.to_csv(output_file, index=False)
print(f"\nSaved → {output_file}")
print("Final column: Chemistry_Index_100 (0–100)")