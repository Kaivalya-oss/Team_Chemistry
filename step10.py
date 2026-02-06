# ============================================================
# STEP 10: CROSS-VALIDATION
# Project: Team Chemistry in Football
# Goal: Get robust performance estimates beyond single split
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score, root_mean_squared_error, mean_absolute_error

# ─── 1. Load full original data (not the split versions) ───────
# Use same features & target as in previous steps
df = pd.read_csv("teams_with_chemistry_index.csv")

features = [
    'NationalityDiversity_norm',
    'WorkRate_Balance_norm',
    'International Reputation_norm',
    'Age_closeness_norm'
]

X = df[features]
y = df['Chemistry_Index_100']

print(f"Full data shape: {X.shape}")

# ─── 2. Define models ───────────────────────────────────────────
# IMPORTANT: Replace example params with your ACTUAL tuned values from Step 8!
models = {
    'Linear Regression': LinearRegression(),
    
    'Random Forest (tuned)': RandomForestRegressor(
        n_estimators=200,           # ← use your real tuned value
        max_depth=None,             # ← from your Step 8 output
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),
    
    'Gradient Boosting (tuned)': GradientBoostingRegressor(
        n_estimators=200,           # ← use your real tuned value
        learning_rate=0.1,          # ← from your Step 8 output
        max_depth=3,
        subsample=0.8,
        min_samples_split=5,
        random_state=42
    )
}

# ─── 3. Cross-validation settings ────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Modern scorers (no 'squared' parameter needed)
scoring = {
    'r2': 'r2',
    'rmse': make_scorer(root_mean_squared_error, greater_is_better=False),
    'mae': make_scorer(mean_absolute_error, greater_is_better=False)
}

# ─── 4. Run cross-validation for each model ──────────────────────
results = []

for name, model in models.items():
    print(f"\nRunning {name}...")
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Aggregate results
    res = {
        'Model': name,
        'CV Mean R²':     cv_results['test_r2'].mean().round(4),
        'CV Std R²':      cv_results['test_r2'].std().round(4),
        'CV Mean RMSE':   (-cv_results['test_rmse']).mean().round(4),  # negate because scorer is negative
        'CV Std RMSE':    (-cv_results['test_rmse']).std().round(4),
        'CV Mean MAE':    (-cv_results['test_mae']).mean().round(4),
        'Mean Train R²':  cv_results['train_r2'].mean().round(4)
    }
    results.append(res)

# ─── 5. Summary table ────────────────────────────────────────────
cv_df = pd.DataFrame(results)

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS (5-fold)")
print("="*70)
print(cv_df.sort_values('CV Mean R²', ascending=False).to_string(index=False))

# Reminder of your previous single-split test scores
print("\nPrevious single test scores (for comparison):")
print(" Tuned GB  → R² 0.9832 | RMSE 1.2613")
print(" Tuned RF  → R² 0.9402 | RMSE 2.3798")
print(" Linear    → R² 1.0000 | RMSE 0.0000  (suspicious – possible leakage?)")