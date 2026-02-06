# ============================================================
# STEP 8: HYPERPARAMETER TUNING
# Project: Team Chemistry in Football
# Goal: Find optimal settings for RF & GB using Grid Search + CV
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ─── 1. Load train & test sets ──────────────────────────────────────
train = pd.read_csv("train_set.csv")
test  = pd.read_csv("test_set.csv")

X_train = train.drop(columns=['Chemistry_Index_100'])
y_train = train['Chemistry_Index_100']

X_test  = test.drop(columns=['Chemistry_Index_100'])
y_test  = test['Chemistry_Index_100']

features = X_train.columns.tolist()

# ─── 2. Define parameter grids ──────────────────────────────────────
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5],
    'subsample': [0.8, 1.0]
}

# ─── 3. Grid Search for Random Forest ───────────────────────────────
print("Tuning Random Forest...")
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)

print("\nBest RF Params:", rf_grid.best_params_)
print("Best RF CV R²:", rf_grid.best_score_.round(4))

# Evaluate best RF on test set
best_rf = rf_grid.best_estimator_
y_pred_rf_test = best_rf.predict(X_test)
rf_test_r2 = r2_score(y_test, y_pred_rf_test)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
print(f"Best RF Test R²:   {rf_test_r2:.4f}")
print(f"Best RF Test RMSE: {rf_test_rmse:.4f}")

# ─── 4. Grid Search for Gradient Boosting ───────────────────────────
print("\nTuning Gradient Boosting...")
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
gb_grid.fit(X_train, y_train)

print("\nBest GB Params:", gb_grid.best_params_)
print("Best GB CV R²:", gb_grid.best_score_.round(4))

# Evaluate best GB on test set
best_gb = gb_grid.best_estimator_
y_pred_gb_test = best_gb.predict(X_test)
gb_test_r2 = r2_score(y_test, y_pred_gb_test)
gb_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb_test))
print(f"Best GB Test R²:   {gb_test_r2:.4f}")
print(f"Best GB Test RMSE: {gb_test_rmse:.4f}")

# ─── 5. Summary comparison ──────────────────────────────────────────
print("\n" + "="*60)
print("TUNED MODEL COMPARISON (Test Set)")
print("="*60)
print(f"{'Model':<20} {'R²':<12} {'RMSE':<12}")
print("-"*60)
print(f"{'Random Forest (tuned)':<20} {rf_test_r2:<12.4f} {rf_test_rmse:<12.4f}")
print(f"{'Gradient Boosting (tuned)':<20} {gb_test_r2:<12.4f} {gb_test_rmse:<12.4f}")

# Compare to previous untuned versions (manual entry)
print("\nReminder: Untuned RF/GB from Step 7 had similar/higher values?")
print("If tuned models are better → tuning helped. If similar → original params were already good.")

# ─── 6. Optional: Feature importance from best models ───────────────
print("\nBest RF Feature Importance:")
rf_imp = pd.Series(best_rf.feature_importances_, index=features).sort_values(ascending=False)
print(rf_imp.round(4))

print("\nBest GB Feature Importance:")
gb_imp = pd.Series(best_gb.feature_importances_, index=features).sort_values(ascending=False)
print(gb_imp.round(4))

print("\nNext: Step 9 – final results comparison | Step 10 – cross-validation | Step 11 – interpretation")

# ─── 7. Save the best model ─────────────────────────────────────────
import pickle
model_filename = "best_gb_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(best_gb, f)
print(f"\nSaved best model to: {model_filename}")

# ─── 8. Calculate and print min/max for Step 11 (Normalization) ─────
print("\n" + "="*60)
print("NORMALIZATION PARAMETERS FOR STEP 11")
print("="*60)
try:
    # Re-load original data to get min/max used in Step 4
    # (We need to replicate the exact feature engineering from Step 4)
    raw_df = pd.read_csv("team_level_processed.csv")
    
    # Re-create derived features
    raw_df['WorkRate_Balance'] = 1 / (1 + abs(raw_df['Attack_Work_Intensity'] - raw_df['Defense_Work_Intensity']))
    optimal_age = 26.5
    raw_df['Age_closeness'] = 1 - abs(raw_df['Age'] - optimal_age) / raw_df['Age'].std()

    # The 4 features to check
    # Note: 'Age_closeness' here is BEFORE normalization (0-1), but Step 4 normalized it AGAIN?
    # Wait, Step 4 normalized:
    # 1. NationalityDiversity
    # 2. WorkRate_Balance
    # 3. International Reputation
    # 4. Age_closeness (which was already 'semantically' normalized but then MinMaxed again)
    
    norm_cols = {
        'div': 'NationalityDiversity',
        'bal': 'WorkRate_Balance',
        'rep': 'International Reputation',
        'agec': 'Age_closeness'
    }
    
    print("Replace these values in step11.py > 'mins' and 'maxs':")
    print("mins = {")
    for key, col in norm_cols.items():
        print(f"    '{key}': {raw_df[col].min():.4f},")
    print("}")
    
    print("maxs = {")
    for key, col in norm_cols.items():
        print(f"    '{key}': {raw_df[col].max():.4f},")
    print("}")

except Exception as e:
    print(f"Could not calculate min/max values automatically: {e}")
    print("Please check step4.py or manual data inspection.")