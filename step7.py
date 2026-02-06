# ============================================================
# STEP 7: RANDOM FOREST & GRADIENT BOOSTING
# Project: Team Chemistry in Football
# Goal: Compare tree-based models to linear baseline
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ─── 1. Load train & test sets ──────────────────────────────────────
train = pd.read_csv("train_set.csv")
test  = pd.read_csv("test_set.csv")

X_train = train.drop(columns=['Chemistry_Index_100'])
y_train = train['Chemistry_Index_100']

X_test  = test.drop(columns=['Chemistry_Index_100'])
y_test  = test['Chemistry_Index_100']

features = X_train.columns.tolist()

# ─── 2. Define models ───────────────────────────────────────────────
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
}

# ─── 3. Train & evaluate each model ─────────────────────────────────
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2   = r2_score(y_train, y_pred_train)
    test_r2    = r2_score(y_test, y_pred_test)
    
    results.append({
        'Model': name,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R²': train_r2,
        'Test R²': test_r2
    })
    
    # Feature importance (for tree models)
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=features)
        imp = imp.sort_values(ascending=False)
        print(f"\n{name} Feature Importance:")
        print(imp.round(4))
    
    # Plot actual vs predicted (test set)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Chemistry Index")
    plt.ylabel("Predicted")
    plt.title(f"{name}: Actual vs Predicted (Test Set)")
    plt.tight_layout()
    plt.show()

# ─── 4. Summary table ───────────────────────────────────────────────
results_df = pd.DataFrame(results).round(3)
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(results_df)

# Compare to Linear Regression from Step 6 (manual entry or re-run if needed)
print("\nQuick reminder from Linear Regression (Step 6):")
print("  Test RMSE ≈ [your value from step 6]")
print("  Test R²   ≈ [your value from step 6]")

# ─── 5. Takeaways ───────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7 TAKEAWAYS")
print("="*60)
print("• If Test R² > Linear Regression → trees capture better patterns")
print("• If Test RMSE much lower → non-linear effects or interactions exist")
print("• Feature importance shows which proxy really drives predictions")
print("• If RF/GB overfit (Train R² >> Test R²) → tune max_depth or add regularization")
print("Next steps: Hyperparameter tuning (Step 8), cross-validation (Step 10), interpretation (Step 11)")