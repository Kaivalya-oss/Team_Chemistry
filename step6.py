# ============================================================
# STEP 6: LINEAR REGRESSION MODELING
# Project: Team Chemistry in Football
# Goal: Fit linear model + evaluate baseline performance
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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

print("Train shape:", X_train.shape)
print("Test shape: ", X_test.shape)

# ─── 2. Train Linear Regression ─────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ─── 3. Make predictions ────────────────────────────────────────────
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

# ─── 4. Performance metrics ─────────────────────────────────────────
def print_metrics(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{dataset_name} Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.3f}")

print_metrics(y_train, y_pred_train, "Train")
print_metrics(y_test,  y_pred_test,  "Test")

# ─── 5. Coefficients (feature importance) ───────────────────────────
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print("\nFeature Coefficients (higher = stronger positive impact):")
print(coef_df.round(3))

intercept = model.intercept_
print(f"\nIntercept: {intercept:.2f}")

# ─── 6. Visual: Actual vs Predicted (Test set) ──────────────────────
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Chemistry Index (0-100)")
plt.ylabel("Predicted Chemistry Index")
plt.title("Linear Regression: Actual vs Predicted (Test Set)")
plt.tight_layout()
plt.show()

# ─── 7. Quick interpretation ────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6 TAKEAWAYS")
print("="*60)
print("• R² on test set shows how much variance the 4 features explain")
print("• Coefficients reveal which proxy (e.g. NationalityDiversity) matters most")
print("• If RMSE is low and R² > 0.6–0.7 → features explain index well")
print("• If train R² much higher than test → slight overfitting (normal for linear)")
print("Next: Step 7 – try Random Forest / Gradient Boosting for comparison")